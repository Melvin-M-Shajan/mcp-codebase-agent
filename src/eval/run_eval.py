"""§9.2/§9.3 eval runner -- runs the full agent graph against every question in
eval/questions.jsonl, scores each run with RAGAS, traces every run with Langfuse (one
session groups the whole eval run), and writes eval/EVAL_REPORT.md.

No fabricated scores: every number in the output report comes from an actual agent run
against the live MCP server + Pinecone index + Gemini, and an actual RAGAS scoring pass.
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

for _noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "huggingface_hub", "pinecone"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from src.agent.graph import build_graph, new_state
from src.agent.llm import get_answer_llm, get_llm, get_planner_llm
from src.agent.mcp_client import mcp_session
from src.rag.embeddings import get_embeddings_client


def load_questions(path: str) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def extract_retrieved_contexts(final_state: dict) -> list[str]:
    contexts = [hit["snippet"] for hit in final_state["retrieved_context"]]
    for call in final_state["tool_calls"]:
        result_str = json.dumps(call["result"], default=str)[:3000]
        contexts.append(f"Evidence from the {call['tool']} tool, called with {call['args']}: {result_str}")
    return contexts or ["(no context retrieved)"]


def files_touched(final_state: dict) -> set[str]:
    touched: set[str] = set()
    for hit in final_state["retrieved_context"]:
        if hit.get("file_path"):
            touched.add(hit["file_path"])
    for call in final_state["tool_calls"]:
        args = call.get("args") or {}
        if args.get("path"):
            touched.add(args["path"])
        if call["tool"] == "search_code" and isinstance(call["result"], list):
            for item in call["result"]:
                if isinstance(item, dict) and item.get("file_path"):
                    touched.add(item["file_path"])
    return touched


def file_grounding_hit(final_state: dict, reference_files: list[str]) -> bool | None:
    if not reference_files:
        return None
    touched = files_touched(final_state)
    return any(ref in touched for ref in reference_files)


async def run_agent_for_question(
    session, planner_llm, answer_llm, question_obj: dict, langfuse_handler, session_id: str
) -> dict:
    graph = build_graph(session, planner_llm, answer_llm)
    state = new_state(question_obj["question"])
    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_tags": [question_obj["id"]],
        },
        "run_name": f"eval-{question_obj['id']}",
    }
    start = time.monotonic()
    final_state = await graph.ainvoke(state, config=config)
    elapsed = time.monotonic() - start
    return {"question_obj": question_obj, "final_state": final_state, "elapsed_seconds": elapsed}


async def run_all_questions(questions: list[dict], max_steps: int = 6) -> tuple[list[dict], str]:
    load_dotenv()
    planner_llm = get_planner_llm()
    answer_llm = get_answer_llm()
    langfuse_handler = CallbackHandler()
    session_id = f"eval-run-{uuid.uuid4().hex[:8]}"

    results = []
    async with mcp_session() as session:
        for q in questions:
            print(f"  running {q['id']}: {q['question'][:70]}...", flush=True)
            result = await run_agent_for_question(session, planner_llm, answer_llm, q, langfuse_handler, session_id)
            results.append(result)

    get_client().flush()
    return results, session_id


def score_with_ragas(results: list[dict]) -> Any:
    # Imported here, not at module level: ragas.executor calls nest_asyncio.apply() as an
    # import side effect, which breaks anyio's event-loop detection inside mcp_session's
    # subprocess handling if it happens before that runs. Scoring only starts after all
    # agent runs (and their MCP subprocess work) are already complete, so it's safe here.
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    ragas_llm = LangchainLLMWrapper(get_llm(temperature=0.0))
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings_client())

    samples = []
    for r in results:
        state = r["final_state"]
        q = r["question_obj"]
        samples.append(
            SingleTurnSample(
                user_input=q["question"],
                response=state["answer"] or "",
                retrieved_contexts=extract_retrieved_contexts(state),
                reference=q["reference_answer"],
            )
        )
    dataset = EvaluationDataset(samples=samples)

    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
    return evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)


def build_report(results: list[dict], ragas_result: Any, session_id: str) -> str:
    scores_df = ragas_result.to_pandas()

    rows = []
    for i, r in enumerate(results):
        q = r["question_obj"]
        state = r["final_state"]
        row = {
            "id": q["id"],
            "question": q["question"],
            "requires_multistep": q.get("requires_multistep", False),
            "num_tool_calls": len(state["tool_calls"]),
            "faithfulness": float(scores_df.iloc[i]["faithfulness"]),
            "answer_relevancy": float(scores_df.iloc[i]["answer_relevancy"]),
            "context_precision": float(scores_df.iloc[i]["context_precision"]),
            "file_grounding_rate": file_grounding_hit(state, q.get("reference_files", [])),
            "answer": state["answer"] or "",
            "tool_calls": state["tool_calls"],
            "elapsed_seconds": r["elapsed_seconds"],
        }
        rows.append(row)

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else float("nan")

    grounding_vals = [r["file_grounding_rate"] for r in rows if r["file_grounding_rate"] is not None]
    grounding_rate = sum(1 for v in grounding_vals if v) / len(grounding_vals) if grounding_vals else float("nan")

    multistep_count = sum(1 for r in rows if r["num_tool_calls"] >= 2)

    lines = []
    lines.append("# Eval Report\n")
    lines.append(f"Langfuse session: `{session_id}`\n")
    lines.append(f"Questions run: {len(rows)}. Questions with num_tool_calls >= 2: {multistep_count}.\n")

    lines.append("## Per-question scores\n")
    lines.append(
        "| id | multistep? | tool calls | faithfulness | answer_relevancy | context_precision | file_grounding |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        grounding_cell = "n/a" if r["file_grounding_rate"] is None else ("yes" if r["file_grounding_rate"] else "no")
        lines.append(
            f"| {r['id']} | {r['requires_multistep']} | {r['num_tool_calls']} | "
            f"{r['faithfulness']:.3f} | {r['answer_relevancy']:.3f} | {r['context_precision']:.3f} | "
            f"{grounding_cell} |"
        )

    lines.append("\n## Aggregate means\n")
    lines.append(f"- faithfulness: {mean('faithfulness'):.3f}")
    lines.append(f"- answer_relevancy: {mean('answer_relevancy'):.3f}")
    lines.append(f"- context_precision: {mean('context_precision'):.3f}")
    lines.append(f"- file_grounding_rate: {grounding_rate:.3f}")
    lines.append(f"- mean num_tool_calls: {mean('num_tool_calls'):.2f}")

    lines.append("\n## Annotated failure cases\n")
    worst = sorted(rows, key=lambda r: r["faithfulness"] + r["answer_relevancy"] + r["context_precision"])[:3]
    for r in worst:
        lines.append(f"### {r['id']}: {r['question']}")
        lines.append(
            f"faithfulness={r['faithfulness']:.3f}, answer_relevancy={r['answer_relevancy']:.3f}, "
            f"context_precision={r['context_precision']:.3f}, file_grounding={r['file_grounding_rate']}\n"
        )
        lines.append("Tool-call trace:")
        if r["tool_calls"]:
            for call in r["tool_calls"]:
                lines.append(f"- `{call['tool']}({call['args']})`")
        else:
            lines.append("- (no tool calls -- answered from RAG context alone)")
        lines.append(f"\nAgent answer:\n> {r['answer'][:800]}\n")

    return "\n".join(lines)


async def main_async(questions_path: str, out_path: str) -> None:
    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    results, session_id = await run_all_questions(questions)
    print("All agent runs complete. Scoring with RAGAS...")

    ragas_result = score_with_ragas(results)
    report = build_report(results, ragas_result, session_id)

    Path(out_path).write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--out", default="eval/EVAL_REPORT.md")
    args = parser.parse_args()

    asyncio.run(main_async(args.questions, args.out))


if __name__ == "__main__":
    main()
