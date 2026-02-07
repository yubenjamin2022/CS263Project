import argparse
import json
from typing import Dict, Any, List

from vllm import LLM, SamplingParams


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no}: {e}") from e
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="JSONL with fields: prompt (str) and optional id")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallel")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=None, help="Override context length if needed")
    ap.add_argument("--batch_size", type=int, default=None, help="Optional micro-batch size for generate()")
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)
    prompts = []
    ids = []
    for i, r in enumerate(rows):
        if "prompt" not in r:
            raise ValueError(f"Row {i} missing 'prompt' key.")
        prompts.append(r["prompt"])
        ids.append(r.get("id", i))

    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,  # vLLM's eager mode is usually faster for small batches and short generations
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    # vLLM returns outputs aligned with prompts order
    if args.batch_size is None:
        outputs = llm.generate(prompts, sampling)
    else:
        # micro-batching to control memory
        outputs = []
        for i in range(0, len(prompts), args.batch_size):
            outputs.extend(llm.generate(prompts[i:i+args.batch_size], sampling))

    out_rows = []
    for pid, prompt, out in zip(ids, prompts, outputs):
        # vLLM can return multiple candidates; we take the first.
        gen_text = out.outputs[0].text if out.outputs else ""
        out_rows.append(
            {
                "id": pid,
                "prompt": prompt,
                "generation": gen_text,
            }
        )

    write_jsonl(args.output_jsonl, out_rows)
    print(f"Wrote {len(out_rows)} generations to {args.output_jsonl}")


if __name__ == "__main__":
    main()
