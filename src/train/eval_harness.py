from __future__ import annotations

import json
import re

from src.config import get_base_model_id, get_settings
from src.logging import get_logger

logger = get_logger(__name__)


def _load_repo_symbols(slug: str) -> set[str]:
    settings = get_settings()
    chunks_path = settings.data_dir / slug / "chunks.jsonl"
    symbols: set[str] = set()
    if chunks_path.exists():
        with open(chunks_path) as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    sym = c.get("symbol")
                    if sym:
                        symbols.add(sym)
                        for part in re.split(r"[._]", sym):
                            if len(part) >= 3:
                                symbols.add(part)
    return symbols


def _style_match_score(answer: str, repo_symbols: set[str]) -> float:
    if not repo_symbols or not answer:
        return 0.0
    answer_lower = answer.lower()
    matched = sum(1 for s in repo_symbols if s.lower() in answer_lower)
    return min(matched / max(1, len(repo_symbols) * 0.01), 1.0)


def _groundedness_score(answer: str, repo_symbols: set[str]) -> float:
    if not answer:
        return 0.0
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", answer)
    if not tokens:
        return 1.0
    repo_lower = {s.lower() for s in repo_symbols}
    grounded = sum(1 for t in tokens if t.lower() in repo_lower)
    return grounded / len(tokens)


def _generate_base(model_id: str, prompt: str, max_tokens: int = 256) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _generate_ft(adapter_path: str, model_id: str, prompt: str, max_tokens: int = 256) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_eval(slug: str, max_samples: int = 10) -> dict:
    from src.train.dataset import load_raw_pairs

    settings = get_settings()
    adapter_dir = settings.data_dir / slug / "adapters"
    eval_path = settings.data_dir / slug / "eval.jsonl"

    if not eval_path.exists():
        logger.warning("No eval.jsonl for %s", slug)
        return {"error": "no eval data"}

    _, eval_raw = load_raw_pairs(slug)
    if not eval_raw:
        return {"error": "no eval pairs"}

    samples = eval_raw[:max_samples]
    repo_symbols = _load_repo_symbols(slug)

    results: list[dict] = []
    base_total = 0.0
    ft_total = 0.0

    try:
        for sample in samples:
            instruction = sample["instruction"]
            reference = sample["response"]

            prompt = (
                f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

            try:
                base_answer = _generate_base(get_base_model_id(), prompt)
            except Exception as e:
                logger.warning("Base generation failed: %s", e)
                base_answer = "(generation failed)"

            try:
                ft_answer = _generate_ft(str(adapter_dir), get_base_model_id(), prompt)
            except Exception as e:
                logger.warning("FT generation failed: %s", e)
                ft_answer = "(generation failed)"

            base_style = _style_match_score(base_answer, repo_symbols)
            ft_style = _style_match_score(ft_answer, repo_symbols)
            base_ground = _groundedness_score(base_answer, repo_symbols)
            ft_ground = _groundedness_score(ft_answer, repo_symbols)

            base_score = (base_style + base_ground) / 2
            ft_score = (ft_style + ft_ground) / 2
            base_total += base_score
            ft_total += ft_score

            results.append({
                "instruction": instruction[:200],
                "reference": reference[:200],
                "base_answer": base_answer[:300],
                "ft_answer": ft_answer[:300],
                "base_style_match": round(base_style, 3),
                "ft_style_match": round(ft_style, 3),
                "base_groundedness": round(base_ground, 3),
                "ft_groundedness": round(ft_ground, 3),
                "base_score": round(base_score, 3),
                "ft_score": round(ft_score, 3),
            })

    except Exception as e:
        logger.error("Eval harness failed: %s", e)
        return {"error": str(e)}

    n = len(results)
    report = {
        "num_samples": n,
        "base_avg_score": round(base_total / max(1, n), 3),
        "ft_avg_score": round(ft_total / max(1, n), 3),
        "delta": round((ft_total - base_total) / max(1, n), 3),
        "ft_beats_base": ft_total > base_total,
        "per_example": results,
    }

    report_path = settings.data_dir / slug / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Eval report written to %s", report_path)
    return report
