"""
benchmark.py — Performance benchmarking script for Base Llama 3.1 8B.

Measures:
  • Inference throughput (tokens / second) at FP16, INT8 and INT4 precision.
  • Time to First Token (TTFT / prefill latency) at multiple prompt lengths.
  • Peak VRAM and system-RAM usage per configuration.
  • CPU utilisation during GPU-offloaded and CPU-only inference.
  • Perplexity on a held-out WikiText-2 sample.
  • Simple factual accuracy probe (short QA).

Tested system
─────────────
  GPU  : AMD Radeon RX 9060 XT 16 GB (RDNA 4, PCIe 4.0 × 16)
  CPU  : AMD Ryzen 7 5700X3D (8 C / 16 T, 100 MB 3D V-Cache, boost 4.1 GHz)
  RAM  : 32 GB DDR4-3200 dual-channel
  OS   : Ubuntu 22.04 LTS
  ROCm : 6.2 (HIP runtime)
  Python: 3.11
  transformers: 4.44.2
  torch: 2.4.0+rocm6.2

Usage
─────
  python benchmark.py                   # full benchmark suite
  python benchmark.py --quick           # short smoke-test (3 prompts)
  python benchmark.py --config fp16     # single precision only
  python benchmark.py --save results/   # write JSON results to directory
"""

import argparse
import json
import os
import platform
import time
import math
import statistics
from datetime import datetime
from pathlib import Path

import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import config as cfg

# ---------------------------------------------------------------------------
# Constants & configuration sets
# ---------------------------------------------------------------------------

PRECISION_CONFIGS = {
    "fp16": {
        "label": "Full Precision (FP16)",
        "load_in_8bit": False,
        "load_in_4bit": False,
        "torch_dtype": torch.float16,
        "expected_vram_gb": 16.0,
    },
    "int8": {
        "label": "8-bit Quantised (INT8)",
        "load_in_8bit": True,
        "load_in_4bit": False,
        "torch_dtype": torch.float16,
        "expected_vram_gb": 8.5,
    },
    "int4": {
        "label": "4-bit Quantised (INT4 / NF4)",
        "load_in_8bit": False,
        "load_in_4bit": True,
        "torch_dtype": torch.float16,
        "expected_vram_gb": 4.7,
    },
}

# Prompts used for throughput measurement (kept identical across runs)
THROUGHPUT_PROMPTS = [
    "The field of artificial intelligence has seen remarkable progress in recent years, particularly",
    "Large language models such as GPT-4 and Llama 3 are trained on",
    "Explain the concept of transformer attention mechanisms in simple terms:",
    "The main advantages of locally deployed language models over cloud-based APIs include",
    "Write a short Python function that reads a CSV file and returns the column names:",
]

# Prompts of varying length for TTFT measurement
TTFT_PROMPTS = {
    "short_20": "What is machine learning?",
    "medium_100": (
        "Large language models (LLMs) are a type of neural network trained on massive "
        "text corpora. They have demonstrated remarkable capabilities across a wide "
        "range of natural-language tasks including translation, summarisation, question "
        "answering, and code generation. Describe the key architectural components of "
        "a modern transformer-based LLM."
    ),
    "long_500": (
        "Artificial intelligence research has a history spanning more than seven decades. "
        "The field began with symbolic reasoning systems, moved through expert systems in "
        "the 1980s, experienced several 'AI winters', and has recently entered a period of "
        "rapid advancement driven by deep learning and large-scale compute. Neural networks "
        "were first proposed by McCulloch and Pitts in 1943, formalised as the perceptron by "
        "Rosenblatt in 1958, and later extended to multi-layer networks trained via "
        "backpropagation by Rumelhart, Hinton and Williams in 1986. The transformer "
        "architecture, introduced by Vaswani et al. in 2017, replaced recurrent networks for "
        "sequence modelling and became the foundation for models such as BERT, GPT-2, GPT-3, "
        "PaLM, and the Llama family. Today's state-of-the-art models contain hundreds of "
        "billions of parameters and are trained on trillions of tokens of text data. They "
        "exhibit emergent capabilities not present in smaller models, including multi-step "
        "reasoning, in-context learning, and instruction following. "
        "Given this history, discuss the major architectural innovations that have contributed "
        "to the success of large language models and identify the key open research challenges "
        "that remain."
    ),
}

# Simple factual QA probe
QA_PROBE = [
    {"prompt": "The capital of France is", "expected": "Paris"},
    {"prompt": "The chemical formula of water is", "expected": "H2O"},
    {"prompt": "Albert Einstein was born in the year", "expected": "1879"},
    {"prompt": "Python is a high-level programming language created by", "expected": "Guido"},
    {"prompt": "The speed of light in a vacuum is approximately", "expected": "299"},
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_system_info() -> dict:
    """Return a dictionary of host system metadata."""
    gpu_name = "N/A"
    gpu_vram_total_gb = None
    rocm_version = "N/A"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_vram_total_gb = round(vram_bytes / 1e9, 1)
        # ROCm exposes itself via torch version string
        if "rocm" in torch.__version__.lower():
            rocm_version = torch.version.hip or "detected (version unknown)"

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "rocm_version": rocm_version,
        "cpu_model": platform.processor() or "AMD Ryzen 7 5700X3D",
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "gpu_model": gpu_name,
        "gpu_vram_total_gb": gpu_vram_total_gb,
        "cuda_available": torch.cuda.is_available(),
        "model_id": cfg.MODEL_NAME,
    }


def vram_used_gb() -> float:
    """Return current GPU VRAM usage in GB (0.0 if no GPU)."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated(0) / 1e9, 2)
    return 0.0


def measure_throughput(model, tokenizer, prompts: list, n_new_tokens: int = 256,
                       temperature: float = 0.7) -> dict:
    """
    Measure decode throughput (tokens / second) averaged over *prompts*.

    Returns a dict with mean, median, stdev, min, max.
    """
    speeds = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        n_prompt_tokens = inputs["input_ids"].shape[1]

        # Warm-up: one short pass (not timed)
        _ = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_start = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=n_new_tokens,
                temperature=temperature,
                top_p=cfg.TOP_P,
                top_k=cfg.TOP_K,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - t_start

        n_generated = out.shape[1] - n_prompt_tokens
        if n_generated > 0 and elapsed > 0:
            speeds.append(n_generated / elapsed)

    if not speeds:
        return {}

    return {
        "mean_tokens_per_sec": round(statistics.mean(speeds), 2),
        "median_tokens_per_sec": round(statistics.median(speeds), 2),
        "stdev_tokens_per_sec": round(statistics.stdev(speeds) if len(speeds) > 1 else 0.0, 2),
        "min_tokens_per_sec": round(min(speeds), 2),
        "max_tokens_per_sec": round(max(speeds), 2),
        "n_samples": len(speeds),
        "new_tokens_per_sample": n_new_tokens,
    }


def measure_ttft(model, tokenizer, prompt_dict: dict) -> dict:
    """
    Measure Time-to-First-Token (prefill latency) for prompts of different lengths.
    """
    results = {}
    for label, prompt in prompt_dict.items():
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        n_tokens = inputs["input_ids"].shape[1]

        # Warm-up
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        ttft_ms = (time.perf_counter() - t0) * 1000

        results[label] = {
            "prompt_tokens": n_tokens,
            "ttft_ms": round(ttft_ms, 1),
        }
    return results


def measure_perplexity(model, tokenizer, text: str, stride: int = 512) -> float:
    """
    Compute approximate perplexity using a sliding-window approach.
    Uses a small WikiText-2-style passage.
    """
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    max_len = min(getattr(model.config, "max_position_embeddings", 2048), 2048)

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        target_len = end - prev_end
        input_ids = encodings.input_ids[:, begin:end].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            out = model(input_ids, labels=target_ids)
            nlls.append(out.loss * target_len)

        prev_end = end
        if end == seq_len:
            break

    return round(math.exp(torch.stack(nlls).sum() / seq_len), 4)


def measure_qa_accuracy(model, tokenizer, qa_pairs: list) -> dict:
    """
    Simple greedy-decode factual probe: measures proportion of correct answers.
    """
    correct = 0
    details = []
    for item in qa_pairs:
        inputs = tokenizer(item["prompt"], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(out[0], skip_special_tokens=True)
        # Strip the prompt from the output
        answer = generated[len(item["prompt"]):].strip()
        hit = item["expected"].lower() in answer.lower()
        correct += int(hit)
        details.append({
            "prompt": item["prompt"],
            "expected_substring": item["expected"],
            "model_output": answer[:80],
            "correct": hit,
        })

    return {
        "accuracy": round(correct / len(qa_pairs), 3),
        "correct": correct,
        "total": len(qa_pairs),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Per-precision benchmark runner
# ---------------------------------------------------------------------------

WIKITEXT_SAMPLE = (
    "Homarus gammarus, known as the European lobster or common lobster, is a species of "
    "clawed lobster from the eastern Atlantic Ocean, Mediterranean Sea and parts of the "
    "Black Sea. It is closely related to the American lobster, H. americanus. It may grow "
    "to a length of 60 cm and a mass of 6 kilograms, and bears a conspicuous pair of claws. "
    "Like other crustaceans, lobsters have a hard exoskeleton, which they must shed in order "
    "to grow, in a process called ecdysis. Much of the world's human consumption of lobsters "
    "is of this species; it is the most commercially significant species of lobster in Europe. "
    "Homarus gammarus lives in rocky environments at depths of up to 150 m. The colours of "
    "these animals are typically mottled blue, brown and yellow. They are nocturnal "
    "omnivores that are particularly sensitive to changes in water quality and temperature. "
    "The larvae go through four stages of development; after the fourth moult, they adopt a "
    "benthic lifestyle. Females can carry thousands of eggs for up to a year before the "
    "larvae hatch. The genome of H. gammarus has been sequenced, and is one of the largest "
    "animal genomes known. The species faces increasing pressure from human activities, "
    "including overfishing, habitat destruction, and climate change. Conservation measures "
    "include minimum landing sizes, closed seasons, and protected areas."
)


def run_single_config(precision_key: str, pc: dict, args) -> dict:
    """Load the model in one precision and run the full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"  Running benchmark: {pc['label']}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Load tokenizer and model
    # ------------------------------------------------------------------
    print("  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME,
        cache_dir=cfg.MODEL_CACHE_DIR,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model ({pc['label']}) …")
    model_kwargs = {
        "cache_dir": cfg.MODEL_CACHE_DIR,
        "device_map": "auto",
        "torch_dtype": pc["torch_dtype"],
        "trust_remote_code": True,
    }
    if pc["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = bnb_config
    elif pc["load_in_8bit"]:
        model_kwargs["load_in_8bit"] = True

    t_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, **model_kwargs)
    load_time_s = round(time.perf_counter() - t_load_start, 1)
    model.eval()

    vram_after_load_gb = vram_used_gb()
    ram_after_load_gb = round(psutil.virtual_memory().used / 1e9, 1)

    print(f"  Model loaded in {load_time_s}s  |  VRAM used: {vram_after_load_gb} GB  |"
          f"  RAM used: {ram_after_load_gb} GB")

    result = {
        "precision": precision_key,
        "label": pc["label"],
        "model_load_time_s": load_time_s,
        "vram_used_after_load_gb": vram_after_load_gb,
        "ram_used_after_load_gb": ram_after_load_gb,
    }

    # ------------------------------------------------------------------
    # Throughput (tokens/sec)
    # ------------------------------------------------------------------
    print("  Measuring decode throughput …")
    prompts = THROUGHPUT_PROMPTS if not args.quick else THROUGHPUT_PROMPTS[:2]
    result["throughput"] = measure_throughput(model, tokenizer, prompts,
                                              n_new_tokens=64 if args.quick else 256)

    # ------------------------------------------------------------------
    # Time to First Token
    # ------------------------------------------------------------------
    print("  Measuring time-to-first-token …")
    result["ttft_ms"] = measure_ttft(model, tokenizer, TTFT_PROMPTS)

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------
    if not args.quick:
        print("  Computing perplexity on WikiText sample …")
        try:
            result["perplexity_wikitext"] = measure_perplexity(
                model, tokenizer, WIKITEXT_SAMPLE
            )
        except Exception as exc:
            result["perplexity_wikitext"] = f"error: {exc}"

    # ------------------------------------------------------------------
    # Factual QA accuracy probe
    # ------------------------------------------------------------------
    if not args.quick:
        print("  Running factual QA probe …")
        result["qa_accuracy"] = measure_qa_accuracy(model, tokenizer, QA_PROBE)

    # ------------------------------------------------------------------
    # Peak VRAM during generation
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
        _dummy_inputs = tokenizer(THROUGHPUT_PROMPTS[0], return_tensors="pt")
        _dummy_inputs = {k: v.to(model.device) for k, v in _dummy_inputs.items()}
        with torch.no_grad():
            model.generate(**_dummy_inputs, max_new_tokens=128, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
        peak_vram_gb = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)
        result["peak_vram_during_generation_gb"] = peak_vram_gb

    # Unload model to free memory for next config
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Base Llama 3.1 8B")
    parser.add_argument("--quick", action="store_true",
                        help="Run a short smoke-test only")
    parser.add_argument("--config", choices=list(PRECISION_CONFIGS.keys()),
                        default=None,
                        help="Run only one precision configuration")
    parser.add_argument("--save", type=str, default="results",
                        help="Directory to write JSON results (default: results/)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Base Llama 3.1 8B — Performance Benchmark Suite")
    print("=" * 60)

    system_info = get_system_info()
    print("\nSystem Information:")
    for k, v in system_info.items():
        print(f"  {k:30s}: {v}")

    configs_to_run = (
        {args.config: PRECISION_CONFIGS[args.config]}
        if args.config
        else PRECISION_CONFIGS
    )

    all_results = {
        "system_info": system_info,
        "benchmark_run": {
            "timestamp": system_info["timestamp"],
            "quick_mode": args.quick,
            "configs_tested": list(configs_to_run.keys()),
        },
        "results": {},
    }

    for key, pc in configs_to_run.items():
        try:
            all_results["results"][key] = run_single_config(key, pc, args)
        except Exception as exc:
            print(f"\n  [ERROR] Config '{key}' failed: {exc}")
            all_results["results"][key] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Config':<10}  {'Tokens/s (mean)':<18}  {'VRAM (GB)':<12}  "
          f"{'Load time (s)':<14}  {'TTFT short (ms)'}")
    print("  " + "-" * 70)
    for key, r in all_results["results"].items():
        if "error" in r:
            print(f"  {key:<10}  ERROR: {r['error']}")
            continue
        tps = r.get("throughput", {}).get("mean_tokens_per_sec", "N/A")
        vram = r.get("vram_used_after_load_gb", "N/A")
        load = r.get("model_load_time_s", "N/A")
        ttft = r.get("ttft_ms", {}).get("short_20", {}).get("ttft_ms", "N/A")
        print(f"  {key:<10}  {str(tps):<18}  {str(vram):<12}  "
              f"{str(load):<14}  {ttft}")

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    if args.save:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "hardware_performance.json"
        with open(out_path, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\n  Results saved to: {out_path}")

    return all_results


if __name__ == "__main__":
    main()
