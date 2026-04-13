#!/usr/bin/env python3

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


PROMPTS = {
    "explain": "Explain the difference between a CLI and a TUI in two short sentences.",
    "code": "Write a Python function fib(n) that returns the nth Fibonacci number using iteration and include a short docstring.",
}


def median(values):
    return statistics.median(values) if values else None


def build_mode_args(mode: str, args) -> list[str]:
    common = [
        "--parallel", "1",
        "-ngl", "all",
        "-n", str(args.n_predict),
        "--temp", "0",
        "--top-k", "1",
        "--top-p", "1",
    ]
    if mode == "gguf_plain":
        return common + ["-md", str(args.draft_model)]
    if mode == "hybrid_mtp":
        if not args.sidecar_dir:
            raise SystemExit("--sidecar-dir is required for hybrid_mtp")
        return common + [
            "-md", str(args.draft_model),
            "--gemma4-mtp-sidecar", str(args.sidecar_dir),
        ]
    if mode == "helper_plain":
        return common + [
            "--gemma4-draft-backend", "helper",
            "--gemma4-litert-draft-helper-bin", str(args.helper_bin),
            "--gemma4-litert-draft-manifest", str(args.manifest),
            "--gemma4-litert-draft-mode", "plain",
        ]
    if mode == "helper_mtp":
        return common + [
            "--gemma4-draft-backend", "helper",
            "--gemma4-litert-draft-helper-bin", str(args.helper_bin),
            "--gemma4-litert-draft-manifest", str(args.manifest),
            "--gemma4-litert-draft-mode", "mtp",
        ]
    if mode == "ffi_plain":
        return common + [
            "--gemma4-draft-backend", "ffi",
            "--gemma4-litert-draft-lib", str(args.ffi_lib),
            "--gemma4-litert-draft-manifest", str(args.manifest),
            "--gemma4-litert-draft-mode", "plain",
        ]
    if mode == "ffi_mtp":
        return common + [
            "--gemma4-draft-backend", "ffi",
            "--gemma4-litert-draft-lib", str(args.ffi_lib),
            "--gemma4-litert-draft-manifest", str(args.manifest),
            "--gemma4-litert-draft-mode", "mtp",
        ]
    raise SystemExit(f"unknown mode: {mode}")


def summarize_runs(runs):
    numeric_keys = [
        "chunks",
        "proposed_tokens",
        "accepted_from_draft",
        "accepted_tokens",
        "acceptance_rate",
        "verifier_substitutions",
        "full_match_chunks",
        "first_token_mismatches",
        "draft_build_s",
        "draft_decode_s",
        "draft_mtp_s",
        "draft_self_verify_s",
        "transport_s",
        "verify_s",
        "draft_sync_s",
        "avg_proposed_per_chunk",
        "avg_accepted_per_chunk",
        "emitted_per_chunk",
    ]
    out = {}
    first = runs[0]
    for key in ["draft_backend", "litert_draft_mode", "use_mtp"]:
        out[key] = first.get(key)
    for key in numeric_keys:
        vals = [run[key] for run in runs if key in run and run[key] is not None]
        out[key] = median(vals) if vals else None
    out["generated_text"] = first.get("generated_text")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, default=Path("/home/neilopet/src/github.com/neilopet/llama.cpp-turboquant-eval/build-hip-turboeval/bin/llama-gemma4-mtp-prototype"))
    p.add_argument("--target-model", type=Path, default=Path("/home/neilopet/models/spec-proof-gemma4/gemma-4-31B-it-UD-Q4_K_XL.gguf"))
    p.add_argument("--draft-model", type=Path, default=Path("/home/neilopet/models/spec-proof-gemma4/gemma-4-E2B-it-UD-Q4_K_XL.gguf"))
    p.add_argument("--manifest", type=Path, default=Path("/home/neilopet/litert_bundles/gemma-4-E2B-it/manifest.json"))
    p.add_argument("--helper-bin", type=Path, default=Path("/home/neilopet/src/github.com/neilopet/strix_infer_spike/target/release/xtask"))
    p.add_argument("--ffi-lib", type=Path, default=Path("/home/neilopet/src/github.com/neilopet/strix_infer_spike/target/release/libstrix_gemma_draft_ffi.so"))
    p.add_argument("--sidecar-dir", type=Path, default=Path("/tmp/gemma4_e2b_mtp_sidecar"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--n-predict", type=int, default=32)
    p.add_argument("--modes", default="gguf_plain,hybrid_mtp,helper_plain,helper_mtp,ffi_plain,ffi_mtp")
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {"runs": {}, "summary": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for prompt_name, prompt in PROMPTS.items():
        results["runs"][prompt_name] = {}
        results["summary"][prompt_name] = {}
        for mode in modes:
            runs = []
            for rep in range(args.reps):
                with tempfile.NamedTemporaryFile(prefix=f"{mode}_{prompt_name}_{rep}_", suffix=".json", delete=False) as tf:
                    report_path = Path(tf.name)
                cmd = [
                    str(args.binary),
                    "-m", str(args.target_model),
                    "--gemma4-mtp-report", str(report_path),
                    "-p", prompt,
                    *build_mode_args(mode, args),
                ]
                subprocess.run(cmd, check=True)
                with report_path.open() as f:
                    run = json.load(f)
                run["mode"] = mode
                run["prompt_name"] = prompt_name
                run["rep"] = rep
                runs.append(run)
                report_path.unlink(missing_ok=True)
            results["runs"][prompt_name][mode] = runs
            results["summary"][prompt_name][mode] = summarize_runs(runs)

    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed: {exc.cmd}", file=sys.stderr)
        raise
