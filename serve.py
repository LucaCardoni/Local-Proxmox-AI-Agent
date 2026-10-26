#!/usr/bin/env python3
"""Launch llama_cpp's OpenAI-compatible server without WebSocket support.

We only need HTTP for pve-agent; this avoids the uvicorn↔websockets
version clash on Debian/Ubuntu where the distro websockets is too old.

The context size is configurable — the model itself supports 131072 tokens,
but the usable limit depends on how much VRAM is left for the KV cache
after the weights are loaded.  Measured on this box (MX250, 2 GB), with 42 GPU layers
and the model weights occupying ~1.3 GiB:

    # q8_0 KV cache (default — higher quality, larger memory)
    ctx= 8192  KV= 174 MiB  VRAM=1542 MiB   OK
    ctx=16384  KV= 349 MiB  VRAM=1726 MiB   OK
    ctx=24576  KV= 523 MiB  VRAM=1912 MiB   OK   <- default
    ctx=28672  FAILS: out of VRAM

    # q4_0 KV cache (½ the memory of q8_0 — ~15% quality loss, often imperceptible)
    ctx=32768  KV= 262 MiB  VRAM=1850 MiB   OK   <- high‑context mode
    ctx=65536  KV= 524 MiB  VRAM=2066 MiB   OK   (experimental)

Override with environment variables, e.g.
    PVE_N_CTX=32768 PVE_KV_TYPE=q4_0 pve
"""
import argparse
from pathlib import Path

import uvicorn
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings

from pve_config import load_settings

# The model's own limit (from the GGUF metadata: llama.context_length).
MODEL_MAX_CTX = 131072

# KV cache quantization.  llama.cpp takes the ggml type as an int:
#   1 = f16, 8 = q8_0, 2 = q4_0
# q8_0 halves the KV cache memory vs f16 at a small quality cost, which is
# what makes long contexts affordable on a 2 GB card.
KV_TYPES = {"f16": 1, "q8_0": 8, "q4_0": 2}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="path to TOML config (default: config.toml)")
    args = parser.parse_args(argv)
    try:
        config, config_path = load_settings(args.config)
    except ValueError as exc:
        parser.error(str(exc))

    server = config["server"]
    kv_type_name = server["kv_type"]
    if kv_type_name not in KV_TYPES:
        parser.error(f"server.kv_type must be one of {sorted(KV_TYPES)}")
    kv_type = KV_TYPES[kv_type_name]
    n_ctx = server["n_ctx"]
    model_path = Path(server["model_path"]).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    if not model_path.is_file():
        parser.error(f"model file does not exist: {model_path}")

    kv_type_bytes = {"f16": 2, "q8_0": 1, "q4_0": 0.5}[kv_type_name]
    kv_mib = (
        n_ctx * server["embedding_size_estimate"] * 2 * kv_type_bytes
        / (1024 * 1024)
    )
    total_vram_mib = server["model_vram_mib"] + kv_mib
    if kv_mib > 600:
        print(
            f"\n[LLAMA] KV-cache estimate: {kv_mib:,.0f} MiB "
            f"(n_ctx={n_ctx}, type={kv_type_name})\n"
            f"[LLAMA] Total VRAM needed: {total_vram_mib:,.0f} MiB "
            f"(model {server['model_vram_mib']} MiB + KV)\n"
            "[LLAMA] GPU VRAM available: check with your GPU's monitoring tools. "
            "Server will start, but OOM may occur if your GPU is tighter than the test box."
        )

    model_alias = server["model_alias"].replace("{n_ctx}", str(n_ctx))
    settings = Settings(
        model=str(model_path),
        model_alias=model_alias,
        n_gpu_layers=server["n_gpu_layers"],
        n_ctx=n_ctx,
        n_batch=server["n_batch"],
        n_ubatch=server["n_ubatch"],
        n_threads=server["n_threads"],
        n_threads_batch=server["n_threads_batch"],
        type_k=kv_type,
        type_v=kv_type,
        flash_attn=kv_type != KV_TYPES["f16"],
        # llama-cpp-python's server defaults logits_all=True, which allocates
        # an n_ctx × vocabulary float32 scores array. At 131k context this can
        # require tens of GiB. Chat completions only need the latest logits.
        logits_all=False,
        chat_format=(
            None if server["chat_format"] == "auto" else server["chat_format"]
        ),
        chat_template_kwargs={"enable_thinking": False},
        host=server["host"],
        port=server["port"],
    )

    app = create_app(settings)
    uvicorn.run(app, host=server["host"], port=server["port"], ws="none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())