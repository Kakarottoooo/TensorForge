"""Run a deterministic baseline generation smoke test on CPU or CUDA."""

from __future__ import annotations

import argparse

import torch

from tensorforge.model.config import tiny_config
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.dtypes import parse_dtype, validate_dtype_support
from tensorforge.runtime.generation import GenerationConfig, greedy_generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = parse_dtype(args.dtype)
    validate_dtype_support(dtype, device)

    torch.manual_seed(7)
    config = tiny_config(max_position_embeddings=args.prompt_length + args.max_new_tokens)
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    input_ids = torch.randint(
        config.vocab_size,
        (args.batch_size, args.prompt_length),
        device=device,
        dtype=torch.long,
    )
    output = greedy_generate(model, input_ids, GenerationConfig(args.max_new_tokens))
    print(
        {
            "device": str(device),
            "dtype": str(dtype),
            "input_shape": list(input_ids.shape),
            "output_shape": list(output.shape),
            "checksum": int(output.long().sum().item()),
        }
    )


if __name__ == "__main__":
    main()
