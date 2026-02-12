"""Run inference with the exported ONNX model to verify export correctness."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

DEFAULT_MODEL_PATH = "./qwen25_05b_onnx"
DEFAULT_PROMPT = "What is 2 + 2? Reply in one short sentence."


def load_model_and_tokenizer(model_path: str):
    """Load ONNX model and tokenizer from an export directory."""
    path = Path(model_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"ONNX model dir not found: {path}")

    # Prefer CUDA when available in ONNX Runtime; else CPU
    try:
        model = ORTModelForCausalLM.from_pretrained(
            str(path),
            export=False,
            provider="CUDAExecutionProvider",
        )
        print("Using provider: CUDAExecutionProvider")
    except ValueError as e:
        if "execution provider" in str(e).lower() or "available" in str(e).lower():
            model = ORTModelForCausalLM.from_pretrained(
                str(path),
                export=False,
                provider="CPUExecutionProvider",
            )
            print("Using provider: CPUExecutionProvider")
        else:
            raise
    except Exception as e:
        if "InvalidProtobuf" in type(e).__name__ or "INVALID_PROTOBUF" in str(e):
            print(
                "ONNX model failed to load (InvalidProtobuf). Re-export with "
                "USE_ONNXSLIM = False in qwen-export.py and run export again.",
                file=sys.stderr,
            )
        raise

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    return model, tokenizer


def run_inference(
    model_path: str = DEFAULT_MODEL_PATH,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 128,
) -> None:
    """
    Run a short text generation with the ONNX model.

    :param model_path: Path to the exported ONNX dir (e.g. ./qwen25_05b_onnx).
    :param prompt: User prompt; for Qwen Instruct, wrapped in chat template.
    :param max_new_tokens: Maximum new tokens to generate.
    """
    print(f"Loading ONNX model from: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Qwen Instruct: use chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = getattr(model, "device", torch.device("cpu"))
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"\nPrompt: {prompt}")
    print("Generating...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # Decode only the new tokens (move to CPU for decode if tensor on GPU)
    input_len = inputs["input_ids"].shape[1]
    reply_ids = outputs[0][input_len:]
    if isinstance(reply_ids, torch.Tensor) and reply_ids.is_cuda:
        reply_ids = reply_ids.cpu()
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True)

    print(f"\nReply: {reply.strip()}")
    print("\nDone. ONNX inference ran successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test exported ONNX Qwen model with a short generation."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to ONNX export dir (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt for the model",
    )
    parser.add_argument(
        "-n",
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens to generate (default: 128)",
    )
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
