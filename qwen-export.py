"""将 Qwen 类因果语言模型导出为 ONNX，可选 OnnxSlim 优化。"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import onnx
import onnxslim
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM


def get_model_weights(
    model_id: str, cache_root: str = "./model_weights"
) -> tuple[str, bool]:
    """获取模型权重：有本地缓存则直接用，否则用 ModelScope 下载。返回 (路径, 是否原本就存在)。"""
    safe_name = model_id.replace("/", "--")
    local_path = Path(cache_root) / safe_name

    # 已有完整缓存则直接返回
    if local_path.exists() and (local_path / "config.json").exists():
        print(f"✅ 发现本地缓存: {local_path.absolute()}")
        return str(local_path), True

    print("🚀 本地未发现模型，使用 ModelScope 下载...")
    local_path.mkdir(parents=True, exist_ok=True)
    from modelscope import snapshot_download as ms_snapshot

    path = ms_snapshot(model_id=model_id, local_dir=str(local_path))
    return path, False


def _provider_for_device(device: str) -> str:
    """根据 device 字符串返回 ONNX Runtime 的 provider。"""
    if device.lower() == "cuda":
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def export_to_onnx(
    model_id: str,
    output_dir: str,
    *,
    device: str = "cpu",
    dtype: str = "fp32",
    keep_weights: bool = True,
    use_onnxslim: bool = False,
) -> None:
    """
    导出主流程：拉取/加载权重 → 导出 ONNX → 可选 OnnxSlim → 可选删除权重。
    device: cpu / cuda；dtype: fp32 / fp16；keep_weights 为 False 且导出成功时删除权重目录。
    """
    weights_path, _ = get_model_weights(model_id)
    export_succeeded = False
    provider = _provider_for_device(device)

    try:
        # 1. 导出 ONNX
        print(f"\n📦 导出 ONNX 至: {output_dir} (device={device}, dtype={dtype})")
        model = ORTModelForCausalLM.from_pretrained(
            weights_path,
            export=True,
            trust_remote_code=True,
            provider=provider,
            dtype=dtype if dtype in ("fp32", "fp16", "bf16") else "fp32",
        )
        model.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(weights_path)
        tokenizer.save_pretrained(output_dir)

        # 2. 可选 OnnxSlim：先写临时文件，验证能加载再覆盖，失败则保留原文件
        if use_onnxslim:
            print("\n🪄 OnnxSlim 优化...")
            import onnxruntime as ort

            for p in Path(output_dir).glob("*.onnx"):
                orig_path = str(p)
                with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                    tmp_path = f.name
                try:
                    slim_model = onnxslim.slim(onnx.load(orig_path))
                    onnx.save(slim_model, tmp_path)
                    # 验证用 CPU 即可，避免未装 TensorRT 时的 EP 报错
                    ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
                    shutil.move(tmp_path, orig_path)
                except Exception as e:
                    print(f"   ⚠️ OnnxSlim 失败，保留原文件: {e}")
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

        print("\n✨ 导出完成。")
        export_succeeded = True
    finally:
        # 3. 可选清理：仅当导出成功且 keep_weights=False 时删除权重目录
        if not keep_weights and export_succeeded:
            print(f"\n🗑️ 清理权重: {weights_path}")
            try:
                shutil.rmtree(weights_path)
                print("✅ 已清理。")
            except OSError as e:
                print(f"❌ 清理失败: {e}")
        elif keep_weights:
            print(f"\n💾 权重保留: {weights_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Qwen 类因果语言模型导出为 ONNX。")
    parser.add_argument(
        "model_id",
        nargs="?",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="模型 ID，默认 Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./qwen25_05b_onnx",
        help="ONNX 输出目录，默认 ./qwen25_05b_onnx",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="设备：cpu 或 cuda，默认 cpu",
    )
    parser.add_argument(
        "-p",
        "--dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help="精度：fp32（全精度）或 fp16（半精度），默认 fp32",
    )
    parser.add_argument(
        "--no-keep-weights",
        action="store_true",
        help="导出成功后删除下载的权重目录以节省空间",
    )
    parser.add_argument(
        "--onnxslim",
        action="store_true",
        help="对 ONNX 做 OnnxSlim 优化（部分环境可能导致加载时 InvalidProtobuf）",
    )
    args = parser.parse_args()

    export_to_onnx(
        args.model_id,
        args.output,
        device=args.device,
        dtype=args.dtype,
        keep_weights=not args.no_keep_weights,
        use_onnxslim=args.onnxslim,
    )


if __name__ == "__main__":
    main()
