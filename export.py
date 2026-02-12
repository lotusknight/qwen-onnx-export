import os
import torch
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# --- 环境变量配置 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def get_model_weights(model_id, cache_root="./model_weights"):
    """
    获取模型权重，返回路径
    """
    safe_name = model_id.replace("/", "--")
    local_path = Path(cache_root) / safe_name
    
    # 检查本地是否已存在完整模型
    if local_path.exists() and (local_path / "config.json").exists():
        print(f"✅ 发现本地缓存: {local_path.absolute()}")
        return str(local_path), True # 返回路径及“是否原本就在本地”
    
    print(f"🚀 本地未发现模型，开始下载...")
    local_path.mkdir(parents=True, exist_ok=True)
    
    # 优先尝试 HF 镜像
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.tflite"]
        )
        return path, False
    except Exception as e:
        print(f"⚠️ HF 下载失败，尝试 ModelScope: {e}")
        from modelscope import snapshot_download as ms_snapshot
        path = ms_snapshot(model_id=model_id, local_dir=str(local_path))
        return path, False

def export_to_onnx(model_id, output_dir, keep_weights=True):
    """
    导出主函数
    :param model_id: 模型 ID
    :param output_dir: ONNX 输出路径
    :param keep_weights: 是否保留原始权重 (默认 True)
    """
    # 1. 准备权重
    weights_path, already_existed = get_model_weights(model_id)
    
    try:
        # 2. 执行导出
        print(f"\n📦 开始导出 ONNX 至: {output_dir}")
        use_fp16 = torch.cuda.is_available()
        
        model = ORTModelForCausalLM.from_pretrained(
            weights_path,
            export=True,
            task="text-generation-with-past",
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )

        model.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(weights_path)
        tokenizer.save_pretrained(output_dir)

        # 3. OnnxSlim 优化
        print(f"\n🪄 正在运行 OnnxSlim 优化...")
        import onnx, onnxslim
        for p in Path(output_dir).glob("*.onnx"):
            print(f"   优化中: {p.name}")
            slim_model = onnxslim.slim(onnx.load(str(p)))
            onnx.save(slim_model, str(p))
            
        print("\n✨ ONNX 导出与优化成功完成！")

    finally:
        # 4. 清理逻辑
        # 如果 keep_weights 为 False，且模型是本次脚本刚下载的，则清理
        if not keep_weights:
            print(f"\n🗑️ 参数 keep_weights=False，正在清理原始权重目录: {weights_path}")
            try:
                # 使用 shutil.rmtree 删除整个文件夹
                shutil.rmtree(weights_path)
                print("✅ 原始权重已清理。")
            except Exception as e:
                print(f"❌ 清理失败: {e}")
        else:
            print(f"\n💾 原始权重保留在: {weights_path}")

if __name__ == "__main__":
    # --- 用户配置区 ---
    TARGET_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    EXPORT_PATH = "./qwen25_05b_onnx"
    
    # 设置为 True 则保留下载的 1GB+ 原始权重
    # 设置为 False 则在生成 ONNX 后删除原始权重，节省空间
    KEEP_ORIGINAL = True 

    export_to_onnx(TARGET_MODEL, EXPORT_PATH, keep_weights=KEEP_ORIGINAL)
