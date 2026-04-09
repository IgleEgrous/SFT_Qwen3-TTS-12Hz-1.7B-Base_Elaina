# coding=utf-8
"""
merge_all_checkpoints.py — 批量合并所有 LoRA checkpoint 到 Base 模型

将 output/ 下的所有 checkpoint-epoch-* 目录中的 LoRA 权重合并到 Base 模型，
输出为独立的完整 HF 模型，可直接上传。

用法：
    python script/merge_all_checkpoints.py \
        --base_model ../models/Qwen3-TTS-12Hz-1.7B-Base \
        --checkpoints_dir ../output \
        --output_parent ../output/merged_models \
        --speaker_name elaina

输出结构：
    output/merged_models/
    ├── checkpoint-epoch-2-merged/
    ├── checkpoint-epoch-4-merged/
    ├── checkpoint-epoch-8-merged/
    ├── checkpoint-epoch-16-merged/
    └── checkpoint-epoch-32-merged/
"""
import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import load_file, save_file
from transformers import AutoConfig


def find_checkpoints(checkpoints_dir: str):
    """找到所有 checkpoint-epoch-* 目录"""
    checkpoints = []
    for name in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-epoch-"):
            checkpoints.append((int(name.replace("checkpoint-epoch-", "")), name, path))
    checkpoints.sort(key=lambda x: x[0])
    return [(name, path) for _, name, path in checkpoints]


def merge_and_save(base_model_path: str, lora_dir: str, output_dir: str,
                   speaker_name: str):
    """
    合并单个 LoRA checkpoint 到 Base 模型并保存。

    注意：这里采用"权重注入"方式而非 from_pretrained，因为：
    - checkpoint 里只有 LoRA 的 adapter weights（adapter_model.safetensors）
    - 需要先加载 base，再把 LoRA 权重注入
    """
    print(f"\n{'='*50}")
    print(f"合并: {lora_dir}")
    print(f"输出: {output_dir}")

    # 1. 加载 base 模型（不做任何合并的干净加载）
    print("  [1/4] 加载 Base 模型...")
    base_model = Qwen3TTSModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # 2. 找到 adapter weights 文件
    adapter_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        print(f"  ⚠️  未找到 adapter_model.safetensors，跳过: {lora_dir}")
        return

    print(f"  [2/4] 加载 LoRA 权重: {adapter_path}")
    adapter_state = load_file(adapter_path, device="cpu")

    # 3. 注入 LoRA 权重到 base（用 PeftModel 方式）
    print("  [3/4] 注入 LoRA 权重...")
    base_model.model = PeftModel.from_pretrained(
        base_model.model,
        lora_dir,
        is_trainable=False,
    )

    # Merge LoRA into base
    merged_model = base_model.model.merge_and_unload()
    base_model.model = merged_model

    # 4. 保存完整合并模型
    print("  [4/4] 保存合并模型...")
    os.makedirs(output_dir, exist_ok=True)

    # 复制所有 base 模型文件
    for f in os.listdir(base_model_path):
        src = os.path.join(base_model_path, f)
        dst = os.path.join(output_dir, f)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # 保存合并后的权重
    state_dict = {k: v.cpu() for k, v in base_model.state_dict().items()}

    # 删除 speaker_encoder（训练时被 frozen，不参与推理）
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
    for k in keys_to_drop:
        del state_dict[k]

    # 保存 safetensors
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # 更新 config.json
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["tts_model_type"] = "custom_voice"
    config["_name_or_path"] = output_dir  # 记录来源
    talker_config = config.get("talker_config", {})
    talker_config["spk_id"] = {speaker_name: 3000}
    talker_config["spk_is_dialect"] = {speaker_name: False}
    config["talker_config"] = talker_config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  ✅ 已保存: {output_dir}")

    # 清理
    del base_model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="批量合并 LoRA checkpoints 到 Base 模型")
    parser.add_argument("--base_model", type=str,
                        default="../models/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Base 模型路径")
    parser.add_argument("--checkpoints_dir", type=str,
                        default="../output",
                        help="checkpoint-epoch-* 所在目录")
    parser.add_argument("--output_parent", type=str,
                        default="../output/merged_models",
                        help="合并模型输出父目录")
    parser.add_argument("--speaker_name", type=str, default="elaina",
                        help="说话人名称（更新 config.json 用）")
    parser.add_argument("--suffix", type=str, default="-merged",
                        help="输出目录后缀，如 checkpoint-epoch-2-merged")
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        print(f"❌ 未找到任何 checkpoint-epoch-* 目录: {args.checkpoints_dir}")
        return

    print(f"找到 {len(checkpoints)} 个 checkpoints:")
    for name, _ in checkpoints:
        print(f"  - {name}")

    os.makedirs(args.output_parent, exist_ok=True)

    for name, path in checkpoints:
        epoch_num = name.replace("checkpoint-epoch-", "")
        output_name = f"checkpoint-epoch-{epoch_num}{args.suffix}"
        output_dir = os.path.join(args.output_parent, output_name)

        try:
            merge_and_save(
                base_model_path=args.base_model,
                lora_dir=path,
                output_dir=output_dir,
                speaker_name=args.speaker_name,
            )
        except Exception as e:
            print(f"  ❌ 合并失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"✅ 全部完成！合并模型保存在: {args.output_parent}")
    print("每个目录都是完整的 HF 模型，可直接上传。")


if __name__ == "__main__":
    main()
