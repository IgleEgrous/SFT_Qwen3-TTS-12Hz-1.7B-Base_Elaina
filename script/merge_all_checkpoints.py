# coding=utf-8
"""
merge_all_checkpoints.py — 批量合并 LoRA checkpoint 到 Base 模型

将 output/ 下的所有 checkpoint-epoch-* 目录中的 LoRA adapter 合并到 Base 模型，
输出为独立的完整 HF 模型，可直接上传。

前提：checkpoint 目录包含以下文件（由 sft_12hz_lora.py 生成）：
    adapter_model.safetensors   # LoRA 权重（A、B 矩阵）
    adapter_config.json         # PEFT adapter 配置
    model.safetensors           # Base 模型权重 + speaker embedding 在 slot 3000

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
from safetensors.torch import load_file, save_file


def find_checkpoints(checkpoints_dir: str):
    """找到所有 checkpoint-epoch-* 目录（需包含 adapter_model.safetensors）"""
    checkpoints = []
    for name in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, name)
        adapter_file = os.path.join(path, "adapter_model.safetensors")
        if os.path.isdir(path) and name.startswith("checkpoint-epoch-") and os.path.exists(adapter_file):
            try:
                epoch = int(name.replace("checkpoint-epoch-", ""))
            except ValueError:
                continue
            checkpoints.append((epoch, name, path))
    checkpoints.sort(key=lambda x: x[0])
    return [(name, path) for _, name, path in checkpoints]


def merge_single_checkpoint(base_model_path: str, checkpoint_path: str, output_dir: str,
                             speaker_name: str):
    """
    合并单个 checkpoint：

    1. 从 base_model_path 加载完整 base 模型
    2. 从 checkpoint 的 model.safetensors 取出 speaker embedding 注入到 slot 3000
    3. 用 PeftModel 加载 adapter + merge
    4. 保存完整合并模型
    """
    print(f"\n{'=' * 50}")
    print(f"合并: {checkpoint_path}")
    print(f"输出: {output_dir}")

    # 1. 加载 base 模型（完整权重）
    print("  [1/4] 加载 Base 模型...")
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    base_model = Qwen3TTSModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # 2. 注入 speaker embedding（从 checkpoint 的 model.safetensors 读取 slot 3000）
    checkpoint_model_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(checkpoint_model_path):
        print("  [2/4] 注入 speaker embedding 到 slot 3000...")
        ckpt_state = load_file(checkpoint_model_path, device="cpu")
        spk_embedding = ckpt_state.get("talker.model.codec_embedding.weight")
        if spk_embedding is not None:
            with torch.no_grad():
                base_model.model.talker.model.codec_embedding.weight[3000] = spk_embedding[3000].clone()
            print(f"       speaker embedding 已注入（来自 model.safetensors slot 3000）")
        else:
            print("       ⚠️  未在 checkpoint 中找到 speaker embedding，跳过注入")
    else:
        print("       ⚠️  checkpoint 中无 model.safetensors，跳过 speaker 注入")

    # 3. 加载 LoRA adapter 并 merge
    print("  [3/4] 加载 LoRA adapter 并合并...")
    peft_model = PeftModel.from_pretrained(
        base_model.model,
        checkpoint_path,
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()
    base_model.model = merged_model

    # 4. 保存完整合并模型
    print("  [4/4] 保存完整合并模型...")
    os.makedirs(output_dir, exist_ok=True)

    # 复制 Base 模型的所有文件
    for f in os.listdir(base_model_path):
        src = os.path.join(base_model_path, f)
        dst = os.path.join(output_dir, f)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # 保存合并后的权重（覆盖 model.safetensors）
    state_dict = {k: v.cpu() for k, v in base_model.state_dict().items()}

    # 删除 speaker_encoder（frozen，不参与推理）
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder.")]
    for k in keys_to_drop:
        del state_dict[k]

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # 更新 config.json
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["tts_model_type"] = "custom_voice"
    config["_name_or_path"] = output_dir
    talker_config = config.get("talker_config", {})
    talker_config["spk_id"] = {speaker_name: 3000}
    talker_config["spk_is_dialect"] = {speaker_name: False}
    config["talker_config"] = talker_config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  ✅ 已保存: {output_dir}")

    # 清理显存
    del base_model, peft_model, merged_model
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
        print(f"❌ 未找到任何包含 adapter_model.safetensors 的 checkpoint:")
        print(f"   目录: {args.checkpoints_dir}")
        print(f"   请先运行 sft_12hz_lora.py 生成 checkpoint")
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
            merge_single_checkpoint(
                base_model_path=args.base_model,
                checkpoint_path=path,
                output_dir=output_dir,
                speaker_name=args.speaker_name,
            )
        except Exception as e:
            print(f"  ❌ 合并失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"✅ 全部完成！合并模型保存在: {args.output_parent}")
    print("每个目录都是完整的 HF 模型，可直接上传。")


if __name__ == "__main__":
    main()
