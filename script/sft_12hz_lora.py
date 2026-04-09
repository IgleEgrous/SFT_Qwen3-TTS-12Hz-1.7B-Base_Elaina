# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# LoRA-modified version of sft_12hz.py
# Based on: Qwen3-TTS/finetuning/sft_12hz.py
#
# Changes:
#   - Added PEFT LoRA integration (get_peft_model)
#   - Added LoRA-specific CLI arguments
#   - Frozen code_predictor embeddings (no gradient)
#   - Optimizer only updates LoRA parameters

import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from peft import LoraConfig, TaskType, get_peft_model, get_peft_state_dict
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()

    # === 官方原有参数 ===
    parser.add_argument("--init_model_path", type=str,
                        default="../models/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="../output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)   # LoRA 建议用小 lr
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--save_epochs", type=int, nargs="+", default=None,
                        help="指定保存 checkpoint 的 epoch 列表，如 --save_epochs 2 4 8 16 32。默认 None 表示每个 epoch 都保存")

    # === LoRA 新增参数 ===
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank, 越大表达能力越强，首次建议 16")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha，通常设为 2*r")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="LoRA 目标层，逗号分隔")
    parser.add_argument("--train_code_predictor", action="store_true",
                        help="是否也训练 code_predictor embedding（默认 False，Frozen）")

    args = parser.parse_args()

    # === Accelerator (LoRA + Accelerator 兼容) ===
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard"
    )

    # === 加载模型 ===
    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # ============================================================
    # 改动点 1：Freeze code_predictor embedding（不参与训练）
    # ============================================================
    if not args.train_code_predictor:
        for i in range(1, 16):
            for param in qwen3tts.model.talker.code_predictor.get_input_embeddings()[i - 1].parameters():
                param.requires_grad = False

    # ============================================================
    # 改动点 2：应用 PEFT LoRA
    # ============================================================
    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    qwen3tts.model = get_peft_model(qwen3tts.model, lora_config)

    # 打印 LoRA 参数数量（方便确认）
    trainable_params, total_params = 0, 0
    for p in qwen3tts.model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        total_params += p.numel()
    print(f"[LoRA] Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # ============================================================
    # 数据加载
    # ============================================================
    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    # ============================================================
    # 改动点 3：Optimizer 只优化 LoRA 参数（全模型 requires_grad=False 时）
    # ============================================================
    optimizer = AdamW(
        qwen3tts.model.parameters(),   # 只有 LoRA 的 A、B 矩阵的 requires_grad=True
        lr=args.lr,
        weight_decay=0.01
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    save_epochs = set(args.save_epochs) if args.save_epochs else None  # None = 保存所有
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                # Speaker embedding frozen，detach 避免干扰
                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(
                    input_text_ids
                ) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(
                    input_codec_ids
                ) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                # Sub-talker loss（code_predictor frozen 时理论上为 0，但保留兼容）
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss_val:.4f}"
                )

        # ============================================================
        # Checkpoint 保存（仅指定 epoch）
        # ============================================================
        should_save = (save_epochs is None) or (epoch in save_epochs)
        if should_save and accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            # 更新 config.json
            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # ============================================================
            # 保存 LoRA adapter（标准 PEFT 格式）
            # ============================================================
            unwrapped_model = accelerator.unwrap_model(model)

            # 用 PEFT 标准方式保存 adapter weights + config
            # 不依赖 accelerator，避免 key 不匹配问题
            peft_state_dict = get_peft_state_dict(unwrapped_model, "cuda:0" if torch.cuda.is_available() else "cpu")

            # 删除 frozen speaker_encoder（如有）
            keys_to_drop = [k for k in peft_state_dict.keys() if k.startswith("speaker_encoder.")]
            for k in keys_to_drop:
                del peft_state_dict[k]

            # 保存 adapter weights（LoRA A、B 矩阵，仅几 MB）
            save_file(peft_state_dict, os.path.join(output_dir, "adapter_model.safetensors"))

            # 保存 adapter config（PEFT 标准格式）
            unwrapped_model.peft_config["default"].to_json_file(
                os.path.join(output_dir, "adapter_config.json")
            )
            print(f"✅ Checkpoint saved: {save_path}")


if __name__ == "__main__":
    train()
