# SFT Qwen3-TTS-12Hz-1.7B-Base

基于 Qwen3-TTS-12Hz-1.7B-Base 的 LoRA 微调项目，使用伊蕾娜（魔女之旅/本渡枫）语音数据进行训练。

## 项目背景

零样本语音克隆方便快捷，但存在音色不稳定、跨语言带口音、情感表达不够自然等问题。微调后的模型能够：
- **音色稳定**：在各种文本上都保持一致的声音特征
- **情感丰富**：支持自然语言语气/语速指示（如"用悲伤的语气说"）
- **跨语言地道**：有效防止跨语言推理时的母语口音

## 目录结构

```
SFT_Qwen3-TTS-12Hz-1.7B-Base/
├── models/
│   ├── Qwen3-TTS-12Hz-1.7B-Base/    # Base 模型权重
│   └── Qwen3-TTS-Tokenizer-12Hz/     # 专用 tokenizer
├── datasets/                          # 数据集输出目录
├── output/                           # 训练输出目录
├── script/
│   ├── dataset.py                    # 官方数据集加载逻辑
│   ├── prepare_data.py               # 数据 tokenize 脚本（支持 HF / Local 两种模式）
│   ├── load_elaina.py                # 伊蕾娜数据集预览/下载工具
│   ├── sft_12hz_lora.py             # LoRA SFT 训练脚本
│   ├── eval_checkpoints.py           # 多 checkpoint 横向对比脚本
│   └── merge_all_checkpoints.py      # 批量合并 LoRA 到 Base 模型脚本
└── README.md
```

## 完整工作流程

```
Step 1: Tokenize 数据
Step 2: LoRA SFT 训练
Step 3: 横向对比各 epoch 效果（选最优 checkpoint）
Step 4: 批量合并所有 LoRA checkpoint → 完整 HF 模型
```

## 数据集

### 伊蕾娜语音数据集

- **HF Repo**: [yeeko/Elaina_WanderingWitch_audio_JA](https://huggingface.co/datasets/yeeko/Elaina_WanderingWitch_audio_JA)
- **内容**: 1444 条伊蕾娜日语语音 + transcription
- **采样率**: 原始音频 24kHz
- **格式**: `train/metadata.parquet` + `train/` 下 1444 个音频文件（MP3/WAV）

### JSONL 训练数据格式

tokenize 后输出格式（`train_with_codes.jsonl`）：

```jsonl
{"audio": "/path/to/audio.wav", "text": "其实我真的有发现...", "audio_codes": [...], "ref_audio": "/path/to/ref.wav"}
```

**关键要求**：
- `ref_audio`：参考音频路径，所有样本使用同一个 ref_audio（长度 3~10 秒，24kHz，干净无噪）。用户可自由选择任意音频作为参考，常见做法是取数据集内某一条音频，或使用外部高质量音频。

## 环境依赖

```bash
# 核心依赖（pyproject.toml 中已声明）
torch
transformers >= 4.38.0
accelerate
peft
datasets
qwen-tts
librosa
soundfile
scipy >= 1.10.0
```

---

## Step 1: Tokenize 数据

将音频转换为离散的 audio codes，生成 `train_with_codes.jsonl`。

**方式 A：HuggingFace 官方 `load_dataset`（推荐，租GPU后使用）**

```bash
python script/prepare_data.py --mode hf \
  --hf_repo yeeko/Elaina_WanderingWitch_audio_JA \
  --hf_split train \
  --audio_col audio \
  --text_col transcription \
  --ref_audio "/path/to/你的参考音频.wav"  # 用户自由选择参考音频\
  --tokenizer_model_path ../models/Qwen3-TTS-Tokenizer-12Hz \
  --output_jsonl ../datasets/train_with_codes.jsonl \
  --batch_size 32 \
  --device cuda:0
```

> **ref_audio 选择方式**：
> - 直接指定 `--ref_audio /path/to/audio.wav`（推荐，可自由选择任意高质量音频）
> - 或指定 `--ref_audio_idx N` 从数据集中取第 N 条（默认 0）
> - 优先级：`--ref_audio` > `--ref_audio_idx`

**方式 B：本地 parquet + `hf_hub_download`（Windows / 调试用）**

解决 Windows Python 3.11+ multiprocessing RLock bug：

```bash
python script/prepare_data.py --mode local \
  --parquet_url https://huggingface.co/datasets/yeeko/Elaina_WanderingWitch_audio_JA/resolve/main/train/metadata.parquet \
  --audio_col file_name \
  --text_col transcription \
  --audio_dir ../datasets/audio \
  --ref_audio "/path/to/你的参考音频.wav"  # 用户自由选择参考音频\
  --tokenizer_model_path ../models/Qwen3-TTS-Tokenizer-12Hz \
  --output_jsonl ../datasets/train_with_codes.jsonl \
  --batch_size 32 \
  --device cuda:0
```

**调试选项**：`--limit N` 可仅处理前 N 条数据，快速验证 pipeline。

---

## Step 2: LoRA SFT 训练

```bash
python script/sft_12hz_lora.py \
  --init_model_path ../models/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl ../datasets/train_with_codes.jsonl \
  --output_model_path ../output \
  --speaker_name elaina \
  --batch_size 2 \
  --lr 2e-6 \
  --num_epochs 3
```

**常用参数说明**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--init_model_path` | `../models/Qwen3-TTS-12Hz-1.7B-Base` | Base 模型路径 |
| `--train_jsonl` | **必填** | tokenize 后的训练数据 |
| `--output_model_path` | `../output` | checkpoint 输出目录 |
| `--speaker_name` | `speaker_test` | 自定义说话人名称 |
| `--batch_size` | `2` | 每 GPU batch size（24GB 卡建议 16） |
| `--lr` | `2e-6` | 学习率（1.7B 模型建议从 2e-6 开始） |
| `--num_epochs` | `3` | 训练轮数 |
| `--save_epochs` | `None` | 指定保存 checkpoint 的 epoch 列表，如 `--save_epochs 2 4 8 16 32` |
| `--lora_r` | `16` | LoRA rank，越大越强 |
| `--lora_alpha` | `32` | 2×r |
| `--lora_dropout` | `0.0` | LoRA dropout |
| `--lora_target_modules` | `q_proj,k_proj,v_proj,o_proj` | LoRA 目标层 |
| `--train_code_predictor` | `False` | 是否训练 code_predictor embedding（默认 False） |

**示例：训练 32 epoch，只保存 2/4/8/16/32**：

```bash
python script/sft_12hz_lora.py \
  --init_model_path ../models/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl ../datasets/train_with_codes.jsonl \
  --output_model_path ../output \
  --speaker_name elaina \
  --num_epochs 32 \
  --batch_size 2 \
  --lr 2e-6 \
  --save_epochs 2 4 8 16 32
```

**checkpoint 输出结构**：

```
output/
└── checkpoint-epoch-{N}/
    ├── adapter_model.safetensors    # LoRA 权重（仅此文件，约几 MB）
    └── ...（其余文件复制自 Base 模型）
```

---

## Step 3: 横向对比各 epoch 效果

训练完后用同一段 ref_audio + 同一批测试文本，对比不同 epoch 的生成效果。

```bash
python script/eval_checkpoints.py \
  --checkpoints \
    ../output/checkpoint-epoch-2 \
    ../output/checkpoint-epoch-4 \
    ../output/checkpoint-epoch-8 \
    ../output/checkpoint-epoch-16 \
    ../output/checkpoint-epoch-32 \
  --ref_audio path/to/ref.wav \
  --test_texts \
    "学校行くのは嫌だけど、私みたいな人間は一日行かなかっただけでクラスの皆から存在を忘れられてしまうんだよ" \
    "こんばんは、こっちはボッチです" \
    "本当に、お前助かったな" \
  --output_dir ../output/eval_samples \
  --speaker elaina
```

**输出结构**：

```
output/eval_samples/
├── checkpoint-epoch-2/
│   ├── 学校行くのは嫌だと....wav
│   └── ...
├── checkpoint-epoch-4/
└── ...
```

**验证方法**：用音频播放器横向对比同一文本在不同 epoch 下的效果，选音色最稳、语气最自然的 epoch。

---

## Step 4: 批量合并 LoRA → 完整 HF 模型

将所有保存的 LoRA checkpoint 分别合并到 Base 模型，输出为独立的完整 HF 模型（可直接上传）。

```bash
python script/merge_all_checkpoints.py \
  --base_model ../models/Qwen3-TTS-12Hz-1.7B-Base \
  --checkpoints_dir ../output \
  --output_parent ../output/merged_models \
  --speaker_name elaina
```

**输出结构**：

```
output/merged_models/
├── checkpoint-epoch-2-merged/     # 完整 HF 模型
├── checkpoint-epoch-4-merged/
├── checkpoint-epoch-8-merged/
├── checkpoint-epoch-16-merged/
└── checkpoint-epoch-32-merged/
```

每个目录都是完整的 HF 模型，`from_pretrained` 直接加载。

---

## 硬件需求

| 模型大小 | 最低显存 | 推荐显存 |
|---|---|---|
| 0.6B | 8GB | 16GB |
| 1.7B（LoRA） | 8GB | 12GB |
| 1.7B（全量微调） | 16GB | 24GB |

LoRA 模式下训练参数量约为全模型的 0.1%~1%，大幅降低显存需求。

---

## 技术细节

### LoRA 配置

基于 [Hu et al. 2021] LoRA: Low-Rank Adaptation of Large Language Models。

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### 训练配置

- **混合精度**: bfloat16
- **梯度累积**: 4 steps
- **优化器**: AdamW (weight_decay=0.01)
- **梯度裁剪**: max_norm=1.0
- **Loss**: `outputs.loss + 0.3 * sub_talker_loss`
- **FlashAttention**: 开启

### 数据预处理流程

```
原始音频(.wav) + 文本(transcription)
        ↓
Qwen3TTSTokenizer.encode() → audio_codes (离散语音token，12Hz)
        ↓
拼接: text_tokens + [BOS] + speech_tokens + [EOS]
        ↓
只对 speech_tokens 部分计算 loss（prompt masking）
```

### LoRA 权重 vs 完整模型

| | LoRA checkpoint | 合并后模型 |
|---|---|---|
| 文件大小 | ~几 MB | ~3.8GB |
| 内容 | 仅 A、B 矩阵 | 完整权重 |
| 加载方式 | 需要配合 Base 模型 | `from_pretrained` 直接加载 |
| 用途 | 中间保存，可选多个 epoch | 最终上传/发布 |

---

## 相关资料

- [Qwen3-TTS 官方文档](https://www.mintlify.com/QwenLM/Qwen3-TTS/advanced/fine-tuning)
- [Qwen3-TTS Easy Finetuning](https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning) — 开箱即用的 WebUI 微调工具
- [PEFT 库](https://github.com/huggingface/peft)
- [LoRA 原文](https://arxiv.org/abs/2106.09685)
- [伊蕾娜语音数据集](https://huggingface.co/datasets/yeeko/Elaina_WanderingWitch_audio_JA)
