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
│   └── sft_12hz_lora.py              # LoRA SFT 训练脚本（基于官方 sft_12hz.py）
└── README.md
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
- `ref_audio`：所有样本建议使用同一个参考音频（长度 3~10 秒，24kHz，干净无噪）
- `text`：日语/中文/英文均可，对应音频内容即可

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

## 完整训练流程

### Step 1: Tokenize 数据

将音频转换为离散的 audio codes，生成 `train_with_codes.jsonl`。

**方式 A：HuggingFace 官方 `load_dataset`（推荐，租GPU后使用）**

```bash
python script/prepare_data.py --mode hf \
  --hf_repo yeeko/Elaina_WanderingWitch_audio_JA \
  --hf_split train \
  --audio_col audio \
  --text_col transcription \
  --ref_audio_idx 0 \
  --tokenizer_model_path ../models/Qwen3-TTS-Tokenizer-12Hz \
  --output_jsonl ../datasets/train_with_codes.jsonl \
  --batch_size 32 \
  --device cuda:0
```

**方式 B：本地 parquet + `hf_hub_download`（Windows / 调试用）**

解决 Windows Python 3.11+ multiprocessing RLock bug：

```bash
python script/prepare_data.py --mode local \
  --parquet_url https://huggingface.co/datasets/yeeko/Elaina_WanderingWitch_audio_JA/resolve/main/train/metadata.parquet \
  --audio_col file_name \
  --text_col transcription \
  --audio_dir ../datasets/audio \
  --ref_audio_idx 0 \
  --tokenizer_model_path ../models/Qwen3-TTS-Tokenizer-12Hz \
  --output_jsonl ../datasets/train_with_codes.jsonl \
  --batch_size 32 \
  --device cuda:0
```

**调试选项**：`--limit N` 可仅处理前 N 条数据，快速验证 pipeline。

---

### Step 2: LoRA SFT 训练

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

## 硬件需求

| 模型大小 | 最低显存 | 推荐显存 |
|---|---|---|
| 0.6B | 8GB | 16GB |
| 1.7B（LoRA） | 8GB | 12GB |
| 1.7B（全量微调） | 16GB | 24GB |

LoRA 模式下训练参数量约为全模型的 0.1%~1%，大幅降低显存需求。

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

### checkpoint 输出

```
output/
└── checkpoint-epoch-{N}/
    ├── config.json            # 已更新 tts_model_type=custom_voice
    ├── model.safetensors      # LoRA权重 + 融合speaker embedding
    └── ...（其余文件同Base模型）
```

## 相关资料

- [Qwen3-TTS 官方文档](https://www.mintlify.com/QwenLM/Qwen3-TTS/advanced/fine-tuning)
- [Qwen3-TTS Easy Finetuning](https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning) — 开箱即用的 WebUI 微调工具
- [PEFT 库](https://github.com/huggingface/peft)
- [LoRA 原文](https://arxiv.org/abs/2106.09685)
- [伊蕾娜语音数据集](https://huggingface.co/datasets/yeeko/Elaina_WanderingWitch_audio_JA)
