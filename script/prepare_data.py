# coding=utf-8
# prepare_data.py — 数据 tokenize 脚本
#
# 支持两种数据加载方式：
#   --mode hf       HuggingFace load_dataset 方式（租GPU后推荐）
#   --mode local    本地 parquet + hf_hub_download（Windows / 调试用）
#
# ref_audio 优先级：
#   1. --ref_audio PATH   直接指定本地音频路径
#   2. --ref_audio_idx N  从数据集中取第 N 条（默认 0）
#
# 输出统一的 train_with_codes.jsonl，格式：
#   {"audio": "...", "text": "...", "audio_codes": [...], "ref_audio": "..."}
#
# 用法：
#   # 方式1：HuggingFace 官方（自动选第一条作 ref_audio）
#   python prepare_data.py --mode hf \
#       --hf_repo yeeko/Elaina_WanderingWitch_audio_JA \
#       --audio_col audio --text_col transcription \
#       --tokenizer_model_path ../models/Qwen3-TTS-Tokenizer-12Hz \
#       --output_jsonl ../datasets/train_with_codes.jsonl
#
#   # 方式2：HuggingFace 官方（指定 ref_audio 索引）
#   python prepare_data.py --mode hf \
#       --hf_repo yeeko/Elaina_WanderingWitch_audio_JA \
#       --audio_col audio --text_col transcription \
#       --ref_audio_idx 42 \
#       --tokenizer_model_path ../models/Qwen3-TTS-Tokenizer-12Hz \
#       --output_jsonl ../datasets/train_with_codes.jsonl
#
#   # 方式3：本地 WAV 文件直接指定为 ref_audio（推荐）
#   python prepare_data.py --mode hf \
#       --hf_repo yeeko/Elaina_WanderingWitch_audio_JA \
#       --audio_col audio --text_col transcription \
#       --ref_audio /path/to/your/ref.wav \
#       --tokenizer_model_path ../models/Qwen3-TTS-Tokenizer-12Hz \
#       --output_jsonl ../datasets/train_with_codes.jsonl
#
#   # 方式4：本地 parquet（Windows multiprocessing 兼容）
#   python prepare_data.py --mode local \
#       --parquet_url .../metadata.parquet \
#       --audio_col file_name --text_col transcription \
#       --audio_dir ../datasets/audio \
#       --ref_audio /path/to/your/ref.wav \
#       --tokenizer_model_path ../models/Qwen3-TTS-Tokenizer-12Hz \
#       --output_jsonl ../datasets/train_with_codes.jsonl

import argparse
import json
import os
import sys

# 解决 Windows multiprocessing RLock bug
sys.stdout.flush()

import pandas as pd
from tqdm import tqdm

# ============================================================
# 数据加载
# ============================================================

def load_hf_dataset(
    hf_repo: str,
    split: str,
    audio_col: str,
    text_col: str,
    ref_audio: str = None,
    ref_audio_idx: int = 0,
    cache_dir: str = None,
):
    """
    HuggingFace load_dataset 方式加载数据。

    Args:
        hf_repo:       HF 数据集 repo id
        split:         split 名
        audio_col:     音频列名 (HF audio feature)
        text_col:      文本列名
        ref_audio:     直接指定 ref_audio 本地路径（优先）
        ref_audio_idx: 当 ref_audio 未指定时，用第几条数据作为 ref_audio
        cache_dir:     datasets 缓存目录
    Returns:
        items: list of dicts {"audio": "...", "text": "...", "ref_audio": "..."}
    """
    os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
    from datasets import load_dataset

    ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir, streaming=False)

    # 确定 ref_audio
    if ref_audio is not None:
        # 用户直接指定，无需从数据集取
        ref_audio_path = ref_audio
        rows = list(ds)
    else:
        # 从数据集中按索引选取
        rows = list(ds)
        audio_obj = rows[ref_audio_idx][audio_col]
        if isinstance(audio_obj, dict):
            ref_audio_path = audio_obj.get("path", "") or str(audio_obj)
        else:
            ref_audio_path = str(audio_obj)

    items = []
    for row in rows:
        audio_obj = row[audio_col]
        if isinstance(audio_obj, dict):
            audio_path = audio_obj.get("path", "") or str(audio_obj)
        else:
            audio_path = str(audio_obj)

        text = str(row[text_col])
        items.append({
            "audio": audio_path,
            "text": text,
            "ref_audio": ref_audio_path,
        })

    print(f"[HF] 加载了 {len(items)} 条 | ref_audio = {ref_audio_path}")
    return items


def load_local_parquet(
    parquet_url: str,
    audio_col: str,
    text_col: str,
    ref_audio: str = None,
    ref_audio_idx: int = 0,
    audio_dir: str = None,
    hf_repo: str = None,
):
    """
    本地 parquet + hf_hub_download 方式加载数据。
    解决 Windows Python 3.11+ multiprocessing RLock bug。

    Args:
        parquet_url:   metadata.parquet 的 HF 直链
        audio_col:     音频文件名/路径列名
        text_col:      文本列名
        ref_audio:     直接指定 ref_audio 本地路径（优先）
        ref_audio_idx: 当 ref_audio 未指定时，用第几条数据作为 ref_audio
        audio_dir:     本地音频缓存目录
        hf_repo:       HF repo id（用于下载）
    Returns:
        items: list of dicts
    """
    from huggingface_hub import hf_hub_download

    df = pd.read_parquet(parquet_url)
    print(f"[Local] parquet: {len(df)} 条 | 列名: {df.columns.tolist()}")

    if audio_col not in df.columns:
        raise ValueError(f"列 '{audio_col}' 不存在。可用列: {df.columns.tolist()}")
    if text_col not in df.columns:
        raise ValueError(f"列 '{text_col}' 不存在。可用列: {df.columns.tolist()}")

    # 确定 ref_audio
    if ref_audio is None:
        ref_audio_obj = df.iloc[ref_audio_idx][audio_col]
        ref_audio_obj = str(ref_audio_obj).replace("\\", "/")
        ref_audio_basename = os.path.basename(ref_audio_obj)

    items = []
    for i, row in df.iterrows():
        filename = str(row[audio_col]).replace("\\", "/")
        basename = os.path.basename(filename)

        # 尝试本地路径
        if audio_dir:
            local_path = os.path.join(audio_dir, basename)
            if os.path.exists(local_path):
                audio_path = local_path
            else:
                audio_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=f"train/{basename}",
                    repo_type="dataset",
                    cache_dir=audio_dir,
                )
        else:
            audio_path = hf_hub_download(
                repo_id=hf_repo,
                filename=f"train/{basename}",
                repo_type="dataset",
            )

        # ref_audio
        if ref_audio is not None:
            ref_audio_path = ref_audio
        else:
            ref_audio_path = audio_path if basename == ref_audio_basename else None

        items.append({
            "audio": audio_path,
            "text": str(row[text_col]),
            "ref_audio": None,  # 后面统一填
        })

    # 统一 ref_audio（第一次出现的那个）
    if ref_audio is None:
        ref_audio_path = items[ref_audio_idx]["audio"]
    for item in items:
        item["ref_audio"] = ref_audio_path

    print(f"[Local] ref_audio = {ref_audio_path}")
    return items


# ============================================================
# Tokenize
# ============================================================

def tokenize_and_save(
    items: list,
    tokenizer_model_path: str,
    output_jsonl: str,
    batch_size: int = 32,
    device: str = "cuda:0",
):
    """用 Qwen3TTSTokenizer 把音频转成 audio_codes，输出 JSONL。"""
    from qwen_tts import Qwen3TTSTokenizer

    print(f"[Tokenizer] 加载模型: {tokenizer_model_path}")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_path,
        device_map=device,
    )

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    batch_items = []
    batch_audios = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in tqdm(items, desc="Tokenizing"):
            batch_items.append(item)
            batch_audios.append(item["audio"])

            if len(batch_audios) >= batch_size:
                _flush_batch(tokenizer, batch_items, batch_audios, f)
                batch_items.clear()
                batch_audios.clear()

        if batch_audios:
            _flush_batch(tokenizer, batch_items, batch_audios, f)

    print(f"✅ 已保存: {output_jsonl}")


def _flush_batch(tokenizer, batch_items, batch_audios, file_handle):
    enc_res = tokenizer.encode(batch_audios)
    for item, audio_codes in zip(batch_items, enc_res.audio_codes):
        item["audio_codes"] = audio_codes.cpu().tolist()
        file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS 数据 tokenize")

    # === 通用参数 ===
    parser.add_argument("--mode", type=str, choices=["hf", "local"], required=True)
    parser.add_argument("--tokenizer_model_path", type=str,
                        default="../models/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 条（调试用）")

    # === HF 模式 ===
    parser.add_argument("--hf_repo", type=str,
                        help="HF repo id (mode=hf 时必填)")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--audio_col", type=str, default="audio")
    parser.add_argument("--text_col", type=str, default="transcription")
    parser.add_argument("--ref_audio", type=str, default=None,
                        help="直接指定 ref_audio 本地路径（优先于 --ref_audio_idx）")
    parser.add_argument("--ref_audio_idx", type=int, default=0,
                        help="当 --ref_audio 未指定时，用第几条数据的声音作为 ref_audio")
    parser.add_argument("--cache_dir", type=str, default=None)

    # === Local 模式 ===
    parser.add_argument("--parquet_url", type=str,
                        help="metadata.parquet 的 HF 直链 (mode=local 时必填)")
    parser.add_argument("--hf_repo_local", type=str,
                        help="HF repo id（用于 mode=local 的 hf_hub_download）")
    parser.add_argument("--audio_dir", type=str, default=None)

    args = parser.parse_args()

    # === 加载数据 ===
    if args.mode == "hf":
        if not args.hf_repo:
            raise ValueError("--hf_repo 在 mode=hf 时必填")
        items = load_hf_dataset(
            hf_repo=args.hf_repo,
            split=args.hf_split,
            audio_col=args.audio_col,
            text_col=args.text_col,
            ref_audio=args.ref_audio,
            ref_audio_idx=args.ref_audio_idx,
            cache_dir=args.cache_dir,
        )
    else:
        if not args.parquet_url:
            raise ValueError("--parquet_url 在 mode=local 时必填")
        if args.ref_audio is None and not args.hf_repo_local:
            parts = args.parquet_url.split("/")
            try:
                idx = parts.index("datasets")
                args.hf_repo_local = "/".join(parts[idx + 1:idx + 3])
            except ValueError:
                raise ValueError("无法从 parquet_url 推断 HF repo，请手动指定 --hf_repo_local")
            print(f"[Local] 推断 hf_repo = {args.hf_repo_local}")

        if args.audio_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            args.audio_dir = os.path.join(
                os.path.dirname(script_dir), "datasets", "audio"
            )

        items = load_local_parquet(
            parquet_url=args.parquet_url,
            audio_col=args.audio_col,
            text_col=args.text_col,
            ref_audio=args.ref_audio,
            ref_audio_idx=args.ref_audio_idx,
            audio_dir=args.audio_dir,
            hf_repo=args.hf_repo_local,
        )

    if args.limit:
        items = items[:args.limit]
        print(f"[Debug] 仅处理前 {args.limit} 条")

    tokenize_and_save(
        items=items,
        tokenizer_model_path=args.tokenizer_model_path,
        output_jsonl=args.output_jsonl,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
