# coding=utf-8
# prepare_data.py — 数据 tokenize 脚本
#
# 支持两种数据加载方式：
#   --mode hf       HuggingFace load_dataset 方式（租GPU后推荐）
#   --mode local    本地 parquet + hf_hub_download（Windows / 调试用）
#
# 输出统一的 train_with_codes.jsonl，格式：
#   {"audio": "...", "text": "...", "audio_codes": [...], "ref_audio": "..."}
#
# 用法：
#   # 方式1：HuggingFace 官方
#   python prepare_data.py --mode hf \
#       --hf_repo yeeko/Elaina_WanderingWitch_audio_JA \
#       --hf_split train \
#       --audio_col audio --text_col transcription \
#       --ref_audio_idx 0 \
#       --tokenizer_model_path ../models/Qwen3-TTS-Tokenizer-12Hz \
#       --output_jsonl ../datasets/train_with_codes.jsonl
#
#   # 方式2：本地 parquet（Windows multiprocessing 兼容）
#   python prepare_data.py --mode local \
#       --parquet_url https://huggingface.co/datasets/yeeko/Elaina_WanderingWitch_audio_JA/resolve/main/train/metadata.parquet \
#       --audio_col file_name --text_col transcription \
#       --audio_dir ../datasets/audio \
#       --ref_audio_idx 0 \
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
    ref_audio_idx: int = 0,
    cache_dir: str = None,
):
    """
    HuggingFace load_dataset 方式加载数据。

    Args:
        hf_repo:       HF 数据集 repo id (e.g. "yeeko/Elaina_WanderingWitch_audio_JA")
        split:         split 名 (e.g. "train")
        audio_col:     音频列名 (HF audio feature)
        text_col:      文本列名
        ref_audio_idx: 用第几条数据的声音作为 ref_audio（默认第0条）
        cache_dir:     datasets 缓存目录
    Returns:
        items: list of dicts {"audio": "...", "text": "...", "ref_audio": "..."}
    """
    os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"

    from datasets import load_dataset

    ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir, streaming=False)
    ds = ds.to_iterable_dataset()  # 迭代器，省内存

    items = []
    ref_audio_path = None

    for i, row in enumerate(ds):
        # HF audio feature 返回 dict: {"path": ..., "array": ..., "sampling_rate": ...}
        audio_obj = row[audio_col]
        if isinstance(audio_obj, dict):
            audio_path = audio_obj.get("path", "")
        else:
            audio_path = str(audio_obj)

        text = str(row[text_col])

        item = {"audio": audio_path, "text": text}

        # 第一条的声音作为 ref_audio
        if ref_audio_path is None:
            ref_audio_path = audio_path

        item["ref_audio"] = ref_audio_path
        items.append(item)

    print(f"[HF] 加载了 {len(items)} 条 | ref_audio = {ref_audio_path}")
    return items


def load_local_parquet(
    parquet_url: str,
    audio_col: str,
    text_col: str,
    audio_dir: str = None,
    ref_audio_idx: int = 0,
    hf_repo: str = None,
):
    """
    本地 parquet + hf_hub_download 方式加载数据。
    解决 Windows Python 3.11+ multiprocessing RLock bug。

    Args:
        parquet_url:   metadata.parquet 的 HF 直链
        audio_col:     音频文件名/路径列名
        text_col:      文本列名
        audio_dir:     本地音频缓存目录（默认 datasets/audio/）
        ref_audio_idx: ref_audio 对应的索引
        hf_repo:       HF repo id（用于下载）
    Returns:
        items: list of dicts {"audio": "...", "text": "...", "ref_audio": "..."}
    """
    from huggingface_hub import hf_hub_download

    # 读取 parquet
    df = pd.read_parquet(parquet_url)
    print(f"[Local] parquet: {len(df)} 条 | 列名: {df.columns.tolist()}")

    if audio_col not in df.columns:
        raise ValueError(f"列 '{audio_col}' 不存在。可用列: {df.columns.tolist()}")
    if text_col not in df.columns:
        raise ValueError(f"列 '{text_col}' 不存在。可用列: {df.columns.tolist()}")

    # 确认 ref_audio
    ref_audio_filename = df.iloc[ref_audio_idx][audio_col]
    ref_audio_path = None

    items = []
    for i, row in df.iterrows():
        filename = row[audio_col]

        # Windows 路径兼容：反斜杠替换
        filename = str(filename).replace("\\", "/")
        filename = os.path.basename(filename)

        # 尝试本地路径（若已下载）
        if audio_dir:
            local_path = os.path.join(audio_dir, os.path.basename(filename))
            if os.path.exists(local_path):
                audio_path = local_path
            else:
                audio_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=f"train/{filename}",
                    repo_type="dataset",
                    cache_dir=audio_dir,
                )
        else:
            # 直接从 HF 下载
            audio_path = hf_hub_download(
                repo_id=hf_repo,
                filename=f"train/{filename}",
                repo_type="dataset",
            )

        # 记录 ref_audio
        if os.path.basename(filename) == os.path.basename(str(ref_audio_filename)):
            ref_audio_path = audio_path

        items.append({
            "audio": audio_path,
            "text": str(row[text_col]),
            "ref_audio": None,  # 后面统一填
        })

    # 统一填 ref_audio（用第一条的）
    if ref_audio_path is None and items:
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
    """
    用 Qwen3TTSTokenizer 把音频转成 audio_codes，输出 JSONL。
    """
    from qwen_tts import Qwen3TTSTokenizer

    print(f"[Tokenizer] 加载模型: {tokenizer_model_path}")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_path,
        device_map=device,
    )

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    # 分批处理
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

        # 处理剩余
        if batch_audios:
            _flush_batch(tokenizer, batch_items, batch_audios, f)

    print(f"✅ 已保存: {output_jsonl}")


def _flush_batch(tokenizer, batch_items, batch_audios, file_handle):
    """处理并写入一批"""
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
    parser.add_argument("--mode", type=str, choices=["hf", "local"], required=True,
                        help="hf=load_dataset 方式; local=parquet+hf_hub_download")
    parser.add_argument("--tokenizer_model_path", type=str,
                        default="../models/Qwen3-TTS-Tokenizer-12Hz",
                        help="Qwen3-TTS-Tokenizer-12Hz 模型路径或 HF repo id")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="输出路径，如 ../datasets/train_with_codes.jsonl")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="每批处理的音频数量（ tokenizer.encode 批次）")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 条（调试用）")

    # === HF 模式参数 ===
    parser.add_argument("--hf_repo", type=str,
                        help="HF repo id (mode=hf 时必填)")
    parser.add_argument("--hf_split", type=str, default="train",
                        help="split 名，默认 train")
    parser.add_argument("--audio_col", type=str, default="audio",
                        help="音频列名")
    parser.add_argument("--text_col", type=str, default="transcription",
                        help="文本列名")
    parser.add_argument("--ref_audio_idx", type=int, default=0,
                        help="用第几条数据的声音作为 ref_audio（默认0）")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF datasets 缓存目录")

    # === Local 模式参数 ===
    parser.add_argument("--parquet_url", type=str,
                        help="metadata.parquet 的 HF 直链 (mode=local 时必填)")
    parser.add_argument("--hf_repo_local", type=str,
                        help="HF repo id（用于 mode=local 的 hf_hub_download）")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="本地音频缓存目录")

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
            ref_audio_idx=args.ref_audio_idx,
            cache_dir=args.cache_dir,
        )
    else:
        if not args.parquet_url:
            raise ValueError("--parquet_url 在 mode=local 时必填")
        if not args.hf_repo_local:
            # 从 parquet_url 推断
            # https://huggingface.co/datasets/yeeko/Elaina_WanderingWitch_audio_JA/resolve/main/train/metadata.parquet
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
            audio_dir=args.audio_dir,
            ref_audio_idx=args.ref_audio_idx,
            hf_repo=args.hf_repo_local,
        )

    # 限制条数（调试用）
    if args.limit:
        items = items[:args.limit]
        print(f"[Debug] 仅处理前 {args.limit} 条")

    # === Tokenize & 保存 ===
    tokenize_and_save(
        items=items,
        tokenizer_model_path=args.tokenizer_model_path,
        output_jsonl=args.output_jsonl,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
