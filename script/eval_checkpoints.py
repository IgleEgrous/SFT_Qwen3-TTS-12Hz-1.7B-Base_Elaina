# coding=utf-8
"""
eval_checkpoints.py — 多 checkpoint 横向对比工具

用法：
    python script/eval_checkpoints.py \
        --checkpoints output/checkpoint-epoch-2 output/checkpoint-epoch-4 output/checkpoint-epoch-8 \
        --ref_audio path/to/ref.wav \
        --test_texts "今天天气真好" "学校行くのは嫌だけど" "你好啊，我是伊蕾娜" \
        --output_dir output/eval_samples \
        --speaker elaina
"""
import argparse
import os

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def load_model(checkpoint_path: str, device: str = "cuda:0"):
    """加载单个 checkpoint"""
    print(f"加载: {checkpoint_path}")
    model = Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return model


def generate_samples(model, ref_audio: str, test_texts: list, speaker: str,
                     device: str = "cuda:0") -> dict:
    """用单个模型生成所有测试文本的音频"""
    results = {}
    for text in test_texts:
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            ref_audio=ref_audio,
        )
        results[text] = (wavs[0], sr)
    return results


def main():
    parser = argparse.ArgumentParser(description="多 checkpoint 横向对比")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="要对比的 checkpoint 路径列表")
    parser.add_argument("--ref_audio", type=str, required=True,
                        help="参考音频路径（24kHz WAV）")
    parser.add_argument("--test_texts", type=str, nargs="+", required=True,
                        help="测试文本列表")
    parser.add_argument("--output_dir", type=str, default="../outputs/eval_samples",
                        help="输出目录")
    parser.add_argument("--speaker", type=str, default="elaina",
                        help="speaker 名称（训练时的 --speaker_name）")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 每个 checkpoint 一个子目录
    for ckpt in args.checkpoints:
        ckpt_name = os.path.basename(os.path.normpath(ckpt))
        ckpt_dir = os.path.join(args.output_dir, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        model = load_model(ckpt, device=args.device)

        for text in args.test_texts:
            wavs, sr = model.generate_custom_voice(
                text=text,
                speaker=args.speaker,
                ref_audio=args.ref_audio,
            )
            # 文件名：把文本前20字做sanitize
            safe_text = text[:20].replace(" ", "_").replace("\n", "_")
            out_path = os.path.join(ckpt_dir, f"{safe_text}.wav")
            sf.write(out_path, wavs[0], sr)
            print(f"  [{ckpt_name}] {text[:30]}... -> {out_path}")

        # 清理显存
        del model
        torch.cuda.empty_cache()

    print(f"\n✅ 对比音频已保存到: {args.output_dir}")
    print("建议用 Audacity 或音频播放器横向对比同一文本在不同 epoch 下的效果")


if __name__ == "__main__":
    main()
