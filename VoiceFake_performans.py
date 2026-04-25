import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path("Datasetasv19/LA/LA")
PROTOCOL_DIR = BASE_DIR / "ASVspoof2019_LA_cm_protocols"
DEFAULT_AUDIO_DIR = BASE_DIR / "ASVspoof2019_LA_dev" / "flac"
DEFAULT_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"

SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT = 512


class SegmentAttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.attn = nn.Linear(128, 1)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor, return_segment_logits: bool = False):
        bsz, segs, c, h, w = x.shape
        x = x.view(bsz * segs, c, h, w)
        x = self.features(x).flatten(1)
        x = self.embedding(x)
        x = x.view(bsz, segs, -1)

        segment_logits = self.classifier(x)
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        clip_emb = torch.sum(x * weights.unsqueeze(-1), dim=1)
        clip_logits = self.classifier(clip_emb)
        if return_segment_logits:
            return clip_logits, segment_logits
        return clip_logits


def parse_protocol(
    protocol_file: Path,
    audio_dir: Path,
    max_files: int,
    seed: int,
    file_id_col: int = 1,
    label_col: int = -1,
    real_tag: str = "bonafide",
) -> List[Tuple[Path, str, int]]:
    if not protocol_file.is_file():
        raise FileNotFoundError(f"Protocol bulunamadi: {protocol_file}")

    rows: List[Tuple[Path, str, int]] = []
    real_tag = real_tag.lower().strip()
    with protocol_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                file_id = parts[file_id_col]
                label_text = parts[label_col].lower()
            except IndexError:
                continue

            wav_path = audio_dir / f"{file_id}.flac"
            if not wav_path.is_file():
                continue
            label = 0 if label_text == real_tag else 1
            rows.append((wav_path, file_id, label))

    if not rows:
        raise RuntimeError("Protocol ile eslesen ses dosyasi bulunamadi.")

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    if max_files is not None and max_files < len(rows):
        rows = rows[:max_files]
    return rows


def infer_label_from_path(path: Path):
    p = str(path).lower()
    real_keys = ["bonafide", "real", "genuine", "human"]
    fake_keys = ["spoof", "fake", "deepfake", "synthetic", "synth", "ai", "tts", "vc"]

    has_real = any(k in p for k in real_keys)
    has_fake = any(k in p for k in fake_keys)
    if has_real and not has_fake:
        return 0
    if has_fake and not has_real:
        return 1
    return None


def build_samples_from_paths(audio_dir: Path, max_files: int, seed: int):
    rows = []
    for p in audio_dir.rglob("*.flac"):
        y = infer_label_from_path(p)
        if y is None:
            continue
        rows.append((p, p.stem, y))

    if not rows:
        raise RuntimeError(
            "Path etiketleme ile real/fake ayrimi bulunamadi. "
            "Klasor/isim icinde bonafide/real veya fake/spoof anahtar kelimeleri olmali."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    if max_files is not None and max_files < len(rows):
        rows = rows[:max_files]
    return rows


def extract_segments_to_logmel(wav_path: Path, segment_sec: float, segments_per_file: int) -> np.ndarray:
    wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    segment_len = int(segment_sec * SAMPLE_RATE)
    target_frames = int(np.ceil(segment_len / HOP_LENGTH))

    if len(wav) == 0:
        wav = np.zeros(segment_len, dtype=np.float32)

    max_start = max(len(wav) - segment_len, 0)
    if max_start == 0:
        starts = [0] * segments_per_file
    elif segments_per_file == 1:
        starts = [max_start // 2]
    else:
        starts = np.linspace(0, max_start, segments_per_file).astype(int).tolist()

    segments = []
    for s in starts:
        seg = wav[s : s + segment_len]
        if len(seg) < segment_len:
            seg = np.pad(seg, (0, segment_len - len(seg)))

        mel = librosa.feature.melspectrogram(
            y=seg,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            fmin=20,
            fmax=7600,
            power=2.0,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = librosa.util.fix_length(mel, size=target_frames, axis=1)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        segments.append(mel.astype(np.float32))

    return np.stack(segments, axis=0)[:, None, :, :]


def metric_block(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def run_eval(model: SegmentAttentionCNN, samples, threshold: float, segment_sec: float, segments_per_file: int):
    file_true, file_pred = [], []
    seg_true, seg_pred, seg_fake_prob = [], [], []
    file_rows, seg_rows = [], []

    for path, file_id, y_true in tqdm(samples, desc="Evaluating"):
        x_np = extract_segments_to_logmel(path, segment_sec=segment_sec, segments_per_file=segments_per_file)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            clip_logits, seg_logits = model(x, return_segment_logits=True)
            file_fake_prob = float(torch.softmax(clip_logits, dim=1)[0, 1].item())
            seg_probs = torch.softmax(seg_logits, dim=2)[0, :, 1].cpu().numpy()

        file_class = int(file_fake_prob >= threshold)
        seg_classes = (seg_probs >= threshold).astype(np.int64)

        file_true.append(y_true)
        file_pred.append(file_class)

        seg_count = int(len(seg_classes))
        fake_ratio = float(np.mean(seg_classes == 1))
        file_rows.append(
            {
                "file_id": file_id,
                "path": str(path),
                "true_label": y_true,
                "pred_label": file_class,
                "fake_probability": file_fake_prob,
                "segment_count": seg_count,
                "segment_fake_ratio": fake_ratio,
            }
        )

        for idx in range(seg_count):
            sec_start = idx * segment_sec
            sec_end = sec_start + segment_sec
            seg_true.append(y_true)
            seg_pred.append(int(seg_classes[idx]))
            seg_fake_prob.append(float(seg_probs[idx]))
            seg_rows.append(
                {
                    "file_id": file_id,
                    "path": str(path),
                    "segment_index": idx,
                    "start_sec": sec_start,
                    "end_sec": sec_end,
                    "true_label": y_true,
                    "pred_label": int(seg_classes[idx]),
                    "fake_probability": float(seg_probs[idx]),
                }
            )

    return {
        "file_true": np.array(file_true),
        "file_pred": np.array(file_pred),
        "seg_true": np.array(seg_true),
        "seg_pred": np.array(seg_pred),
        "seg_fake_prob": np.array(seg_fake_prob),
        "file_df": pd.DataFrame(file_rows),
        "seg_df": pd.DataFrame(seg_rows),
    }


def save_plots(eval_out, run_dir: Path):
    labels = ["Real(0)", "Fake(1)"]

    for level, y_true, y_pred in [
        ("file", eval_out["file_true"], eval_out["file_pred"]),
        ("segment", eval_out["seg_true"], eval_out["seg_pred"]),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(f"{level.capitalize()} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(run_dir / f"confusion_{level}.png", dpi=180)
        plt.close(fig)

    seg_df = eval_out["seg_df"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=seg_df,
        x="fake_probability",
        hue="true_label",
        bins=30,
        stat="density",
        common_norm=False,
        palette={0: "#1f77b4", 1: "#d62728"},
        ax=ax,
    )
    ax.set_title("Segment Fake Probability Distribution")
    ax.set_xlabel("P(fake)")
    ax.legend(["Real", "Fake"], title="True Label")
    fig.tight_layout()
    fig.savefig(run_dir / "segment_probability_distribution.png", dpi=180)
    plt.close(fig)

    summary = (
        eval_out["file_df"]
        .groupby("true_label")["segment_fake_ratio"]
        .describe()[["mean", "std", "min", "max"]]
        .reset_index()
    )
    summary["true_label"] = summary["true_label"].map({0: "Real", 1: "Fake"})
    bar_palette = {"Real": "#1f77b4", "Fake": "#d62728"}
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=summary, x="true_label", y="mean", hue="true_label", palette=bar_palette, legend=False, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Average Segment Fake Ratio by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Mean fake-segment ratio")
    fig.tight_layout()
    fig.savefig(run_dir / "segment_fake_ratio_by_class.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="precision_best.pth icin performans olcumu (performansVF.py ciktilari ile uyumlu)"
    )
    parser.add_argument(
        "--model-path",
        default=r"D:\proje klasörleri\CV_DeepFake\Voice_DF\checkpoints\precision_best.pth",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(DEFAULT_AUDIO_DIR),
    )
    parser.add_argument(
        "--protocol-file",
        default=str(DEFAULT_PROTOCOL),
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\proje klasörleri\CV_DeepFake\Voice_DF\outputs",
    )
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-mode",
        choices=["protocol", "path"],
        default="protocol",
    )
    parser.add_argument("--real-tag", default="bonafide")
    parser.add_argument("--protocol-file-id-col", type=int, default=1)
    parser.add_argument("--protocol-label-col", type=int, default=-1)
    parser.add_argument("--segment-sec", type=float, default=2.0)
    parser.add_argument("--segments-per-eval-file", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None, help="Verilirse checkpoint threshold degerini override eder")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    audio_dir = Path(args.audio_dir)
    protocol_file = Path(args.protocol_file)
    base_output_dir = Path(args.output_dir)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio klasoru bulunamadi: {audio_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"precision_performance_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cihaz: {DEVICE}")
    print(f"Model yukleniyor: {model_path}")

    model = SegmentAttentionCNN().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    ckpt_threshold = 0.5
    ckpt_epoch = None

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=True)
        ckpt_threshold = float(state.get("threshold", 0.5))
        ckpt_epoch = state.get("epoch", None)
    else:
        model.load_state_dict(state, strict=True)

    used_threshold = float(args.threshold) if args.threshold is not None else ckpt_threshold
    model.eval()

    if args.label_mode == "protocol":
        samples = parse_protocol(
            protocol_file=protocol_file,
            audio_dir=audio_dir,
            max_files=args.max_files,
            seed=args.seed,
            file_id_col=args.protocol_file_id_col,
            label_col=args.protocol_label_col,
            real_tag=args.real_tag,
        )
        print(
            f"Etiket modu: protocol | real_tag='{args.real_tag}', "
            f"file_id_col={args.protocol_file_id_col}, label_col={args.protocol_label_col}"
        )
    else:
        samples = build_samples_from_paths(audio_dir, args.max_files, args.seed)
        print("Etiket modu: path (dosya/klasor adindan real-fake)")

    real_n = sum(1 for _, _, y in samples if y == 0)
    fake_n = sum(1 for _, _, y in samples if y == 1)
    print(f"Kullanilan dosya sayisi: {len(samples)}")
    print(f"Dagilim -> Real: {real_n} | Fake: {fake_n}")
    print(f"Kullanilan threshold: {used_threshold:.4f}")

    eval_out = run_eval(
        model=model,
        samples=samples,
        threshold=used_threshold,
        segment_sec=args.segment_sec,
        segments_per_file=args.segments_per_eval_file,
    )

    file_metrics = metric_block(eval_out["file_true"], eval_out["file_pred"])
    seg_metrics = metric_block(eval_out["seg_true"], eval_out["seg_pred"])

    metrics = {
        "config": {
            "model_path": str(model_path),
            "audio_dir": str(audio_dir),
            "protocol_file": str(protocol_file),
            "max_files": args.max_files,
            "used_files": int(len(samples)),
            "sample_rate": SAMPLE_RATE,
            "segment_sec": args.segment_sec,
            "segments_per_eval_file": args.segments_per_eval_file,
            "device": DEVICE,
            "created_at": datetime.now().isoformat(),
        },
        "checkpoint_info": {
            "checkpoint_threshold": ckpt_threshold,
            "used_threshold": used_threshold,
            "checkpoint_epoch": ckpt_epoch,
        },
        "file_level": file_metrics,
        "segment_level": seg_metrics,
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    eval_out["file_df"].to_csv(run_dir / "file_predictions.csv", index=False, encoding="utf-8")
    eval_out["seg_df"].to_csv(run_dir / "segment_predictions.csv", index=False, encoding="utf-8")
    save_plots(eval_out, run_dir)

    print("\n=== FILE LEVEL ===")
    print(json.dumps(file_metrics, indent=2))
    print("\n=== SEGMENT LEVEL ===")
    print(json.dumps(seg_metrics, indent=2))
    print(f"\nTum ciktilar: {run_dir}")


if __name__ == "__main__":
    main()
