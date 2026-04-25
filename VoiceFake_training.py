import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path("Datasetasv19/LA/LA")
PROTOCOL_DIR = BASE_DIR / "ASVspoof2019_LA_cm_protocols"
TRAIN_AUDIO_DIR = BASE_DIR / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO_DIR = BASE_DIR / "ASVspoof2019_LA_dev" / "flac"
TRAIN_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"


@dataclass
class Sample:
    speaker_id: str
    utt_id: str
    attack_id: str
    label: int
    wav_path: Path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_protocol(protocol_file: Path, audio_dir: Path, max_files: int = None, seed: int = 42) -> List[Sample]:
    rows: List[Sample] = []
    with protocol_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, utt_id, _, attack_id, key = parts[:5]
            wav_path = audio_dir / f"{utt_id}.flac"
            if not wav_path.exists():
                continue
            label = 0 if key == "bonafide" else 1
            rows.append(
                Sample(
                    speaker_id=speaker_id,
                    utt_id=utt_id,
                    attack_id=attack_id,
                    label=label,
                    wav_path=wav_path,
                )
            )
    if not rows:
        raise RuntimeError(f"No usable samples in {protocol_file}")
    if max_files is not None and max_files < len(rows):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=max_files, replace=False)
        rows = [rows[int(i)] for i in idx]
    return rows


class SegmentASVDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        segment_sec: float = 2.0,
        segments_per_file: int = 6,
        train: bool = True,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 160,
        win_length: int = 400,
        n_fft: int = 512,
    ):
        self.samples = samples
        self.segment_sec = segment_sec
        self.segments_per_file = segments_per_file
        self.train = train
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.segment_len = int(segment_sec * sample_rate)
        self.target_frames = int(np.ceil(self.segment_len / hop_length))

    def __len__(self) -> int:
        return len(self.samples)

    def _get_starts(self, total_len: int) -> List[int]:
        max_start = max(total_len - self.segment_len, 0)
        if max_start == 0:
            return [0] * self.segments_per_file

        if self.train:
            return [random.randint(0, max_start) for _ in range(self.segments_per_file)]

        if self.segments_per_file == 1:
            return [max_start // 2]
        return np.linspace(0, max_start, self.segments_per_file).astype(int).tolist()

    def _augment_wave(self, seg: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            seg = librosa.effects.time_stretch(seg, rate=np.random.uniform(0.95, 1.05))
        if random.random() < 0.3:
            seg = librosa.effects.pitch_shift(seg, sr=self.sample_rate, n_steps=np.random.uniform(-0.5, 0.5))
        if random.random() < 0.7:
            seg = seg + np.random.randn(len(seg)) * np.random.uniform(0.0005, 0.0025)
        return seg

    def _specaugment(self, mel: np.ndarray) -> np.ndarray:
        if not self.train:
            return mel
        out = mel.copy()

        if random.random() < 0.6:
            f = random.randint(4, 12)
            f0 = random.randint(0, max(out.shape[0] - f, 0))
            out[f0:f0 + f, :] = 0
        if random.random() < 0.6:
            t = random.randint(6, 18)
            t0 = random.randint(0, max(out.shape[1] - t, 0))
            out[:, t0:t0 + t] = 0
        return out

    def _wave_to_logmel(self, seg: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=seg,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=20,
            fmax=7600,
            power=2.0,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = librosa.util.fix_length(mel, size=self.target_frames, axis=1)
        mel = self._specaugment(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        wav, _ = librosa.load(sample.wav_path, sr=self.sample_rate)
        if len(wav) == 0:
            wav = np.zeros(self.segment_len, dtype=np.float32)

        segments = []
        for s in self._get_starts(len(wav)):
            seg = wav[s:s + self.segment_len]
            if len(seg) < self.segment_len:
                seg = np.pad(seg, (0, self.segment_len - len(seg)))
            if self.train:
                seg = self._augment_wave(seg)
                if len(seg) < self.segment_len:
                    seg = np.pad(seg, (0, self.segment_len - len(seg)))
                elif len(seg) > self.segment_len:
                    seg = seg[:self.segment_len]
            mel = self._wave_to_logmel(seg)
            segments.append(mel)

        x = np.stack(segments, axis=0)[:, None, :, :]
        y = np.int64(sample.label)
        return torch.tensor(x), torch.tensor(y)


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

        segment_logits = self.classifier(x)  # [B, S, 2]
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        clip_emb = torch.sum(x * weights.unsqueeze(-1), dim=1)
        clip_logits = self.classifier(clip_emb)
        if return_segment_logits:
            return clip_logits, segment_logits
        return clip_logits


class WeightedFocalCE(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0, label_smoothing: float = 0.03):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "bonafide_precision": float(precision[0]),
        "spoof_precision": float(precision[1]),
        "bonafide_recall": float(recall[0]),
        "spoof_recall": float(recall[1]),
        "confusion_matrix": cm,
    }


def compute_selection_score(metrics: dict) -> float:
    # Equal emphasis to macro-F1 and balanced accuracy, with extra push for both recalls.
    mean_recall = 0.5 * (metrics["bonafide_recall"] + metrics["spoof_recall"])
    return float(0.45 * metrics["macro_f1"] + 0.45 * metrics["balanced_accuracy"] + 0.10 * mean_recall)


def compute_threshold(y_true: np.ndarray, spoof_probs: np.ndarray) -> Tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = None
    best_score = -1.0

    for thr in np.linspace(0.05, 0.95, 181):
        y_pred = (spoof_probs >= thr).astype(np.int64)
        metrics = compute_metrics(y_true, y_pred)
        score = compute_selection_score(metrics)
        better = score > best_score + 1e-9
        tie_break = abs(score - best_score) <= 1e-9 and metrics["macro_f1"] > (best_metrics["macro_f1"] if best_metrics else -1.0)
        if better or tie_break:
            best_threshold = float(thr)
            best_metrics = metrics
            best_score = score

    best_metrics["selection_score"] = float(best_score)
    return best_threshold, best_metrics


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module):
    model.eval()
    losses = []
    all_y = []
    all_file_probs = []
    all_seg_y = []
    all_seg_probs = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits, seg_logits = model(x, return_segment_logits=True)
            loss = loss_fn(logits, y)
            file_probs = torch.softmax(logits, dim=1)[:, 1]
            seg_probs = torch.softmax(seg_logits, dim=2)[:, :, 1]

            losses.append(loss.item())
            all_y.append(y.cpu().numpy())
            all_file_probs.append(file_probs.cpu().numpy())
            all_seg_probs.append(seg_probs.reshape(-1).cpu().numpy())
            all_seg_y.append(y.unsqueeze(1).repeat(1, seg_probs.shape[1]).reshape(-1).cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    y_true = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    file_probs = np.concatenate(all_file_probs) if all_file_probs else np.array([], dtype=np.float32)
    seg_y_true = np.concatenate(all_seg_y) if all_seg_y else np.array([], dtype=np.int64)
    seg_probs = np.concatenate(all_seg_probs) if all_seg_probs else np.array([], dtype=np.float32)
    return avg_loss, y_true, file_probs, seg_y_true, seg_probs


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    model.train()
    losses = []
    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def build_weighted_sampler(samples: List[Sample]) -> WeightedRandomSampler:
    labels = np.array([s.label for s in samples], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(samples),
        replacement=True,
    )


def save_log(path: Path, row: dict) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "precision_best.pth"
    summary_path = out_dir / "precision_summary.json"
    log_path = out_dir / "precision_training_log.csv"

    train_samples = parse_protocol(TRAIN_PROTOCOL, TRAIN_AUDIO_DIR, max_files=args.max_train_files, seed=args.seed)
    dev_samples = parse_protocol(DEV_PROTOCOL, DEV_AUDIO_DIR, max_files=args.max_dev_files, seed=args.seed + 1)

    train_dataset = SegmentASVDataset(
        train_samples,
        segment_sec=args.segment_sec,
        segments_per_file=args.segments_per_train_file,
        train=True,
    )
    dev_dataset = SegmentASVDataset(
        dev_samples,
        segment_sec=args.segment_sec,
        segments_per_file=args.segments_per_eval_file,
        train=False,
    )

    sampler = build_weighted_sampler(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=DEVICE == "cuda",
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=DEVICE == "cuda",
    )

    class_counts = np.bincount(np.array([s.label for s in train_samples], dtype=np.int64), minlength=2).astype(np.float32)
    class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1.0), dtype=torch.float32, device=DEVICE)
    class_weights = class_weights / class_weights.sum() * 2.0

    model = SegmentAttentionCNN().to(DEVICE)
    loss_fn = WeightedFocalCE(class_weights=class_weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {DEVICE}")
    print(f"Train samples: {len(train_samples)} | Dev samples: {len(dev_samples)}")
    print(f"Class counts [bonafide, spoof]: {class_counts.tolist()}")

    best = {
        "epoch": -1,
        "threshold": 0.5,
        "val_loss": float("inf"),
        "file_metrics": {
            "selection_score": -1.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "bonafide_recall": 0.0,
            "spoof_recall": 0.0,
        },
        "segment_metrics": {
            "selection_score": -1.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "bonafide_recall": 0.0,
            "spoof_recall": 0.0,
        },
    }
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, y_true, file_probs, seg_y_true, seg_probs = evaluate(model, dev_loader, loss_fn)
        thr, file_metrics = compute_threshold(y_true, file_probs)
        seg_pred = (seg_probs >= thr).astype(np.int64)
        segment_metrics = compute_metrics(seg_y_true, seg_pred)
        segment_metrics["selection_score"] = compute_selection_score(segment_metrics)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "threshold": thr,
            "file_macro_f1": file_metrics["macro_f1"],
            "file_balanced_accuracy": file_metrics["balanced_accuracy"],
            "file_bonafide_recall": file_metrics["bonafide_recall"],
            "file_spoof_recall": file_metrics["spoof_recall"],
            "segment_macro_f1": segment_metrics["macro_f1"],
            "segment_balanced_accuracy": segment_metrics["balanced_accuracy"],
            "segment_bonafide_recall": segment_metrics["bonafide_recall"],
            "segment_spoof_recall": segment_metrics["spoof_recall"],
            "selection_score": 0.6 * file_metrics["selection_score"] + 0.4 * segment_metrics["selection_score"],
        }
        save_log(log_path, row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"thr={thr:.3f} file(mF1={file_metrics['macro_f1']:.4f}, bAcc={file_metrics['balanced_accuracy']:.4f}) "
            f"seg(mF1={segment_metrics['macro_f1']:.4f}, bAcc={segment_metrics['balanced_accuracy']:.4f})"
        )

        current_score = 0.6 * file_metrics["selection_score"] + 0.4 * segment_metrics["selection_score"]
        best_score = 0.6 * best["file_metrics"]["selection_score"] + 0.4 * best["segment_metrics"]["selection_score"]
        improved = current_score > best_score + 1e-6
        tie_break = abs(current_score - best_score) <= 1e-6 and file_metrics["macro_f1"] > best["file_metrics"]["macro_f1"]
        if improved or tie_break:
            best = {
                "epoch": epoch,
                "threshold": thr,
                "val_loss": val_loss,
                "file_metrics": file_metrics,
                "segment_metrics": segment_metrics,
            }
            torch.save({"model": model.state_dict(), "threshold": thr, "epoch": epoch}, ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping: precision-oriented metric stopped improving.")
                break

    summary = {
        "best_epoch": best["epoch"],
        "best_threshold": best["threshold"],
        "best_val_loss": best["val_loss"],
        "best_file_metrics": best["file_metrics"],
        "best_segment_metrics": best["segment_metrics"],
        "best_global_selection_score": 0.6 * best["file_metrics"]["selection_score"] + 0.4 * best["segment_metrics"]["selection_score"],
        "config": vars(args),
        "train_class_counts": class_counts.tolist(),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training finished.")
    print(json.dumps(summary, indent=2))


def get_args():
    parser = argparse.ArgumentParser(description="Precision-oriented segment-based ASVspoof 2019 LA training")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--segment-sec", type=float, default=2.0)
    parser.add_argument("--segments-per-train-file", type=int, default=6)
    parser.add_argument("--segments-per-eval-file", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-dev-files", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
