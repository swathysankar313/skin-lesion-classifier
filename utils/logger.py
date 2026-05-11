"""
utils/logger.py

CSV + TensorBoard training logger.
"""

import csv
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from config import LOG_DIR


class Logger:
    """Writes per-epoch metrics to CSV and TensorBoard simultaneously."""

    def __init__(self, log_dir: Path = LOG_DIR):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "training_log.csv"
        self.writer   = SummaryWriter(log_dir=str(log_dir / "tb"))

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc",
                "val_loss", "val_acc", "val_f1"
            ])

    def log(self, epoch: int, train_loss: float, train_acc: float,
            val_loss: float, val_acc: float, val_f1: float):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}",
            ])

        self.writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        self.writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        self.writer.add_scalar("Val/F1",    val_f1,  epoch)

    def close(self):
        self.writer.close()