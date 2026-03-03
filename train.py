"""
train.py — entry point for IMDB sentiment training.
"""

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import IMDBDataset, build_vocab, load_imdb
from model import BiLSTMClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log"),
    ],
)
log = logging.getLogger(__name__)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def train(args):
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info("Loading IMDB dataset ...")
    train_texts, val_texts, test_texts, \
    train_labels, val_labels, test_labels = load_imdb(seed=args.seed)

    log.info(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    vocab = build_vocab(train_texts, max_vocab=args.vocab_size)
    log.info(f"Vocabulary size: {len(vocab)}")

    train_ds = IMDBDataset(train_texts, train_labels, vocab, args.max_len)
    val_ds   = IMDBDataset(val_texts,   val_labels,   vocab, args.max_len)
    test_ds  = IMDBDataset(test_texts,  test_labels,  vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    writer = SummaryWriter(log_dir="logs/tensorboard")

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
        t0 = time.time()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            epoch_correct += (preds == yb).sum().item()
            epoch_n += len(yb)

        train_loss = epoch_loss / epoch_n
        train_acc  = epoch_correct / epoch_n
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = f"checkpoints/best_epoch{epoch:02d}_acc{val_acc:.4f}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "args": vars(args),
                "vocab_size": len(vocab),
            }, ckpt_path)
            log.info(f"  Checkpoint saved -> {ckpt_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping triggered (patience={args.patience})")
                break

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    log.info(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f}")
    writer.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--embed_dim",    type=int,   default=128)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--max_len",      type=int,   default=256)
    p.add_argument("--vocab_size",   type=int,   default=20000)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=3)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    train(parse_args())