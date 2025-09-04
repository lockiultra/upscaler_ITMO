from __future__ import annotations

import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from upscaler.data.dataset import CurriculumSampler
from upscaler.config import CSV_FILE, DATA_FOLDER, DEVICE
from upscaler.data.dataset import (
    ProteinUpscalingDataset,
    collate_batch,
)
from upscaler.model.upscaler import ProteinUpscaler
from upscaler.loss.loss import ProteinUpscalingLoss
from upscaler.utils.metrics import QualityMetrics
from upscaler.training.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_optimizer(model: nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


def get_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    checkpoint_dir: str,
    pipeline_state: dict | None = None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": float(metric),
    }
    if pipeline_state is not None:
        payload["pipeline_state"] = pipeline_state
    torch.save(payload, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    pipeline: TrainingPipeline | None = None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if pipeline is not None and "pipeline_state" in checkpoint:
        pipeline.load_state_dict(checkpoint["pipeline_state"])
    epoch = int(checkpoint.get("epoch", 0))
    metric = float(checkpoint.get("metric", 0.0))
    logger.info(f"Loaded checkpoint {checkpoint_path} (epoch {epoch}, metric {metric:.4f})")
    return epoch, metric


def build_dataloaders(
    dataset: ProteinUpscalingDataset,
    batch_size: int,
    device: torch.device,
    use_bucket: bool = True,
    num_workers: int = 4,
    use_curriculum: bool = False,
    pin_memory: bool | None = None,
):
    """
    Build train/val DataLoaders. Uses BucketBatchSampler for train to reduce padding.
    Returns (train_loader, val_loader).
    """
    if pin_memory is None:
        pin_memory = True if device.type == "cuda" else False

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    indices = list(range(total))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    if use_curriculum:
        train_loader = None  # Will be handled per-epoch
    elif use_bucket:
        lengths = [dataset.get_num_atoms(i) for i in train_indices]
        pairs = list(zip(train_indices, lengths))
        pairs.sort(key=lambda x: x[1])
        batches = []
        for i in range(0, len(pairs), batch_size):
            batch = [p[0] for p in pairs[i : i + batch_size]]
            batches.append(batch)
        train_loader = DataLoader(
            dataset,
            batch_sampler=batches,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=max(0, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=(num_workers // 2 > 0),
    )

    return train_loader, val_loader, train_subset, val_subset


def train_model(
    csv_file: str,
    data_folder: str,
    checkpoint_dir: str = "checkpoints",
    num_epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: torch.device = DEVICE,
    resume_from_checkpoint: str | None = None,
    use_amp: bool = True,
    use_bucket: bool = True,
    use_curriculum: bool = False,
):
    """
    High-level training loop. Uses TrainingPipeline internally.
    Saves best model according to validation RMSD.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Loading dataset...")
    dataset = ProteinUpscalingDataset(csv_file, data_folder)

    # build loaders
    train_loader, val_loader, _, _ = build_dataloaders(
        dataset, batch_size, device, use_bucket=use_bucket
    )

    # init model
    logger.info("Initializing model...")
    model = ProteinUpscaler()
    model.to(device)

    loss_fn = ProteinUpscalingLoss(device=device)
    optimizer = get_optimizer(model, lr=lr)
    scheduler = get_scheduler(optimizer)
    metrics_calculator = QualityMetrics(device=device)

    pipeline = TrainingPipeline(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=1.0,
        metrics_calculator=metrics_calculator,
        scheduler=scheduler,
        use_amp=use_amp,
    )

    start_epoch = 0
    if resume_from_checkpoint:
        if os.path.exists(resume_from_checkpoint):
            start_epoch, _ = load_checkpoint(model, optimizer, resume_from_checkpoint, pipeline=pipeline)
            start_epoch += 1
        else:
            logger.warning(f"resume_from_checkpoint path not found: {resume_from_checkpoint}")

    logger.info("Starting training...")
    best_val_metric = float("inf")

    for epoch in range(start_epoch, num_epochs):
        if use_curriculum:
            logger.info(f"Epoch {epoch+1}/{num_epochs} (Curriculum mode)")
            curriculum_sampler = CurriculumSampler(dataset, batch_size=batch_size)
            epoch_iterator = curriculum_sampler.get_epoch_iterator(epoch)

            total_loss = 0.0
            total_metrics = {
                "coord_rmsd": 0.0,
                "lddt": 0.0,
                "clash": 0.0,
                "physics": 0.0,
                "total": 0.0,
            }
            num_batches = 0
            for batch_indices in epoch_iterator:
                batch_samples = [dataset[i] for i in batch_indices]
                batch = collate_batch(batch_samples)
                loss, metrics = pipeline.train_one_batch(batch)
                total_loss += float(loss.cpu().item())
                for k in total_metrics.keys():
                    if k in metrics:
                        total_metrics[k] += float(metrics[k].cpu().item())
                    else:
                        total_metrics[k] += 0.0
                num_batches += 1
            avg_loss = total_loss / num_batches if num_batches else 0.0
            avg_metrics = {k: v / num_batches if num_batches else 0.0 for k, v in total_metrics.items()}
            avg_metrics["loss"] = avg_loss
            train_metrics = avg_metrics
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            train_metrics = pipeline.train_epoch(train_loader)
            logger.info(f"Train metrics: {train_metrics}")

        val_metrics = pipeline.validate(val_loader)
        logger.info(f"Val metrics: {val_metrics}")

        if scheduler is not None:
            try:
                scheduler.step(val_metrics["val_rmsd"])
            except Exception:
                # fallback if scheduler expects loss
                scheduler.step(train_metrics["loss"])

        # Save best by validation RMSD
        if val_metrics["val_rmsd"] < best_val_metric:
            best_val_metric = val_metrics["val_rmsd"]
            pipeline_state = pipeline.state_dict()
            save_checkpoint(model, optimizer, epoch, best_val_metric, checkpoint_dir, pipeline_state)
            logger.info(f"New best model saved (val_rmsd={best_val_metric:.4f})")

        # periodic save every 10 epochs
        if (epoch + 1) % 10 == 0:
            pipeline_state = pipeline.state_dict()
            save_checkpoint(model, optimizer, epoch, val_metrics["val_rmsd"], checkpoint_dir, pipeline_state)

    logger.info("Training finished.")


if __name__ == "__main__":
    train_model(
        csv_file=CSV_FILE,
        data_folder=DATA_FOLDER,
        checkpoint_dir="checkpoints",
        num_epochs=50,
        batch_size=4,
        lr=1e-4,
        device=DEVICE,
        resume_from_checkpoint=None,
        use_amp=True,
        use_bucket=True,
        use_curriculum=True,
    )
