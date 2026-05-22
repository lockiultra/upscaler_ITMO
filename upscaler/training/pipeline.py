from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _make_grad_scaler(enabled: bool):
    """Backwards-compatible GradScaler factory.
    """
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    """Аналогично — новый torch.amp.autocast, fallback на старый."""
    try:
        return torch.amp.autocast('cuda', enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.autocast(enabled=enabled)


class TrainingPipeline:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        clip_grad_norm: float | None = 1.0,
        metrics_calculator=None,
        scheduler: Any | None = None,
        use_amp: bool = True,
        log_every: int = 10,
    ):
        """
        Args:
            model, loss_fn, optimizer: usual PyTorch objects.
            device: torch.device
            clip_grad_norm: if not None, will clip gradients to this norm.
            metrics_calculator: object with compute_rmsd/compute_lddt/compute_clash_score methods.
            scheduler: learning rate scheduler (can be stepped externally).
            use_amp: enable mixed precision (only effective on CUDA).
            log_every: how often (in batches) to emit info logs.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.metrics_calculator = metrics_calculator
        self.scheduler = scheduler
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.log_every = log_every

        # Ensure model on device
        self.model.to(self.device)

        # GradScaler for AMP
        self.scaler = _make_grad_scaler(enabled=self.use_amp)

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Train one epoch by repeatedly calling train_one_batch().
        Returns averaged metrics over the dataloader.
        """
        self.model.train()

        total_loss = 0.0
        total_metrics = {
            "coord_rmsd": 0.0,
            "lddt": 0.0,
            "clash": 0.0,
            "physics": 0.0,
            "total": 0.0,
            "loss": 0.0,
        }

        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):

            loss, metrics = self.train_one_batch(batch)

            total_loss += float(loss)

            for k in total_metrics:
                if k in metrics:
                    total_metrics[k] += float(metrics[k].detach().cpu().item())
                else:
                    total_metrics[k] += 0.0

            if (batch_idx % self.log_every) == 0:
                logger.info(f"Train Batch {batch_idx}/{num_batches} — loss {float(loss):.4f}")

        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics["loss"] = total_loss / num_batches

        return avg_metrics


    def train_one_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train on a single batch and return ``(loss, metrics)``."""
        batch = {
            k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        coords_good = batch["coords_good"]
        atom_types = batch["atom_types"]
        atom_names = batch["atom_names"]
        res_map = batch["res_map"]

        self.optimizer.zero_grad(set_to_none=True)

        with _autocast(self.use_amp):
            pred_data = self.model(batch)

            true_data = {
                'rots': batch["rots_good"],
                'trans': batch["trans_good"],
                'coords': coords_good,
                'atom_types': atom_types,
                'atom_names': atom_names,
                'res_map': res_map,
                'res_chain': batch.get("res_chain"),
            }
            loss, metrics = self.loss_fn(pred_data, true_data)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

        metrics['total'] = loss.detach()
        return loss.detach(), {k: v.detach() for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Run validation loop. Returns averaged metrics dict:
        { 'val_rmsd', 'val_lddt', 'val_clash' }
        """
        self.model.eval()
        total_rmsd = 0.0
        total_lddt = 0.0
        total_clash = 0.0

        num_batches = len(dataloader) if hasattr(dataloader, "__len__") else 0
        for batch_idx, batch in enumerate(dataloader):
            # Переносим данные на устройство
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with _autocast(self.use_amp):
                # Вызываем модель с батчем в виде словаря
                pred_data = self.model(batch)
                pred_coords = pred_data['coords']

            # compute metrics
            coords_good = batch["coords_good"]
            atom_types = batch["atom_types"]
            mask = batch.get("mask", None)

            if self.metrics_calculator is not None:
                rmsd = self.metrics_calculator.compute_rmsd(pred_coords, coords_good, mask=mask)
                lddt = self.metrics_calculator.compute_lddt(pred_coords, coords_good, mask=mask)
                clash = self.metrics_calculator.compute_clash_score(pred_coords, atom_types, mask=mask)
            else:
                diff = (pred_coords - coords_good)
                rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean())
                lddt = torch.tensor(0.0, device=self.device)
                clash = torch.tensor(0.0, device=self.device)

            total_rmsd += float(rmsd.detach().cpu().item())
            total_lddt += float(lddt.detach().cpu().item())
            total_clash += float(clash.detach().cpu().item())

            if (batch_idx % self.log_every) == 0:
                logger.info(f"Val Batch {batch_idx}/{num_batches if num_batches else '?'}")

        if num_batches:
            return {
                "val_rmsd": total_rmsd / num_batches,
                "val_lddt": total_lddt / num_batches,
                "val_clash": total_clash / num_batches,
            }
        else:
            return {
                "val_rmsd": total_rmsd,
                "val_lddt": total_lddt,
                "val_clash": total_clash,
            }


    def _model_accepts_mask(self) -> bool:
        return True

    def state_dict(self) -> dict:
        """Return minimal pipeline state (scaler state) for checkpointing."""
        state = {}
        if hasattr(self, "scaler") and self.scaler is not None:
            try:
                state["scaler_state_dict"] = self.scaler.state_dict()
            except Exception:
                state["scaler_state_dict"] = None
        return state

    def load_state_dict(self, state: dict):
        """Load pipeline state (scaler)."""
        if "scaler_state_dict" in state and state["scaler_state_dict"] is not None:
            try:
                self.scaler.load_state_dict(state["scaler_state_dict"])
            except Exception:
                logger.warning("Failed to load scaler state_dict into TrainingPipeline.")
