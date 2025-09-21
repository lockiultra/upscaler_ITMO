from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        clip_grad_norm: float | None = 1.0,
        metrics_calculator=None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
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
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Train one epoch over dataloader.
        Expects dataloader to yield samples in collated dict-form produced by collate_batch:
        { 'coords_bad', 'coords_good', 'atom_types', 'residue_types', 'lengths', 'mask' }
        mask/lengths are optional — pipeline will handle missing keys.
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = {
            "coord_rmsd": 0.0,
            "lddt": 0.0,
            "clash": 0.0,
            "physics": 0.0,
            "total": 0.0,
        }

        num_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
        for batch_idx, batch in enumerate(dataloader):
            coords_bad = batch["coords_bad"].to(self.device, non_blocking=True)
            coords_good = batch["coords_good"].to(self.device, non_blocking=True)
            atom_types = batch["atom_types"].to(self.device, non_blocking=True)
            residue_types = batch["residue_types"].to(self.device, non_blocking=True)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred_coords = self.model(coords_bad, atom_types, residue_types, mask=mask) if self._model_accepts_mask() else self.model(coords_bad, atom_types, residue_types)
                try:
                    loss, metrics = self.loss_fn(pred_coords, coords_good, atom_types, mask=mask)
                except TypeError:
                    loss, metrics = self.loss_fn(pred_coords, coords_good, atom_types)

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

            total_loss += float(loss.detach().cpu().item())
            for k in total_metrics.keys():
                if k in metrics:
                    total_metrics[k] += float(metrics[k].detach().cpu().item())
                else:
                    total_metrics[k] += 0.0

            if (batch_idx % self.log_every) == 0:
                if num_batches:
                    logger.info(f"Train Batch {batch_idx}/{num_batches} — loss {loss.item():.4f}")
                else:
                    logger.info(f"Train Batch {batch_idx} — loss {loss.item():.4f}")

        if num_batches:
            avg_loss = total_loss / num_batches
            avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        else:
            avg_loss = total_loss
            avg_metrics = total_metrics
        avg_metrics["loss"] = avg_loss
        return avg_metrics

    def train_one_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Train on a single batch and return loss and metrics.
        """
        coords_bad = batch["coords_bad"].to(self.device, non_blocking=True)
        coords_good = batch["coords_good"].to(self.device, non_blocking=True)
        atom_types = batch["atom_types"].to(self.device, non_blocking=True)
        residue_types = batch["residue_types"].to(self.device, non_blocking=True)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred_coords = self.model(coords_bad, atom_types, residue_types, mask=mask) if self._model_accepts_mask() else self.model(coords_bad, atom_types, residue_types)
            try:
                loss, metrics = self.loss_fn(pred_coords, coords_good, atom_types, mask=mask)
            except TypeError:
                loss, metrics = self.loss_fn(pred_coords, coords_good, atom_types)

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

        metrics['total'] = loss.detach()  # Add total loss to metrics for consistency
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
            coords_bad = batch["coords_bad"].to(self.device, non_blocking=True)
            coords_good = batch["coords_good"].to(self.device, non_blocking=True)
            atom_types = batch["atom_types"].to(self.device, non_blocking=True)

            residue_types = batch.get("residue_types", None)
            if residue_types is not None:
                residue_types = residue_types.to(self.device, non_blocking=True)

            # mask optional
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if self._model_accepts_mask():
                    pred_coords = self.model(coords_bad, atom_types, residue_types, mask=mask)
                else:
                    pred_coords = self.model(coords_bad, atom_types, residue_types)

            # compute metrics
            if self.metrics_calculator is not None:
                try:
                    rmsd = self.metrics_calculator.compute_rmsd(pred_coords, coords_good, mask=mask)
                    lddt = self.metrics_calculator.compute_lddt(pred_coords, coords_good, mask=mask)
                    clash = self.metrics_calculator.compute_clash_score(pred_coords, atom_types, mask=mask)
                except TypeError:
                    rmsd = self.metrics_calculator.compute_rmsd(pred_coords, coords_good)
                    lddt = self.metrics_calculator.compute_lddt(pred_coords, coords_good)
                    clash = self.metrics_calculator.compute_clash_score(pred_coords, atom_types)
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