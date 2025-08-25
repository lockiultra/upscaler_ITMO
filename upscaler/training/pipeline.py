import logging
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

# from upscaler.model.upscaler import ProteinUpscaler
# from upscaler.loss.loss import ProteinUpscalingLoss
from upscaler.utils.metrics import QualityMetrics

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        clip_grad_norm: float | None = 1.0,
        metrics_calculator: QualityMetrics | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        use_amp: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.metrics_calculator = metrics_calculator or QualityMetrics(device=device)
        self.scheduler = scheduler
        self.use_amp = use_amp

        self.model.to(self.device)

        # GradScaler для AMP: включаем только если CUDA и use_amp=True
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp and (self.device.type == "cuda")))

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_metrics = { 'coord_rmsd': 0.0, 'lddt': 0.0, 'clash': 0.0, 'physics': 0.0, 'total': 0.0 }
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            coords_bad = batch['coords_bad'].to(self.device, non_blocking=True)
            coords_good = batch['coords_good'].to(self.device, non_blocking=True)
            atom_types = batch['atom_types'].to(self.device, non_blocking=True)
            residue_types = batch['residue_types'].to(self.device, non_blocking=True)

            # zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward + loss inside autocast
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                pred_coords = self.model(coords_bad, atom_types, residue_types)
                loss, metrics = self.loss_fn(pred_coords, coords_good, atom_types)

            # Backward: используем scaler, если включён AMP
            if self.scaler.is_enabled():
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

            # аккумулируем метрики (detach -> cpu -> item)
            total_loss += float(loss.detach().cpu().item())
            for key in total_metrics:
                total_metrics[key] += float(metrics[key].detach().cpu().item())

            if batch_idx % 10 == 0:
                logger.info(f"Train Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics['loss'] = avg_loss
        return avg_metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_rmsd = total_lddt = total_clash = 0.0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            coords_bad = batch['coords_bad'].to(self.device, non_blocking=True)
            coords_good = batch['coords_good'].to(self.device, non_blocking=True)
            atom_types = batch['atom_types'].to(self.device, non_blocking=True)
            residue_types = batch['residue_types'].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                pred_coords = self.model(coords_bad, atom_types, residue_types)

            rmsd = self.metrics_calculator.compute_rmsd(pred_coords, coords_good)
            lddt = self.metrics_calculator.compute_lddt(pred_coords, coords_good)
            clash = self.metrics_calculator.compute_clash_score(pred_coords, atom_types)

            total_rmsd += float(rmsd.detach().cpu().item())
            total_lddt += float(lddt.detach().cpu().item())
            total_clash += float(clash.detach().cpu().item())

            if batch_idx % 10 == 0:
                logger.info(f"Val Batch {batch_idx}/{num_batches}")

        return {
            'val_rmsd': total_rmsd / num_batches,
            'val_lddt': total_lddt / num_batches,
            'val_clash': total_clash / num_batches
        }
