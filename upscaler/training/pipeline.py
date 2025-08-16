import logging
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from upscaler.model.upscaler import ProteinUpscaler
from upscaler.loss.loss import ProteinUpscalingLoss
from upscaler.utils.metrics import QualityMetrics

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Пайплайн для обучения модели апскейлинга белковых структур."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        clip_grad_norm: float | None = 1.0,
        metrics_calculator: QualityMetrics | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ):
        """
        Args:
            model (nn.Module): Модель для обучения.
            loss_fn (nn.Module): Функция потерь.
            optimizer (torch.optim.Optimizer): Оптимизатор.
            device (torch.device): Устройство для вычислений.
            clip_grad_norm (float, optional): Норма для обрезки градиентов.
            metrics_calculator (QualityMetrics, optional): Калькулятор метрик.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Планировщик LR.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.metrics_calculator = metrics_calculator or QualityMetrics(device=device)
        self.scheduler = scheduler
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Обучает модель одну эпоху.
        
        Args:
            dataloader (DataLoader): Загрузчик обучающих данных.
            
        Returns:
            dict[str, float]: Словарь со средними значениями потерь и метрик.
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = {
            'coord_rmsd': 0.0,
            'lddt': 0.0,
            'clash': 0.0,
            'physics': 0.0,
            'total': 0.0
        }
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            coords_bad = batch['coords_bad'].to(self.device)
            coords_good = batch['coords_good'].to(self.device)
            atom_types = batch['atom_types'].to(self.device)
            residue_types = batch['residue_types'].to(self.device)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_coords = self.model(coords_bad, atom_types, residue_types)
            
            # Вычисляем функцию потерь
            loss, metrics = self.loss_fn(pred_coords, coords_good, atom_types)
            
            # Backward pass
            loss.backward()
            
            # Обрезка градиентов
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            # Шаг оптимизатора
            self.optimizer.step()
            
            # Аккумулируем потери и метрики
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key].item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Train Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Усредняем по всем батчам
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Валидирует модель.
        
        Args:
            dataloader (DataLoader): Загрузчик валидационных данных.
            
        Returns:
            dict[str, float]: Словарь со средними значениями метрик.
        """
        self.model.eval()
        total_rmsd = 0.0
        total_lddt = 0.0
        total_clash = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            coords_bad = batch['coords_bad'].to(self.device)
            coords_good = batch['coords_good'].to(self.device)
            atom_types = batch['atom_types'].to(self.device)
            residue_types = batch['residue_types'].to(self.device)
            
            # Forward pass
            pred_coords = self.model(coords_bad, atom_types, residue_types)
            
            # Вычисляем метрики
            rmsd = self.metrics_calculator.compute_rmsd(pred_coords, coords_good)
            lddt = self.metrics_calculator.compute_lddt(pred_coords, coords_good)
            clash = self.metrics_calculator.compute_clash_score(pred_coords, atom_types)
            
            total_rmsd += rmsd.item()
            total_lddt += lddt.item()
            total_clash += clash.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Val Batch {batch_idx}/{num_batches}")
        
        # Усредняем по всем батчам
        avg_rmsd = total_rmsd / num_batches
        avg_lddt = total_lddt / num_batches
        avg_clash = total_clash / num_batches
        
        return {
            'val_rmsd': avg_rmsd,
            'val_lddt': avg_lddt,
            'val_clash': avg_clash
        }
