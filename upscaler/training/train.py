import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from upscaler.config import CSV_FILE, DATA_FOLDER, DEVICE
from upscaler.data.dataset import ProteinUpscalingDataset, collate_batch
from upscaler.model.upscaler import ProteinUpscaler
from upscaler.loss.loss import ProteinUpscalingLoss
from upscaler.utils.metrics import QualityMetrics
from upscaler.training.pipeline import TrainingPipeline


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_optimizer(model: nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
    """Создает оптимизатор."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

def get_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """Создает планировщик learning rate."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, checkpoint_dir: str):
    """Сохраняет чекпоинт модели."""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str):
    """Загружает чекпоинт модели."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}, loss {loss:.4f}")
    return epoch, loss

def train_model(
    csv_file: str,
    data_folder: str,
    checkpoint_dir: str = "checkpoints",
    num_epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: torch.device = DEVICE,
    resume_from_checkpoint: str = None,
):
    """
    Основной скрипт для обучения модели.
    
    Args:
        csv_file (str): Путь к CSV файлу с данными.
        data_folder (str): Путь к папке с PDB файлами.
        checkpoint_dir (str): Путь к папке для сохранения чекпоинтов.
        num_epochs (int): Количество эпох обучения.
        batch_size (int): Размер батча.
        lr (float): Начальная скорость обучения.
        device (torch.device): Устройство для вычислений.
        resume_from_checkpoint (str): Путь к чекпоинту для возобновления обучения.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("Loading dataset...")
    dataset = ProteinUpscalingDataset(csv_file, data_folder)
    
    # Разделение на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    # Инициализируем модель
    logger.info("Initializing model...")
    model = ProteinUpscaler()
    
    # Инициализируем функцию потерь
    loss_fn = ProteinUpscalingLoss(device=device)
    
    # Инициализируем оптимизатор
    optimizer = get_optimizer(model, lr=lr)
    
    # Инициализируем планировщик LR
    scheduler = get_scheduler(optimizer)
    
    # Инициализируем калькулятор метрик
    metrics_calculator = QualityMetrics(device=device)
    
    # Инициализируем пайплайн обучения
    pipeline = TrainingPipeline(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=1.0,
        metrics_calculator=metrics_calculator,
        scheduler=scheduler,
    )
    
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from_checkpoint)
        start_epoch += 1
    
    logger.info("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Обучение
        train_metrics = pipeline.train_epoch(train_loader)
        logger.info(f"Train Metrics: {train_metrics}")
        
        # Валидация
        val_metrics = pipeline.validate(val_loader)
        logger.info(f"Val Metrics: {val_metrics}")
        
        # Шаг планировщика
        if scheduler is not None:
            scheduler.step(train_metrics['loss'])
        
        # Сохраняем лучший чекпоинт
        if train_metrics['loss'] < best_val_loss:
            best_val_loss = train_metrics['loss']
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], checkpoint_dir)
            logger.info(f"New best model saved with loss {best_val_loss:.4f}")
        
        # Сохраняем чекпоинт каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], checkpoint_dir)
    
    logger.info("Training completed.")

if __name__ == "__main__":
    csv_file = CSV_FILE
    data_folder = DATA_FOLDER
    checkpoint_dir = "checkpoints"
    
    train_model(
        csv_file=csv_file,
        data_folder=data_folder,
        checkpoint_dir=checkpoint_dir,
        num_epochs=50,
        batch_size=4,
        lr=1e-4,
        device=DEVICE,
        resume_from_checkpoint=None,
    )
