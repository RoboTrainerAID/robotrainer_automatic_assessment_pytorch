from torch.utils.data import DataLoader, Dataset

from config.config import TrainConfig


def create_dataloader(train_config: TrainConfig, dataset: Dataset) -> DataLoader:
    """
    Create training or validation dataloaders.

    Args:
        train_config (TrainConfig): Training configuration containing batch size.
        dataset (Dataset): The dataset to create dataloaders from.

    Returns:
        DataLoader: Training or validation dataloader.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
 