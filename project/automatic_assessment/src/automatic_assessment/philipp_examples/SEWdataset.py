class SEWDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, config: DatasetConfig, transform=None):
        self.pairs = pairs
        self.config = config
        self.transform = transform

    def __len__(self): ...

    def __getitem__(self, idx: int): ...

    @classmethod
    def create(cls, config: DatasetConfig) -> tuple[SEWDataset, SEWDataset]:
        data_root = config.data_root

        modality_paths = config.modalities

        modality_pairs = DatasetSplitter.collect_image_pairs_from_subfolders(
            Path(data_root), modality_paths
        )

        train_pairs, val_pairs = DatasetSplitter.split_dataset(
            all_pairs=modality_pairs,
            val_split=config.val_split,
            split_seed=config.split_seed,
            dataset_fraction=config.dataset_fraction,
        )

        train_dataset = cls(
            pairs=train_pairs, config=config, transform=get_transforms("train")
        )

        val_dataset = cls(
            pairs=val_pairs, config=config, transform=get_transforms("val")
        )

        logger.info(
            f"Created SEW Dataset with {len(train_dataset)} training samples and {len(val_dataset)} validation samples."
        )

        return train_dataset, val_dataset
 