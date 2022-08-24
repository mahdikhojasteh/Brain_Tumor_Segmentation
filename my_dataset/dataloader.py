from typing import Any
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from typing import Optional
from torch.utils.data import DataLoader
from .dataset import BrainDataset
from my_utilities.data_dir_helper import get_processed_data_path


class BrainDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = None,
            k: int = 1,  # fold number
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,
            batch_size: int = 10,
            num_workers: int = 0,
            pin_memory: bool = False,
            transform: Any = None
        ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 1 <= self.hparams.k <= self.hparams.num_splits, "incorrect fold number"

        self.data_path = get_processed_data_path(stacked_data_path=data_dir)
        
        self.data_train: Optional[BrainDataset] = None
        self.data_val: Optional[BrainDataset] = None

    @property
    def num_node_features() -> int:
        return 4

    @property
    def num_classes() -> int:
        return 4

    def setup(self, stage=None, no_fold=False):
        if no_fold:
            self.data_train = BrainDataset(self.data_path[:, 0].tolist(), self.data_path[:, 1].tolist(), transform=self.hparams.transform) 
            self.data_val = BrainDataset(self.data_path[:, 0].tolist(), self.data_path[:, 1].tolist()) 
            
        elif not self.data_train and not self.data_val:
            dataset_full = BrainDataset(self.data_path[:, 0].tolist(), self.data_path[:, 1].tolist())    
            # choose fold to train on
            kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(dataset_full)]
            train_indexes, val_indexes = all_splits[self.hparams.k-1]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            train_img_paths = self.data_path[train_indexes, 0].tolist()
            train_msk_paths = self.data_path[train_indexes, 1].tolist()
            
            val_img_paths = self.data_path[val_indexes, 0].tolist()
            val_msk_paths = self.data_path[val_indexes, 1].tolist()
            
            self.data_train = BrainDataset(train_img_paths, train_msk_paths, transform=self.hparams.transform)
            self.data_val = BrainDataset(val_img_paths, val_msk_paths)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)
