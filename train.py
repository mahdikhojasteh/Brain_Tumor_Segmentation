import os
import torch
import pytorch_lightning as pl
from my_dataset.dataloader import BrainDataModule
from my_lightning_module.segmenter import SegModel, ModelEnum, MetricTracker
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from my_utilities.data_dir_helper import generate_processed_data_dirs
import imgaug.augmenters as iaa



if __name__ == '__main__':
    # generate_processed_data_dirs(
    #     preprocessed_path = '/media/idham/SpaceX/333_workspace/my_lab/my_processed_data',
    #     check_sync = True,
    #     output_name = 'stacked_img_msk_data_dirs_2'
    # )
    
    nums_folds = 5
    split_seed = 12345
    results = []
        
    model_enum = ModelEnum.TPR_ED_UNET_2
    
    for k in range(1, nums_folds):
        if k > 1:
            break
        
        segmenter = SegModel(model_enum=model_enum)

        # saves top-K checkpoints based on "val_loss" metric
        checkpoint_callback = ModelCheckpoint(
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            filename=f"{model_enum.name}-no_fold_2"+"-{epoch:02d}-{val_loss:.2f}",
        )
        
        metric_tracker_callback = MetricTracker()
        
        trainer = pl.Trainer(accelerator='gpu', callbacks=[checkpoint_callback, metric_tracker_callback], max_epochs=70) # , fast_dev_run=True
        
        
        data_dir = '/media/idham/SpaceX/333_workspace/my_lab/stacked_img_msk_data_dirs_2.npy'
        train_transform = iaa.SomeOf((1, 4), [
            iaa.Dropout([0.05, 0.1]),      # drop 5% or 10% of all pixels
            iaa.Sharpen((0.0, 1.0)),       # sharpen the image
            iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
        ], random_order=True)

        datamodule = BrainDataModule(
            data_dir=data_dir,
            k=k, num_splits=nums_folds, 
            split_seed=split_seed, 
            batch_size=18, 
            transform=train_transform, 
            num_workers=os.cpu_count()
        )
        datamodule.prepare_data()
        datamodule.setup(no_fold=True)

        trainer.fit(model=segmenter, datamodule=datamodule)
        
        val_loss = trainer.logged_metrics['val_loss'].item()
        score = round((1-val_loss)*100,2)        
        results.append(score)
        
        np.save(f"metric_collection_{nums_folds}_fold_k_{k}_{model_enum.name}_no_fold_2_double", metric_tracker_callback.collection)

        score = round(sum(results) / 1, 2)
        np.save(f"score_{nums_folds}_fold_{model_enum.name}_2_double", score)

        # score = round(sum(results) / nums_folds, 2)
        # np.save(f"score_{nums_folds}_fold_{model_enum.name}", score)