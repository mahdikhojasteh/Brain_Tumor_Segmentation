import io
from my_dataset.dataloader import BrainDataset
from matplotlib import pyplot as plt
from my_lightning_module.segmenter import SegModel, ModelEnum
import torch
import numpy as np

data_dir = '/media/idham/SpaceX/333_workspace/my_lab/stacked_img_msk_data_dirs.npy'
data_path = np.load(data_dir, allow_pickle=True)

dataset = BrainDataset(data_path[:,0], data_path[:,1])

f = io.open('plots/indices.txt', mode='r', encoding='utf-8')
idxs = f.read().splitlines()

path = 'lightning_logs/version_12/checkpoints/TPR_ED_UNET_2-no_fold_2-epoch=42-val_loss=0.75.ckpt'
# path = 'lightning_logs/version_9/checkpoints/TPR_ED_UNET_2-epoch=39-val_loss=0.06.ckpt'
# path = 'lightning_logs/version_11/checkpoints/TPR_ED_UNET_2-no_fold-epoch=49-val_loss=0.08.ckpt'
az3_model = SegModel.load_from_checkpoint(checkpoint_path=path)

for idx in idxs:
    idx = int(idx)
    img, msk = dataset[idx]
    img = img.unsqueeze(axis=0)
    msk = msk.unsqueeze(axis=0)
    az3_model.eval()
    with torch.no_grad():
        pred3 = az3_model(img)
    fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
    for i, ax in enumerate(axis):
        ax.axis('off')
        match i:
            case 0:
                ax.set_title('flair', fontsize=20)
                ax.imshow(img[0, 0, ...], cmap="gray")
            case 1:
                ax.set_title('mask', fontsize=20)
                ax.imshow(msk.argmax(dim=1)[0, ...], cmap="gray")
            case 2:
                ax.set_title('prediction', fontsize=20)
                ax.imshow(pred3.argmax(dim=1)[0, ...], cmap="gray")
    fig.savefig(f'plots/tmp/az3_{idx}.png')