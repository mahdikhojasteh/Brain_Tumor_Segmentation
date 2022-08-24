import io
from my_dataset.dataloader import BrainDataset
from matplotlib import pyplot as plt
from my_lightning_module.segmenter import SegModel, ModelEnum
import torch
import numpy as np

data_dir = '/media/idham/SpaceX/333_workspace/my_lab/stacked_img_msk_data_dirs.npy'
data_path = np.load(data_dir, allow_pickle=True)

dataset = BrainDataset(data_path[:,0], data_path[:,1])

f = io.open('plots/diffs/index.txt', mode='r', encoding='utf-8')
idxs = f.read().splitlines()

path = 'lightning_logs/version_5/checkpoints/TPR_E_UNET-epoch=29-val_loss=0.09.ckpt'
az1_model = SegModel.load_from_checkpoint(checkpoint_path=path)

path = 'lightning_logs/version_6/checkpoints/TPR_D_UNET-epoch=29-val_loss=0.09.ckpt'
az2_model = SegModel.load_from_checkpoint(checkpoint_path=path)

path = 'lightning_logs/version_9/checkpoints/TPR_ED_UNET_2-epoch=39-val_loss=0.06.ckpt'
az3_model = SegModel.load_from_checkpoint(checkpoint_path=path)

path = 'lightning_logs/version_4/checkpoints/UNET-epoch=27-val_loss=0.10.ckpt'
az_unet_model = SegModel.load_from_checkpoint(checkpoint_path=path)

for idx in idxs:
    idx = int(idx)
    img, msk = dataset[idx]
    img = img.unsqueeze(axis=0)
    msk = msk.unsqueeze(axis=0)
    az1_model.eval()
    az2_model.eval()
    az3_model.eval()
    az_unet_model.eval()
    with torch.no_grad():
        pred1 = az1_model(img)
        pred2 = az2_model(img)
        pred3 = az3_model(img)
        pred_unet = az_unet_model(img)
    fig, axis = plt.subplots(nrows=2, ncols=5, figsize=(20,7))
    for j, axs in enumerate(axis):
        if j == 0:
            for i, ax in enumerate(axs):
                ax.axis('off')
                match i:
                    case 0:
                        ax.set_title('flair', fontsize=20)
                        ax.imshow(img[0, 0, ...], cmap="gray")
                    case 1:
                        ax.set_title('unet', fontsize=20)
                        ax.imshow(pred_unet.argmax(dim=1)[0, ...], cmap="gray")
                    case 2:
                        ax.set_title('az1', fontsize=20)
                        ax.imshow(pred1.argmax(dim=1)[0, ...], cmap="gray")
                    case 3:
                        ax.set_title('az2', fontsize=20)
                        ax.imshow(pred2.argmax(dim=1)[0, ...], cmap="gray")
                    case 4:
                        ax.set_title('az3', fontsize=20)
                        ax.imshow(pred3.argmax(dim=1)[0, ...], cmap="gray")
            # fig.savefig(f'plots/az_unet_{index}.png')
        if j == 1:
            for i, ax in enumerate(axs):
                ax.axis('off')
                match i:
                    case 0:
                        ax.set_title('GT', fontsize=20)
                        ax.imshow(msk.argmax(dim=1)[0, ...], cmap="gray")
                    case 1:
                        ax.set_title('D_unet', fontsize=20)
                        _pred_unet = pred_unet.argmax(dim=1)[0, ...]
                        _msk = msk.argmax(dim=1)[0, ...]
                        
                        tmp = np.where(_pred_unet==_msk, 0, 1)
                        ax.imshow(tmp, cmap="gray")
                    case 2:
                        ax.set_title('D_az1', fontsize=20)
                        _pred1 = pred1.argmax(dim=1)[0, ...]
                        _msk = msk.argmax(dim=1)[0, ...]
                        
                        tmp1 = np.where(_pred1==_msk, 0, 1)
                        ax.imshow(tmp1, cmap="gray")
                    case 3:
                        ax.set_title('D_az2', fontsize=20)
                        _pred2 = pred2.argmax(dim=1)[0, ...]
                        _msk = msk.argmax(dim=1)[0, ...]
                        
                        tmp2 = np.where(_pred2==_msk, 0, 1)
                        ax.imshow(tmp2, cmap="gray")
                    case 4:
                        ax.set_title('D_az3', fontsize=20)
                        _pred3 = pred3.argmax(dim=1)[0, ...]
                        _msk = msk.argmax(dim=1)[0, ...]
                        
                        tmp3 = np.where(_pred3==_msk, 0, 1)
                        ax.imshow(tmp3, cmap='gray')
                        
    fig.savefig(f'plots/diffs/diff_{idx}.png')