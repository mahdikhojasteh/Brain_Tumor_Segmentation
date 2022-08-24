import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from my_metrics.dice_multiclass import DiceLossMulticlass
from my_metrics.IoU_multiclass import IoULossMulticlass
from my_metrics.Focal_multiclass import FocalLossMulticlass
from my_models.unet import UNet
from my_models.unet_tpr_e import TPR_E_Unet
from my_models.unet_tpr_d import TPR_D_Unet
from my_models.unet_tpr_ed import TPR_ED_Unet
from my_models.unet_tpr_ed_2 import TPR_ED_Unet2

from enum import Enum


class ModelEnum(Enum):
    UNET=1
    TPR_E_UNET=2
    TPR_D_UNET=3
    TPR_ED_UNET=4
    TPR_ED_UNET_2=5
    

class SegModel(pl.LightningModule):

    def __init__(
        self,
        model_enum: ModelEnum,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
                
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        self.dice_multiclass = DiceLossMulticlass(calculate_weight=True)
        self.IoU_multiclass = IoULossMulticlass()
        self.Focal_multiclass = FocalLossMulticlass()

        match self.hparams.model_enum:
            case ModelEnum.UNET:
                self.net = UNet()
            case ModelEnum.TPR_E_UNET:
                self.net = TPR_E_Unet()
            case ModelEnum.TPR_D_UNET:
                self.net = TPR_D_Unet()
            case ModelEnum.TPR_ED_UNET:
                self.net = TPR_ED_Unet()
            case ModelEnum.TPR_ED_UNET_2:
                self.net = TPR_ED_Unet2()
   
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = self.dice_multiclass(out, mask)
        log_dict = {"train_loss": loss}
        # self.log('train_loss', loss)
        return {"loss": loss, "log": log_dict, "progress_bar": log_dict}
    
    # def training_epoch_end(self, outputs):
    #     loss_train = torch.stack([x["loss"] for x in outputs]).mean()
    #     self.log('train_loss', loss_train)

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = self.dice_multiclass(out, mask)
        loss_IoU_val = self.IoU_multiclass(out, mask)
        loss_Focal_val = self.Focal_multiclass(out, mask)
        # self.log('val_loss', loss_val)
        return {"val_loss": loss_val, "loss_IoU_val": loss_IoU_val, "loss_Focal_val":loss_Focal_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        loss_IoU_val = torch.stack([x["loss_IoU_val"] for x in outputs]).mean()
        loss_Focal_val = torch.stack([x["loss_Focal_val"] for x in outputs]).mean()
        self.log('val_loss', loss_val)
        self.log('loss_IoU_val', loss_IoU_val)
        self.log('loss_Focal_val', loss_Focal_val)
        log_dict = {"val_loss": loss_val, "loss_IoU_val":loss_IoU_val, "loss_Focal_val":loss_Focal_val}
        return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer
    
    


class MetricTracker(Callback):

  def __init__(self):
    self.collection = []

  def on_validation_epoch_end(self, trainer, module):
    elogs = trainer.logged_metrics # access it here
    self.collection.append(elogs)
    # do whatever is needed
