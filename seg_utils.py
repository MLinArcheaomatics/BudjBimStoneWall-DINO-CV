import torch
import model.backbone.vision_transformer as vits

from pathlib import Path
from torchvision import models as torchvision_models
from dino_utils import load_pretrained_weights


# Code adapoted from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation monitor doesn't improve after a given patience."""
    def __init__(self, path, best_score=None, patience=10, delta=0.0, verbose=False, trace_func=print):
        """
        Args:
            path (str): Path for the checkpoint to be saved to.
            best_score (flaot or none): Value of metric of the best model.
                            Default: None
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            trace_func (function): trace print function.
                            Default: print            
        """
        self.path = Path(path)
        self.best_score = best_score
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.trace_func = trace_func
        
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss, model, optimizer, epoch, last_lr, cm=None):
        score = loss 
        
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'lr': last_lr[0],
            'confusion_matrix': cm
        }
    
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(checkpoint, score)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'\n Validation loss does not imporove ({self.best_score} --> {score}). \n EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(checkpoint, score)
            self.counter = 0

    def save_checkpoint(self, checkpoint, loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'\n Validation loss decrease ({self.best_score:.6f} --> {loss:.6f}). \n Saving model ...')
            
        checkpoint_path = self.path / 'ckpt'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path.joinpath('best_val_epoch.pth'))
        
        self.best_score = loss


def build_model(
    arch: str,
    weight_mode: str,
    pretrained_weights: str = None,
    img_size: int = None,
    patch_size: int = None,
    ckpt_key: str = "teacher",
):
    """
    Builds a model based on the specified architecture and weight mode.

    Args:
        arch (str): Model architecture name (e.g., 'resnet50', 'vit_base').
        weight_mode (str): Weight loading mode ('load', 'supervised', 'random').
        pretrained_weights (str, optional): Path to pretrained weights file 
            (used if weight_mode is 'load').
        img_size (int, optional): Input image size for Vision Transformer models.
        patch_size (int, optional): Patch size for Vision Transformer models.
        ckpt_key (str, optional): Key to extract weights from the checkpoint.

    Returns:
        torch.nn.Module: The constructed model.

    Raises:
        ValueError: If `arch` is not recognized or `weight_mode` is invalid.
    """
    if arch not in torchvision_models.__dict__ and arch not in vits.__dict__:
        raise ValueError(f"Unknown architecture: {arch}")

    if arch in vits.__dict__ and patch_size is None:
        raise ValueError(
            f"Patch size must be specified for Vision Transformer models like '{arch}'."
        )

    # Initialize model based on weight mode
    if weight_mode == "load":
        if not pretrained_weights:
            raise ValueError(
                "Pretrained weights path must be provided when weight_mode is 'load'."
            )

        model = (
            vits.__dict__[arch](
                patch_size=patch_size,
                drop_rate=0.1,
                attn_drop_rate=0.1,
                drop_path_rate=0.1,
            )
            if arch in vits.__dict__
            else torchvision_models.__dict__[arch](num_classes=0)
        )

        load_pretrained_weights(model, pretrained_weights, ckpt_key, arch, patch_size)
        print(f"Initialized model '{arch}' with self-supervised pre-trained weights.")

    elif weight_mode == "supervised":
        if arch in vits.__dict__ and arch not in torchvision_models.__dict__:
            if arch == "vit_small" and patch_size == 16:
                from timm.models.vision_transformer import vit_small_patch16_384

                model = vit_small_patch16_384(
                    pretrained=True,
                    img_size=img_size,
                    drop_rate=0.1,
                    attn_drop_rate=0.1,
                    drop_path_rate=0.1,
                )
            else:
                raise ValueError(
                    f"Supervised weights are not available for Vision Transformer models like '{arch}'."
                )
        else:
            model = torchvision_models.__dict__[arch](weights="IMAGENET1K_V1")

        print(f"Initialized model '{arch}' with supervised weights (ImageNet1k).")

    elif weight_mode == "random":
        model = (
            vits.__dict__[arch](patch_size=patch_size, img_size=[img_size])
            if arch in vits.__dict__
            else torchvision_models.__dict__[arch](num_classes=0)
        )

        print(f"Initialized model '{arch}' with random weights.")

    else:
        raise ValueError(
            f"Invalid weight_mode: {weight_mode}. Expected 'load', 'supervised', or 'random'."
        )

    return model