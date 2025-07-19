import yaml
import argparse
import numpy as np

from torch.utils.data import DataLoader, Subset

from train.trainer import Trainer, SiameseTrainer
from data.dataset import BudjBimAreaDataset

from seg_utils import build_model
from model.segmentation import segmentation_model, siamese_segmentation_model
from model.loss import DiceBCELogitsSmoothingLoss


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N', default=None, help='Path to config file')
    parser.add_argument('--path', type=str,  metavar='N', default=None, help='Optinal path to save model')
    parser.add_argument('-p', '--train_set_proportion', type=float,  metavar='N', default=1.0, help='Proportion of dataset used for training')
    parser.add_argument('-s', '--siamese', action=argparse.BooleanOptionalAction, default=False, help='Siamese networks for segmentation fine-tuning')
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    
    with open(args.cfg, 'r') as f: 
        cfg = yaml.safe_load(f)   
        
    if args.path: 
        cfg['path'] = args.path
        
    trainer = SiameseTrainer(cfg=cfg) if args.siamese else Trainer(cfg=cfg) 
    trainer.set_seed()
    
    # ============ dataset loader ============
    areas = ['area1', 'area2', 'area3', 'area4', 'area5', 'area6']
    
    for area in areas:        
        train_dataset = BudjBimAreaDataset(
            cfg['root'], 
            split='train', 
            test_area=area, 
            img_size=cfg['img_size'], 
            img_norm=cfg['img_norm']
        )
        train_subset = Subset(
            train_dataset, 
            np.arange(int(len(train_dataset) * args.train_set_proportion))
        )
        
        val_dataset = BudjBimAreaDataset(
            cfg['root'], 
            split='val', 
            test_area=area, 
            img_size=cfg['img_size'], 
            img_norm=cfg['img_norm']
        )
        test_dataset = BudjBimAreaDataset(
            cfg['root'], 
            split='test', 
            test_area=area, 
            img_size=cfg['img_size'], 
            img_norm=cfg['img_norm']
        )

        train_loader = DataLoader(
            train_subset, 
            batch_size=cfg['batch'], 
            shuffle=True, 
            num_workers=cfg['num_workers'],
            drop_last=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['batch'],
            shuffle=False,
            num_workers=cfg['num_workers'],
            drop_last=False,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg['batch'],
            shuffle=False,
            num_workers=cfg['num_workers'],
            drop_last=False,
            pin_memory=True
        )

        if args.siamese:
            mhs_backbone = build_model(
                cfg['arch'], 
                cfg['weight_mode'], 
                cfg['mhs_encoder'], 
                cfg['img_size'], 
                cfg['patch_size']
            )
            vat_backbone = build_model(
                cfg['arch'], 
                cfg['weight_mode'], 
                cfg['vat_encoder'], 
                cfg['img_size'], 
                cfg['patch_size']
            )
            seg_model = siamese_segmentation_model(
                [mhs_backbone, vat_backbone], 
                cfg['feature_indices'], 
                cfg['feature_channels'], 
                cfg['patch_size'], 
                cfg['arch']
            )
        else:
            backbone = build_model(
                cfg['arch'], 
                cfg['weight_mode'], 
                cfg['encoder'], 
                cfg['img_size'], 
                cfg['patch_size']
            )
            seg_model = segmentation_model(
                backbone, 
                cfg['feature_indices'], 
                cfg['feature_channels'], 
                cfg['patch_size'], 
                cfg['arch']
            )

        trainer.path = cfg['path'] + area
        trainer.fit(
            seg_model, 
            criterion=DiceBCELogitsSmoothingLoss(),
            train_loader=train_loader, 
            val_loader=val_loader
        )
        trainer.eval(
            seg_model, 
            criterion=DiceBCELogitsSmoothingLoss(),
            dataloader=test_loader, 
            ckpt='best_val_epoch.pth',
            verbose=True
        )