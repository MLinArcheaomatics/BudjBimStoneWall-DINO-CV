import yaml
import argparse
import copy
import torch

from torch.utils.data import DataLoader

from train.trainer import Trainer, SiameseTrainer
from data.dataset import BudjBimAreaDataset

from seg_utils import build_model
from model.segmentation import segmentation_model, siamese_segmentation_model


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N', default=None, help='Path to config file')
    parser.add_argument('--path', type=str,  metavar='N', default=None, help='Optinal path to save model')
    parser.add_argument('-s', '--siamese', action=argparse.BooleanOptionalAction, default=False, help='Siamese networks for segmentation fine-tuning')
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    
    with open(args.cfg, 'r') as f: 
        cfg = yaml.safe_load(f)   
        
    if args.path: 
        cfg['path'] = args.path
        
    trainer = SiameseTrainer(cfg=cfg) if args.siamese else Trainer(cfg=cfg) 
    trainer.set_seed()
    
    metrics = {
        'Precision': None,
        'Recall': None,
        'F1Score': None,
        'mIoU': None,
    }
    
    # ============ dataset loader ============
    areas = ['area1', 'area2', 'area3', 'area4', 'area5', 'area6']
    
    for idx, area in enumerate(areas):        
        test_dataset = BudjBimAreaDataset(
            cfg['root'], 
            split='test', 
            test_area=area, 
            img_size=cfg['img_size'], 
            img_norm=cfg['img_norm']
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
        mcs = trainer.eval(
            seg_model, 
            dataloader=test_loader, 
            compute_stats=False,
            ckpt='best_val_epoch.pth'
        )
        
        # Copy and merge metrics properly
        for name, mc in mcs.items():
            device = next(mc.parameters(), torch.tensor([])).device  # Handle CPU fallback
            if idx == 0:
                metrics[name] = copy.deepcopy(mc).to(device)
            else:
                metrics[name].merge_state(copy.deepcopy(mc).to(device))
            
    out = []
    for k, v in metrics.items():
        stats = v.compute() * 100
        out.append(f'{k}: {stats:.1f}%')
    print(' | '.join(out))