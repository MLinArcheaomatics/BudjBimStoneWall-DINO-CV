import yaml
import argparse
import numpy as np
import rasterio as rio

from tqdm import tqdm
from pathlib import Path
from typing import Any
from osgeo_utils.gdal_sieve import gdal_sieve

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from train.trainer import SiameseTrainer
from data.dataset import BudjBimLandscapeDataset
from data.data_utils import arr_to_tiff

from seg_utils import build_model
from model.segmentation import siamese_segmentation_model


def predict_mask(model: Any,
                 dataset_dir: str,
                 out_dir: str,
                 conf_level: float = 0.5,
                 sieve_thres: int = 100) -> None:
    
    pred_fp = (Path(out_dir) / 'pred') if out_dir else (Path('/temp') / 'pred')
    pred_fp.mkdir(parents=True, exist_ok=True)
    
    for idx, batch in enumerate(tqdm(dataloader)):
        mhs, vat = batch['mhs'].to(cfg['device']), batch['vat'].to(cfg['device'])
        pred = model([mhs, vat]).sigmoid().squeeze(0)
        pred = (pred > conf_level).float()
        
        if pred.max() == 1.0:
            file_name = batch['file_name'][0]
            pred_out = pred_fp / file_name
            
            with rio.open(Path(dataset_dir) / 'mhs' / file_name) as tiff:            
                pred = TF.resize(pred, size=[tiff.height, tiff.width], interpolation=T.InterpolationMode.NEAREST)
                pred = pred.detach().cpu().numpy().astype(np.uint8)
                # (C, H, W) -> (H, W, C)
                pred = pred.reshape(pred.shape[1], pred.shape[2], pred.shape[0])
                
                arr_to_tiff(pred, output=pred_out, crs=tiff.crs, transform=tiff.transform)
                
            gdal_sieve(
                src_filename=str(pred_out),
                dst_filename=str(pred_out),
                threshold=sieve_thres,
                connectedness=8
            )
            
            # Remove if raster is empty (all zeros) after sieving
            with rio.open(pred_out) as out_raster:
                data = out_raster.read()
                if np.all(data == 0):
                    pred_out.unlink()  # Delete the file
                    
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Prediction of BudjBim Landscape using DINO-CV')
    parser.add_argument('--cfg', type=str, default='cfg/dino_cv_rn50_budjbim_siamese.yaml', metavar='N',
                        help='config file to set up segmentation model')
    parser.add_argument('--area_ckpt', type=str, default='area3', metavar='N',
                        help='load the model checkpoint of area (i.e., area1-6)')
    parser.add_argument('--dataset_dir', type=str, default='data/budjbim_landscape', metavar='N',
                        help='path to BudjBimLandscape dataset')
    parser.add_argument('--img_size', type=int, default=256, metavar='N',
                        help='input image size (H * W)')
    parser.add_argument('--out_dir', type=str, default='data/budjbim_landscape', metavar='N',
                        help='output dir to mask preds')
    parser.add_argument('--conf_level', type=float, default=0.5, metavar='N',
                        help='confidence level of model binary prediction')
    parser.add_argument('--sieve_thres', type=int, default=100, metavar='N',
                        help='sieve threshold for gdal_sieve')
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f) 
        trainer = SiameseTrainer(cfg=cfg) 
    
    dataset = BudjBimLandscapeDataset(
        args.dataset_dir,
        transform=T.Compose([
            T.Resize(size=args.img_size, interpolation=T.InterpolationMode.BICUBIC), 
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        num_workers=cfg['num_workers'], 
        shuffle=False, 
        drop_last=False, 
        pin_memory=True
    )
    
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

    trainer.path = cfg['path'] + args.area_ckpt
    seg_model = trainer.load_weights(seg_model, ckpt='best_val_epoch.pth')

    predict_mask(
        model = seg_model,
        dataset_dir = args.dataset_dir,
        out_dir = args.out_dir, 
        conf_level = args.conf_level,
        sieve_thres = args.sieve_thres
    )