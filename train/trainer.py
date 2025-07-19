import yaml
import logging
import torch
import random
import numpy as np

from pathlib import Path
from typing import Union, Optional, Callable, Dict
from tqdm import tqdm

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.classification import Precision, Recall, F1Score, JaccardIndex

from seg_utils import EarlyStopping


class Trainer:
    def __init__(self, cfg:Union[str, Path, Dict]):
        if isinstance(cfg, str or Path):
            with open(cfg, 'r') as f:
                self.cfg = yaml.safe_load(f)    
        elif isinstance(cfg, Dict):
            self.cfg = cfg
        else:
            raise ValueError("trainer cfg must be a string, Path, or dictionary.")
            
        self.epoch = self.cfg['epoch']
        self.path = self.cfg['path']
        self.patience = self.cfg['patience']
        self.lr = self.cfg['lr']
        self.w_decay = self.cfg['w_decay']
        self.optim = self.cfg['optim']
        
        self.metrics = {
            'Precision': Precision(task='binary', threshold=0.5).to(self.cfg['device']),
            'Recall': Recall(task='binary', threshold=0.5).to(self.cfg['device']),
            'F1Score': F1Score(task='binary', threshold=0.5).to(self.cfg['device']),
            'mIoU': JaccardIndex(task='binary', threshold=0.5).to(self.cfg['device'])
        }
        
        self.set_seed()

    def fit(self, model: torch.nn.Module, 
            criterion: Optional[Callable]=None,  
            train_loader: Optional[DataLoader]=None,
            val_loader: Optional[DataLoader]=None,
            ckpt: Union[str, Path, None]=None,
            save_period: int=1):
        summary(model)
        model = model.to(self.cfg['device'])
        
        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt)['model_state_dict'], strict=False)
        
        criterion = criterion or torch.nn.CrossEntropyLoss()

        if self.optim == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.w_decay
            )
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.w_decay
            )
        else:
            raise Exception("please specify an optimizer")
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epoch, 
            eta_min=1e-6
        )
        
        writer = SummaryWriter(log_dir=f'{self.path}/runs')
        early_stopping = EarlyStopping(path=self.path, patience=self.patience)
         
        for ep in tqdm(range(1, self.epoch + 1)):
            t_loss, t_mcs = self._fit_impl(model, optimizer, criterion, train_loader)
            writer.add_scalar('Loss/train', t_loss, ep)
            print('Loss/train', f"{t_loss:.2f}", end=' -- ')
            
            for mc_name, mc_value in t_mcs.items():
                writer.add_scalar(f'{mc_name}/train', mc_value, ep)
                print(mc_name+'/train', f"{mc_value * 100:.2f}", end=' -- ')

            # Adjust learning rate 
            lr_scheduler.step()
            writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
    
            if val_loader:
                v_loss, v_mcs = self._val_impl(model, criterion, val_loader) 
                writer.add_scalar('Loss/val', v_loss, ep)
                print('Loss/val', f"{v_loss:.2f}", end=' -- ')
                
                for mc_name, mc_value in v_mcs.items():
                    writer.add_scalar(f'{mc_name}/val', mc_value, ep)
                    print(mc_name+'/val', f"{mc_value * 100:.2f}", end=' -- ')
                    
                if ep % save_period == 0: # save model at every n epoch  
                    early_stopping(v_loss, model, optimizer, ep, lr_scheduler.get_last_lr())
                    
                    if early_stopping.early_stop: 
                        break

    def _fit_impl(self, model, optimizer, criterion, dataloader):
        model.train()
        total_loss = 0.0
        self._reset_metrics()
        
        for step, data in enumerate(dataloader):
            mask, input = data['mask'].to(self.cfg['device']), data[self.cfg['view']].to(self.cfg['device'])
            optimizer.zero_grad()         # Clear gradients
            out = model(input)
            loss = criterion(out, mask)   # Compute gradients
            loss.backward()               # Backward pass 
            optimizer.step()              # Update model parameters                                                       
            
            total_loss += loss.item() * out.size(0)
            self._update_metrics(out, mask)
            
        avg_loss = total_loss / len(dataloader.dataset)
        mcs = dict(zip(list(self.metrics.keys()), self._compute_metrics())) 
        return avg_loss, mcs
    
    def _val_impl(self, model, criterion, dataloader):
        model.eval()
        total_loss = 0.0
        self._reset_metrics()
        
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                mask, input = data['mask'].to(self.cfg['device']), data[self.cfg['view']].to(self.cfg['device'])
                out = model(input)
                loss = criterion(out, mask)   
                total_loss += loss.item() * out.size(0)
                self._update_metrics(out, mask)
            
        avg_loss = total_loss / len(dataloader.dataset)
        mcs = dict(zip(list(self.metrics.keys()), self._compute_metrics())) 
        return avg_loss, mcs

    def eval(self, model: torch.nn.Module, 
             criterion: Optional[Callable] = None,  
             dataloader: Optional[DataLoader] = None, 
             ckpt: Union[str, Path, None] = None,
             compute_stats: bool = True,
             verbose: bool = False):
        
        assert dataloader is not None, "Dataloader must be provided for evaluation."
        
        model = model.to(self.cfg['device'])
        
        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt)['model_state_dict'], strict=False)
            
        criterion = criterion or torch.nn.CrossEntropyLoss()
        
        eval_loss, metrics = self._eval_impl(model, criterion, dataloader, compute_stats)
        
        if verbose:
            summary(model)
            print(f"Loss/eval: {eval_loss:.2f}", end=" -- ")
            for name, val in metrics.items():
                print(f"{name}/eval: {val * 100:.2f}", end=" -- ")
            print()

        return metrics

    def _eval_impl(self, model: torch.nn.Module, 
                   criterion: Callable, 
                   dataloader: DataLoader, 
                   compute_stats: bool = True):
        model.eval()
        total_loss = 0.0
        self._reset_metrics()
        
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                mask, input = data['mask'].to(self.cfg['device']), data[self.cfg['view']].to(self.cfg['device'])
                out = model(input)
                loss = criterion(out, mask)   
                total_loss += loss.item() * out.size(0)
                self._update_metrics(out, mask)
            
        avg_loss = total_loss / len(dataloader.dataset)
        
        if compute_stats:
            metrics = dict(zip(list(self.metrics.keys()), self._compute_metrics())) 
        else:
            metrics = self.metrics

        return avg_loss, metrics
    
    def load_weights(self, model: torch.nn.Module, ckpt: Union[str, Path]):
        if isinstance(ckpt, str): ckpt = Path(ckpt)
        model.load_state_dict(self._load_ckpt(ckpt)['model_state_dict'], strict=False) 
        model.eval()
        model.to(self.cfg['device'])
        return model
    
    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        random.seed(seed)
        np.random.seed(seed)
        
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disables optimization for reproducibility
        
    def _save_ckpt(self, model, ckpt_name):
        ckpt_path = Path(self.path) / 'ckpt'
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path / ckpt_name)
        logging.info(f'Model checkpoint saved at {ckpt_path / ckpt_name}')
        
    def _load_ckpt(self, ckpt_name):
        ckpt_path = Path(self.path) / 'ckpt' / ckpt_name 
        return torch.load(ckpt_path, map_location='cpu')
    
    def _update_metrics(self, pred, target):
        for metric in self.metrics.values():
            metric.update(pred, target.long())
            
    def _compute_metrics(self):
        return [metric.compute().item() for metric in self.metrics.values()]
        
    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()
            
            
class SiameseTrainer:
    def __init__(self, cfg:Union[str, Path, Dict]):
        if isinstance(cfg, str or Path):
            with open(cfg, 'r') as f:
                self.cfg = yaml.safe_load(f)    
        elif isinstance(cfg, Dict):
            self.cfg = cfg
        else:
            raise ValueError("trainer cfg must be a string, Path, or dictionary.")
            
        self.epoch = self.cfg['epoch']
        self.path = self.cfg['path']
        self.patience = self.cfg['patience']
        self.lr = self.cfg['lr']
        self.w_decay = self.cfg['w_decay']
        self.optim = self.cfg['optim']
        
        self.metrics = {
            'Precision': Precision(task='binary', threshold=0.5).to(self.cfg['device']),
            'Recall': Recall(task='binary', threshold=0.5).to(self.cfg['device']),
            'F1Score': F1Score(task='binary', threshold=0.5).to(self.cfg['device']),
            'mIoU': JaccardIndex(task='binary', threshold=0.5).to(self.cfg['device'])
        }
        
        self.set_seed()

    def fit(self, model: torch.nn.Module, 
            criterion: Optional[Callable]=None,  
            train_loader: Optional[DataLoader]=None,
            val_loader: Optional[DataLoader]=None,
            ckpt: Union[str, Path, None]=None,
            save_period: int=1):
        summary(model)
        model = model.to(self.cfg['device'])
        
        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt)['model_state_dict'], strict=False)
        
        criterion = criterion or torch.nn.CrossEntropyLoss()

        if self.optim == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.w_decay
            )
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.w_decay
            )
        else:
            raise Exception("please specify an optimizer")
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epoch, 
            eta_min=1e-6
        )
        
        writer = SummaryWriter(log_dir=f'{self.path}/runs')
        early_stopping = EarlyStopping(path=self.path, patience=self.patience)
         
        for ep in tqdm(range(1, self.epoch + 1)):
            t_loss, t_mcs = self._fit_impl(model, optimizer, criterion, train_loader)
            writer.add_scalar('Loss/train', t_loss, ep)
            print('Loss/train', f"{t_loss:.2f}", end=' -- ')
            
            for mc_name, mc_value in t_mcs.items():
                writer.add_scalar(f'{mc_name}/train', mc_value, ep)
                print(mc_name+'/train', f"{mc_value * 100:.2f}", end=' -- ')

            # Adjust learning rate 
            lr_scheduler.step()
            writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
    
            if val_loader:
                v_loss, v_mcs = self._val_impl(model, criterion, val_loader) 
                writer.add_scalar('Loss/val', v_loss, ep)
                print('Loss/val', f"{v_loss:.2f}", end=' -- ')
                
                for mc_name, mc_value in v_mcs.items():
                    writer.add_scalar(f'{mc_name}/val', mc_value, ep)
                    print(mc_name+'/val', f"{mc_value * 100:.2f}", end=' -- ')
                    
                if ep % save_period == 0: # save model at every n epoch  
                    early_stopping(v_loss, model, optimizer, ep, lr_scheduler.get_last_lr())
                    
                    if early_stopping.early_stop: 
                        break

    def _fit_impl(self, model, optimizer, criterion, dataloader):
        model.train()
        total_loss = 0.0
        self._reset_metrics()
        
        for step, data in enumerate(dataloader):
            inputs = [data['mhs'].to(self.cfg['device']), data['vat'].to(self.cfg['device'])]
            mask = data['mask'].to(self.cfg['device'])
            optimizer.zero_grad()         # Clear gradients
            out = model(inputs)
            loss = criterion(out, mask)   # Compute gradients
            loss.backward()               # Backward pass 
            optimizer.step()              # Update model parameters                                                       
            
            total_loss += loss.item() * out.size(0)
            self._update_metrics(out, mask)
            
        avg_loss = total_loss / len(dataloader.dataset)
        mcs = dict(zip(list(self.metrics.keys()), self._compute_metrics())) 
        return avg_loss, mcs
    
    def _val_impl(self, model, criterion, dataloader):
        model.eval()
        total_loss = 0.0
        self._reset_metrics()
        
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                inputs = [data['mhs'].to(self.cfg['device']), data['vat'].to(self.cfg['device'])]
                mask = data['mask'].to(self.cfg['device'])
                out = model(inputs)
                loss = criterion(out, mask)   
                total_loss += loss.item() * out.size(0)
                self._update_metrics(out, mask)
            
        avg_loss = total_loss / len(dataloader.dataset)
        mcs = dict(zip(list(self.metrics.keys()), self._compute_metrics())) 
        return avg_loss, mcs

    def eval(self, model: torch.nn.Module, 
             criterion: Optional[Callable] = None,  
             dataloader: Optional[DataLoader] = None, 
             ckpt: Union[str, Path, None] = None,
             compute_stats: bool = True,
             verbose: bool = False):
        
        assert dataloader is not None, "Dataloader must be provided for evaluation."
        
        model = model.to(self.cfg['device'])
        
        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt)['model_state_dict'], strict=False)
            
        criterion = criterion or torch.nn.CrossEntropyLoss()

        eval_loss, metrics = self._eval_impl(model, criterion, dataloader, compute_stats)
        
        if verbose:
            summary(model)
            print(f"Loss/eval: {eval_loss:.2f}", end=" -- ")
            for name, val in metrics.items():
                print(f"{name}/eval: {val * 100:.2f}", end=" -- ")
            print()

        return metrics

    def _eval_impl(self, model: torch.nn.Module, 
                   criterion: Callable, 
                   dataloader: DataLoader, 
                   compute_stats: bool = True):
        model.eval()
        total_loss = 0.0
        self._reset_metrics()
        
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                inputs = [data['mhs'].to(self.cfg['device']), data['vat'].to(self.cfg['device'])]
                mask = data['mask'].to(self.cfg['device'])
                out = model(inputs)
                loss = criterion(out, mask)   
                total_loss += loss.item() * out.size(0)
                self._update_metrics(out, mask)
            
        avg_loss = total_loss / len(dataloader.dataset)
        
        if compute_stats:
            metrics = dict(zip(list(self.metrics.keys()), self._compute_metrics())) 
        else:
            metrics = self.metrics

        return avg_loss, metrics
    
    def load_weights(self, model: torch.nn.Module, ckpt: Union[str, Path]):
        if isinstance(ckpt, str): ckpt = Path(ckpt)
        model.load_state_dict(self._load_ckpt(ckpt)['model_state_dict'], strict=False) 
        model.eval()
        model.to(self.cfg['device'])
        return model
    
    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        random.seed(seed)
        np.random.seed(seed)
        
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disables optimization for reproducibility
        
    def _save_ckpt(self, model, ckpt_name):
        ckpt_path = Path(self.path) / 'ckpt'
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path / ckpt_name)
        logging.info(f'Model checkpoint saved at {ckpt_path / ckpt_name}')
        
    def _load_ckpt(self, ckpt_name):
        ckpt_path = Path(self.path) / 'ckpt' / ckpt_name 
        return torch.load(ckpt_path, map_location='cpu')
    
    def _update_metrics(self, pred, target):
        for metric in self.metrics.values():
            metric.update(pred, target.long())
            
    def _compute_metrics(self):
        return [metric.compute().item() for metric in self.metrics.values()]
        
    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()