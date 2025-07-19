import torch 
import numpy as np
import geopandas as gpd

from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, List, Dict

from rio_tiler.io import COGReader
from shapely.geometry import box
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

from .data_utils import arr_to_tiff, tiff_to_arr


class BudjBimAreaDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        test_area: str = 'area4',
        train_size: float = 0.8,
        img_size: int = 256,
        grid_size: Optional[int] = 40,
        grid_stride: Optional[List[int]] = [20],
        download: bool = False,
        img_norm: bool = False,
        transform=None,
        target_transform=None
    ):
        self.root = Path(root) if isinstance(root, str) else root
        self.split = split
        self.img_size = img_size
        self.grid_size = grid_size
        self.grid_stride = grid_stride
        self.img_norm = img_norm
        self.transform = transform
        self.target_transform = target_transform

        valid_areas = ['area1', 'area2', 'area3', 'area4', 'area5', 'area6']
        if test_area not in valid_areas:
            raise ValueError(
                f"Invalid test area: {test_area}. Valid options are {valid_areas}."
            )

        self.train_val_areas = [area for area in valid_areas if area != test_area]
        self.test_area = test_area

        if not (0 < train_size <= 1.0):
            raise ValueError("train_size must be in the range (0, 1].")
        self.train_size = train_size

        dataset_url = {
            'rgb': None, 
            'mhs': None, 
            'vat': None, 
            'mask': None,
        }
        self.data_type = list(dataset_url.keys())

        if download:
            self._download_dataset(dataset_url)

        self._prepare_data_split(split)

        
    def _prepare_data_split(self, split):
        """
        Prepare train, validation, or test data splits.
        """
        random_state = 42
        for data_type in self.data_type:
            files = self._collect_files(data_type, split)
            
            if split == 'train':
                train_files, _ = train_test_split(files, train_size=self.train_size, random_state=random_state)
                setattr(self, f"{data_type}_list", train_files)
            elif split == 'val':
                _, val_files = train_test_split(files, train_size=self.train_size, random_state=random_state)
                setattr(self, f"{data_type}_list", val_files)
            else:
                setattr(self, f"{data_type}_list", files)
        
    def _download_dataset(self, dataset_url):
        """
        Download dataset using the provided URLs.
        """        
        if dataset_url['mask'] is None:
            raise ValueError("Mask URL must be provided for downloading.")
        
        for area_id, area in self._split_areas(dataset_url['mask']).iterrows():
            for data_type, url in dataset_url.items():
                output_dir = self.root / f"area{area_id+1}" / data_type
                
                if url and not output_dir.exists():
                    mask_url = dataset_url['mask']
                    area_bounds = area['geometry'].bounds
                    self._download_data(output_dir, data_type, url, mask_url, area_bounds)
                                
            print(f"Completed downloading {data_type} for BudjBimArea")
    
    def _download_data(self, data_path: Path, data_type: str, url: str, mask_url: str, area_bounds: tuple):       
        """
        Download specific data type (e.g., RGB, HS, VAT and Mask) into the given directory.
        """ 
        data_path.mkdir(parents=True)
        with COGReader(url) as cog:
            crs = cog.dataset.crs
            grid_df = self._generate_grids_by_mask_areas(mask_url, self.grid_size, self.grid_stride, area_bounds)
            for _, row in tqdm(grid_df.iterrows(), total=len(grid_df), desc=f"Downloading {data_type}"):
                geometry = row['geometry']
                part = cog.part(geometry.bounds, crs, crs)
                output_file = data_path / f"e{int(geometry.centroid.x)}_n{int(geometry.centroid.y)}_{crs}.tif"
                arr_to_tiff(part.data_as_image(), output=output_file, crs=crs, transform=part.transform)

    def _collect_files(self, data_type: str, split: str) -> List[Path]:        
        """
        Collect available files for a specific data type and split.
        """
        areas = [self.test_area] if split == 'test' else self.train_val_areas
        files = []
        
        for area in areas:         
            area_path = self.root / area / data_type   
            area_files = sorted(area_path.glob('*.tif'))
            
            if not area_files:
                print(f"Warning: No files found for {data_type} in {area_path}")
            files.extend(area_files)
            
        return files
    
    def _split_areas(self, url: str, w_factor: int = 2, h_factor: int = 3):        
        areas = []
        
        with COGReader(url) as cog:
            crs = cog.dataset.crs
            # Tile bounds
            x_min, y_min, x_max, y_max = cog.bounds       
            # Tile size
            w, h = x_max - x_min, y_max - y_min
            # Area size
            area_w = w // w_factor
            area_h = h // h_factor
            # Top-left corner
            window_bounds = (x_min, y_max - area_h, x_min + area_w, y_max)        
            
            for _, i in enumerate(list(range(0, int(w // area_w)))):
                for j in list(range(0, int(h // area_h))):
                    area_bounds = (window_bounds[0] + i * area_w,
                                   window_bounds[1] + j * -area_h,
                                   window_bounds[2] + i * area_w,
                                   window_bounds[3] + j * -area_h)
                    areas.append(box(*area_bounds))            
                    
            return gpd.GeoDataFrame(geometry=areas, crs=crs.data['init'])
        
    def _generate_grids_by_mask_areas(self, mask_url: str, size: int, stride: List[int], area_bounds: tuple = None, thred: float = 0.02): 
        """
        Generate grid-based subdivisions for a given area.
        """       
        grids = []
        
        with COGReader(mask_url) as cog:
            crs = cog.dataset.crs
            bounds = cog.bounds if area_bounds is None else area_bounds
            # Tile bounds
            x_min, y_min, x_max, y_max = bounds
            # Tile size
            w, h = x_max - x_min, y_max - y_min
            # Top-left corner
            window_bounds = (x_min, y_max - size, x_min + size, y_max)
        
            for s in stride:
                # Move from top to down, left to right
                for i in range(0, int(w // s)):
                    for j in range(0, int(h // s)):
                        temp_bounds = (window_bounds[0] + i * s,
                                       window_bounds[1] + j * -s,
                                       window_bounds[2] + i * s,
                                       window_bounds[3] + j * -s)

                        cog_part = cog.part(temp_bounds, crs, crs)
                        cog_img = np.asarray(cog_part.data_as_image())
                        
                        if np.count_nonzero(cog_img > 0) / cog_img.size > thred:
                            grids.append(box(*temp_bounds))          

        return gpd.GeoDataFrame(geometry=grids, crs=crs.data['init'])
    
    def __len__(self):
        if hasattr(self, 'mask_list'):
            return len(self.mask_list)
        else:
            raise AttributeError("Dataset file list is not initialized. Check the split setup.")

    def __getitem__(self, idx):
        data_sample = {}
        state = torch.get_rng_state()
        
        for data_type in self.data_type:
            file_list = getattr(self, f'{data_type}_list')
            
            if file_list:
                data_path = file_list[idx]
                data = T.ToPILImage()(tiff_to_arr(data_path, data_type))
                
                if self.split == 'train':
                    aug_type = 'mask' if data_type == 'mask' else data_type
                else:
                    aug_type = 'plain_mask' if data_type == 'mask' else 'plain'

                transform = self.transform or self.data_augmentations[aug_type]
                data = transform(data)
                
                torch.set_rng_state(state)
                data_sample[data_type] = data

        return data_sample
    
    @property
    def data_augmentations(self) -> Dict:
        norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        to_tensor = [T.ToTensor(), norm] if self.img_norm else [T.ToTensor()]

        return {
            'rgb': T.Compose([
                T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomRotation(
                    degrees=180,
                    interpolation=T.InterpolationMode.BICUBIC
                ),
                T.ColorJitter(
                    hue=0.05,
                    saturation=0.25,
                    brightness=0.25,
                    contrast=0.25
                ),
                *to_tensor
            ]),
            'mhs': T.Compose([
                T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomRotation(
                    degrees=180,
                    interpolation=T.InterpolationMode.BICUBIC
                ),
                *to_tensor
            ]),
            'vat': T.Compose([
                T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomRotation(
                    degrees=180,
                    interpolation=T.InterpolationMode.BICUBIC
                ),
                *to_tensor
            ]),
            'mask': T.Compose([
                T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST),
                T.RandomRotation(
                    degrees=180,
                    interpolation=T.InterpolationMode.NEAREST
                ),
                T.ToTensor()
            ]),
            'plain': T.Compose([
                T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC),
                *to_tensor
            ]),
            'plain_mask': T.Compose([
                T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor()
            ])
        }


class BudjBimLandscapeDataset(Dataset):
    def __init__(
        self, 
        root: Union[str, Path], 
        grid_size: Optional[int] = 40,
        grid_stride: Optional[List[int]] = [40],
        grid_file: str = 'grids.gpkg',
        download: bool = False,
        transform = None,
        target_transform = None
    ):        
        self.root = Path(root) if isinstance(root, str) else root
        self.grid_size = grid_size
        self.grid_stride = grid_stride
        self.grid_file = grid_file
        self.transform = transform
        self.target_transform = target_transform
        
        dataset_url = {
            'rgb': None, 
            'mhs': None, 
            'vat': None, 
        }
        self.data_type = list(dataset_url.keys())

        if download: 
            self._download_dataset(dataset_url)
            
        for data_type in self.data_type:            
            file_path = self.root / data_type
            files = sorted(file_path.glob('*.tif'))
            if not files:
                raise FileNotFoundError(f"No files found for {data_type} in {file_path}.")
            setattr(self, f"{data_type}_list", files)
                    
    def _download_dataset(self, dataset_url):
        for data_type, url in dataset_url.items():
            output_dir = self.root / data_type
            if url and not output_dir.exists():
                self._download_data(output_dir, data_type, url)
                
        print(f"Completed downloading {data_type} for BudjBimLandscape")           

    def _download_data(self, data_path: Path, data_type: str, url: str):        
        data_path.mkdir(parents=True, exist_ok=True)
        with COGReader(url) as cog:
            crs = cog.dataset.crs
            if not (self.root / self.grid_file).exists():
                print('Generating grids for BudjBimLandscape')
                grid_df = self._generate_grids(url, self.grid_size, self.grid_stride)
                grid_df.to_file(self.root / self.grid_file, driver="GPKG")
            else:
                print('Reading grids for BudjBimLandscape')
                grid_df = gpd.read_file(self.root / self.grid_file)
                
            for _, row in tqdm(grid_df.iterrows(), total=len(grid_df), desc=f"Downloading {data_type}"):
                geometry = row['geometry']
                part = cog.part(geometry.bounds, crs, crs)
                output_file = data_path / f"e{int(geometry.centroid.x)}_n{int(geometry.centroid.y)}_{crs}.tif"
                arr_to_tiff(part.data_as_image(), output=output_file, crs=crs, transform=part.transform)
                    
    def _generate_grids(self, url: str, size: int, stride: List[int]):
        grids = []
        
        with COGReader(url) as cog:
            crs = cog.dataset.crs
            # Tile bounds
            x_min, y_min, x_max, y_max = cog.bounds
            # Tile size
            w, h = x_max - x_min, y_max - y_min
            # Top-left corner
            window_bounds = (x_min, y_max - size, x_min + size, y_max)
        
            for s in stride:                        
                for i in tqdm(range(0, int(w // s))):
                    for j in range(0, int(h // s)):
                        temp_bounds = (window_bounds[0] + i * s,
                                       window_bounds[1] + j * -s,
                                       window_bounds[2] + i * s,
                                       window_bounds[3] + j * -s)
                        cog_part = cog.part(temp_bounds, crs, crs)
                        cog_img = np.asarray(cog_part.data_as_image())
                        temp_box = box(*temp_bounds)
                        
                        if cog_img.max() > 0:
                            grids.append(temp_box)
            
        return gpd.GeoDataFrame(geometry=grids, crs=crs.data['init'])
    
    def __len__(self):
        if hasattr(self, 'mhs_list'):
            return len(getattr(self, 'mhs_list'))
        elif hasattr(self, 'vat_list'):
            return len(getattr(self, 'vat_list'))
        else:
            raise('Unable to find corresponding data list')
        
    def __getitem__(self, idx):
        data_sample = {}
        
        for data_type in self.data_type:
            file_list = getattr(self, f'{data_type}_list')
            
            if file_list:
                data_path = file_list[idx]
                data = T.ToPILImage()(tiff_to_arr(data_path, data_type))
    
                transform = self.transform if self.transform else T.ToTensor()
                data = transform(data)
                
                data_sample['file_name'] = data_path.name                
                data_sample[data_type] = data

        return data_sample