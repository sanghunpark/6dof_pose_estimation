# system
import os
import ntpath
import random

import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset

# etc
from PIL import Image

class Linemod(Dataset):
    def __init__(self, root, objects = [], split = 'train', shuffle = True, transform = None) -> None:
        self.root = root
        self.transform = transform
        self.data_paths = self._get_data_path(objects, split)

        # if shuffle:
        #     random.shuffle(self.data_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index error'

        # get paths of other type data from image paths
        rgb_path = self.data_paths[index]
        mask_path = rgb_path.replace('JPEGImages', 'mask').replace('jpg', 'png')
        if not os.path.isfile(mask_path):
            mask_path = os.path.join(ntpath.dirname(mask_path), ntpath.basename(mask_path)[2:] )
        label_path = rgb_path.replace('JPEGImages', 'labels').replace('jpg', 'txt')

        # get data
        rgb = Image.open(rgb_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')
        label = torch.from_numpy(self._get_label(label_path))

        # transform image
        if self.transform is  not None:
            rgb = self.transform(rgb)
            mask = self.transform(mask)
        return {'rgb': rgb, 'mask': mask, 'label':label}

    def __len__(self):
        return len(self.data_paths)

    def _get_data_path(self, objects, split):
        # filter folders by 'objects'
        all_object_list = os.listdir(self.root)
        if len(objects) == 0:
            object_list = all_object_list
        else:
            object_list = set.intersection(set(objects), set(all_object_list))
        
        # make a list of data path
        data_path = []
        for idx, obj in enumerate(object_list):
            data_file = os.path.join(self.root, obj, split + '.txt')
            if not (os.path.isfile(data_file)):
                continue

            with open(data_file) as f:
                lines = f.readlines()                
                data_path.extend(['./data/' + s for s in map(str.rstrip, lines)]) # remove all '\n's and './data/' in lines
        return data_path

    def _get_label(self, label_path): # n_points: 2D bounding boxe corners (8) + 2D centroid point (1)
        # n_label = 2 * n_points + 3 # class idx + x range + y range (in 2D bounding boex)
        if os.path.isfile(label_path):
            label = np.loadtxt(label_path)
            return label
        else:
            return np.array([])