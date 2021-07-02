import os
import numpy as np
from PIL import Image
from skimage import io
from skimage.color import gray2rgb


import torch
from torch.utils.data import Dataset

import warnings

class AEI_Dataset(Dataset):
    def __init__(self, root, transform=None):
        super(AEI_Dataset, self).__init__()
        self.root = root
        print('Loading dataset from {}...'.format(root))
        self.files = [
            os.path.join(root, filename)
            for filename in os.listdir(root)
        ]
        print('Done loading dataset from {}'.format(root))
        self.transform = transform
        self.n_sources = len(self.files)
        self.nonidentical_to_identical_ratio = 5


    def __getitem__(self, index):
        """
        See Appendix B in: https://arxiv.org/pdf/1912.13457.pdf
        for more info.
        """
        s_idx = index % self.n_sources
        if index < self.n_sources:
            f_idx = s_idx
            same = torch.ones(1)
        else:
            f_idx = int(torch.randint(0, self.n_sources, (1,)))
            same = torch.zeros(1)

        with Image.open(self.files[f_idx]) as f_img:
            with Image.open(self.files[s_idx]) as s_img:
                f_img = f_img.convert('RGB')
                s_img = s_img.convert('RGB')
        
                if self.transform is not None:
                    f_img = self.transform(f_img)
                    s_img = self.transform(s_img)
        
                return f_img, s_img, same

    def __len__(self):
        return len(self.files) * self.nonidentical_to_identical_ratio


class AEI_Val_Dataset(Dataset):
    def __init__(self, root, transform=None):
        super(AEI_Val_Dataset, self).__init__()
        self.root = root
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transfrom = transform

    def __getitem__(self, index):
        l = len(self.files)

        f_idx = index // l
        s_idx = index % l

        if f_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        with Image.open(self.files[f_idx]) as f_img:
            with Image.open(self.files[s_idx]) as s_img:
                f_img = f_img.convert('RGB')
                s_img = s_img.convert('RGB')
        
                if self.transfrom is not None:
                    f_img = self.transfrom(f_img)
                    s_img = self.transfrom(s_img)
        
                return f_img, s_img, same

    def __len__(self):
        return len(self.files) * len(self.files)