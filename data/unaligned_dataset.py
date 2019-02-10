import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np


class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print('len(A),len(B)=', self.A_size, self.B_size)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d) / (%s, %s)' % (index_A, index_B, A_path, B_path))

        if 'npy' in A_path: # Very hacky 2d/3d npy load # TODO: proper data loading/normalization
            A_img = np.load(A_path)
            B_img = np.load(B_path)
            if len(A_img.shape) == 3:
                A = torch.from_numpy(A_img).unsqueeze(0).float() / 1000.0 # <-- CT normalization hack HU/1000
                B = torch.from_numpy(B_img).unsqueeze(0).float() / 1000.0
            elif len(A_img.shape) == 4:
                A = torch.from_numpy(A_img).float()
                B = torch.from_numpy(B_img).float()
            else:
                raise NotImplementedError('Unknown number of data dimensions:', A_img.shape)
        else:
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')

            A = self.transform(A_img)
            B = self.transform(B_img)
            if self.opt.which_direction == 'BtoA':
                input_nc = self.opt.output_nc
                output_nc = self.opt.input_nc
            else:
                input_nc = self.opt.input_nc
                output_nc = self.opt.output_nc

            if input_nc == 1:  # RGB to gray
                tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
                A = tmp.unsqueeze(0)

            if output_nc == 1:  # RGB to gray
                tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = tmp.unsqueeze(0)


        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
