import torch
import torchvision
import random
from re import L
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image
totensor = torchvision.transforms.ToTensor()
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
A_ANNOT_FOLDER_NAME = "A_L"
B_ANNOT_FOLDER_NAME = "B_L"

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name, annot):
    return os.path.join(root_dir, annot, img_name) #.replace('.jpg', label_suffix)

num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def TensorIndex2Color(pred):
    colormap = torch.as_tensor(ST_COLORMAP, dtype=torch.uint8)
    x = pred.long()
    return colormap[x, :]

def transform_augment_cd(img, split='val', min_max=(0, 1)):
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img
class SCDDataset(Dataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    datafolder-tree
    dataroot:.
            ├─A
            ├─B
            ├─A_L
            ├─B_L
    """

    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        
        self.res = resolution
        self.data_len = data_len
        self.split = split

        self.root_dir = dataroot
        self.split = split  #train | val | test
        
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        
        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # 读取双时相图片路径
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])
        # 读取图片，转成rgb
        img_A   = Image.open(A_path).convert("RGB")
        img_B   = Image.open(B_path).convert("RGB")
        # 获取label路径
        AL_path  = get_label_path(self.root_dir, self.img_name_list[index % self.data_len], A_ANNOT_FOLDER_NAME)
        BL_path  = get_label_path(self.root_dir, self.img_name_list[index % self.data_len], B_ANNOT_FOLDER_NAME)
        # 转成分类索引0,1,2,..,7
        img_lb_Al = Color2Index(Image.open(AL_path).convert("RGB"))
        img_lb_Bl = Color2Index(Image.open(BL_path).convert("RGB"))
        # 数据增强，归一化
        img_A   = transform_augment_cd(img_A, split=self.split, min_max=(-1, 1))
        img_B   = transform_augment_cd(img_B, split=self.split, min_max=(-1, 1))
        img_lbAl = transform_augment_cd(img_lbAl, split=self.split, min_max=(0, 1))
        img_lbBl = transform_augment_cd(img_lbBl, split=self.split, min_max=(0, 1))
        image_id = A_path.split("/")[-1].split(".")[0]
        CD_L = (img_A>0).astype(np.uint8)
        # 返回数据
        return {'A': img_A, 'B': img_B, 'AL': img_lb_Al, 'BL': img_lb_Bl,'CD_L': CD_L, 'Index': image_id}

