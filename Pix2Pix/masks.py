
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2 
from os.path import splitext
from os import listdir
import logging
from torchvision import transforms
import random

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, size=32, train=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.len = len(self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        if train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((size, size)),
                transforms.CenterCrop((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5), (0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((size, size)),
                transforms.Normalize((0.5), (0.5))
            ])

    def __getitem__(self, i):
        
        idx = self.ids[i]
        mask_file = self.masks_dir + str(idx) + '.png'
        #print(mask_file)
        img_file = self.imgs_dir + str(idx) + '.png'
        #print(img_file)
        
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        #print("image: ", image)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        #print("mask: ", mask)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask
    def __len__(self):
        return self.len