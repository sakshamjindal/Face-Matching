from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from mtcnn.mtcnn import MTCNN

class Face_Dataset(Dataset):
    
    def __init__(self, root_dir='lfw/', file_root = "files/", train=False, test=False):
        
        self.train = train
        self.test = test
        self.root_dir = root_dir
        self.files = []
        self.labels = []
        
        dataset = set()
        if self.train:
            filename= file_root + 'couples/train.txt'
        else:
            filename= file_root + 'couples/test.txt'
            
        self.transform = self.load_transforms()

        with open(filename) as f:
            for line in f:
                line = line.split()
                self.files.append(line[:2])
                self.labels.append(int(line[2]))
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im_names = self.files[idx]
        im1 = Image.open(os.path.join(self.root_dir, im_names[0]))
        im2 = Image.open(os.path.join(self.root_dir, im_names[1]))
        label = np.array([abs(self.labels[idx])],dtype=np.float32)
        
        # cropping face from image
#         im1 = self.align_face(im1)
#         im2 = self.align_face(im2)
        
        #  applying augmentations on the crpped face
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
            
        label = torch.from_numpy(label)
        return im1, im2, label
    
    def align_face(self,im):
        im = np.asarray(im)
        detector = MTCNN()
        results = detector.detect_faces(im)

        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = im[y1:y2, x1:x2]

        face_image = Image.fromarray(face)

        return face_image
    
    def load_transforms(self):
        if self.train:
            tet = transforms.Compose([transforms.Resize((160,160)),
                                               transforms.RandomAffine(15),
                                               transforms.RandomVerticalFlip(),
                                               transforms.ColorJitter(
                                                          brightness=0.3,
                                                          contrast=0.3,
                                                          saturation=0.3),
                                               transforms.ToTensor()])
        else:
            tet = transforms.Compose([transforms.ToTensor()])
        return tet
    
    
class Face_Dataset_Triplet(Dataset):
    
    def __init__(self, root_dir='../lfw/', file_root = "../files/", train=False, test=False):
        
        self.train = train
        self.test = test
        self.root_dir = root_dir
        self.files = []
        self.labels = []
        
        dataset = set()
        if self.train:
            filename= file_root + 'triplets/train.txt'
        else:
            filename= file_root + 'triplets/test.txt'
            
        self.transform = self.load_transforms()

        with open(filename) as f:
            for line in f:
                line = line.split()
                self.files.append(line[:3])
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im_names = self.files[idx]
        im1 = Image.open(os.path.join(self.root_dir, im_names[0]))
        im2 = Image.open(os.path.join(self.root_dir, im_names[1]))
        im3 = Image.open(os.path.join(self.root_dir, im_names[2]))
        
        # cropping face from image
#         im1 = align_face(im1)
#         im2 = align_face(im2)
#         im3 = align_face(im3)
        
        #  applying augmentations on the crpped face
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
            im3 = self.transform(im3)
            
        return im1, im2, im3
    
    @staticmethod
    def align_face(im):
        im = np.asarray(im)
        detector = MTCNN()
        results = detector.detect_faces(im)

        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = im[y1:y2, x1:x2]

        face_image = Image.fromarray(face)

        return face_image
    
    def load_transforms(self):
        if self.train:
            tet = transforms.Compose([transforms.Resize((160,160)),
                                               transforms.RandomAffine(15),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ColorJitter(
                                                          brightness=0.3,
                                                          contrast=0.3,
                                                          saturation=0.3),
                                               transforms.ToTensor()])
        else:
            tet = transforms.Compose([transforms.ToTensor()])
        return tet