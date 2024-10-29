import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import torch
import matplotlib.pyplot as plt
#from typing import List, Union, Tuple

import torchio as tio
           ###########  Dataloader  #############
NUM_WORKERS = 8
PIN_MEMORY=True
DIM_ = 160
      
def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):# org_dim3->numof channels
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_


def Normalization_1(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 

# Define geometric transformations
geometrical_transforms = tio.OneOf({
    tio.RandomFlip(axes=(1, 2)): 0.5,  # Probability for each geometric transformation
    tio.RandomAffine(degrees=(-45, 45), center='image'): 0.5,
})

# Define intensity transformations
intensity_transforms = tio.OneOf({
    tio.RandomBlur(): 0.25,  # Probability for each intensity transformation
    tio.RandomGamma(log_gamma=(-0.2, -0.2)): 0.25,
    tio.RandomNoise(mean=0.1, std=0.1): 0.25,
    tio.RandomGhosting(axes=(1, 2)): 0.25,
})

# Combine all transformations and probabilities in a single Compose
transforms_2d = tio.Compose([
    tio.OneOf({
        geometrical_transforms: 0.4,  # Probability of applying geometric transformations
        intensity_transforms: 0.4,    # Probability of applying intensity transformations
        tio.Lambda(lambda x: x): 0.2  # Probability of no augmentation
    })
])

   
def generate_label(gt):
        temp_ = np.zeros([5,DIM_,DIM_])
        temp_[0:1,:,:][np.where(gt==0)]=1
        temp_[1:2,:,:][np.where(gt==1)]=1
        temp_[2:3,:,:][np.where(gt==2)]=1
        temp_[3:4,:,:][np.where(gt==3)]=1
        temp_[4:5,:,:][np.where(gt==4)]=1
        return temp_


class Dataset_io(Dataset): 
    def __init__(self, images_folder,transformations=transforms_2d):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        self.images_name = os.listdir(images_folder)
        self.transformations = transformations
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3)) 
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_1(img)
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-11]+'_gt.nii.gz'        
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        gt = gt.astype(np.float64)
        
        gt = np.expand_dims(gt, axis=0)
        img = np.expand_dims(img, axis=0)
        
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]
        img = Cropping_3d(C,H,W,DIM_,img)
        
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_3d(C,H,W,DIM_,gt)
        
        ## apply augmentaitons here ###
        
        img = np.expand_dims(img, axis=3)
        gt = np.expand_dims(gt, axis=3)

        d = {}
        d['Image'] = tio.Image(tensor = img, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = gt, type=tio.LABEL)
        sample = tio.Subject(d)
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img = transformed_tensor['Image'].data
            gt = transformed_tensor['Mask'].data
    
        gt = gt[...,0]
        img = img[...,0] 
        
        gt = generate_label(gt)

        return img,gt
    
def Data_Loader_io_transforms(images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

    
class Dataset_val(Dataset): 
    def __init__(self, images_folder):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        self.images_name = os.listdir(images_folder)
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3)) 
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_1(img)
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-11]+'_gt.nii.gz'        
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        gt = gt.astype(np.float64)
        
        gt = np.expand_dims(gt, axis=0)
        img = np.expand_dims(img, axis=0)
        
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]
        img = Cropping_3d(C,H,W,DIM_,img)
        
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_3d(C,H,W,DIM_,gt)
        gt = generate_label(gt)

        return img,gt
        
def Data_Loader_val(images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val(images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader
