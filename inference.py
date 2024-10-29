import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import torch
import matplotlib.pyplot as plt
           ###########  Dataloader  #############
NUM_WORKERS = 8
PIN_MEMORY=True
DIM_ = 256
   
from loader import Data_Loader_val

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from medpy import metric
import kornia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_edges(image,three):
    three = np.stack((three,)*3, axis=2)
    three =torch.tensor(three)
    three = np.transpose(three, (2,0,1))  ## to bring channel first 
    three= torch.unsqueeze(three,axis = 0)
    magnitude, edges=kornia.filters.canny(three, low_threshold=0.1, high_threshold=0.2, kernel_size=(7, 7), sigma=(1, 1), hysteresis=True, eps=1e-06)
    image[np.where(edges[0,0,:,:]!=0)] = 1
    return image
    
def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def blend(image,LV,MYO,RV,three): 

    image = normalize(image)
    image = np.stack((image,)*3, axis=2)
    image[np.where(LV==1)] = [0.9,0.9,0]
    image[np.where(MYO==1)] = [0.9,0,0]
    image[np.where(RV==1)] = [0,0,0.9]
    image = make_edges(image,three)
    return image

  #### Specify all the paths here #####
   
def calculate_metric_percase(pred, gt):      
      gt = gt[0,:]
      pred = pred[0,:]
      dice = metric.binary.dc(pred, gt)
      hd = metric.binary.hd95(pred, gt)
      return dice, hd
  
def Average(lst): 
    return sum(lst) / len(lst)


Dice_LV_LA  = []
HD_LV_LA = []
        
Dice_MYO_LA  = []
HD_MYO_LA = []
        
Dice_RV_LA  = []
HD_RV_LA = []

def check_Dice_Score(loader, model1, device=DEVICE):
    
        Dice_LV  = 0
        HD_LV = 0
        
        Dice_MYO  = 0
        HD_MYO = 0
        
        Dice_RV  = 0
        HD_RV = 0
    
        loop = tqdm(loader)
        model1.eval()
                   
        
        for batch_idx, (img,gt,name) in enumerate(loop):
    
            img = img.to(device=DEVICE,dtype=torch.float)  
            gt = gt.to(device=DEVICE,dtype=torch.float)  

        
            with torch.no_grad(): 
                
                pre_2d = model1(img) 
                pred = torch.argmax(pre_2d, dim=1)
                
                out_LV = torch.zeros_like(pred)
                out_LV[torch.where(pred==1)] = 1           
                out_MYO = torch.zeros_like(pred)
                out_MYO[torch.where(pred==2)] = 1
                out_RV = torch.zeros_like(pred)
                out_RV[torch.where(pred==3)] = 1
                
                
                single_lv,single_hd_lv = 0,0
                single_myo,single_hd_myo = 0,0
                single_rv,single_hd_rv  = 0,0
                
                if torch.sum(out_LV)!=0: 
                  single_lv,single_hd_lv = calculate_metric_percase(out_LV.detach().cpu().numpy(),gt[:,1,:].detach().cpu().numpy())
                if torch.sum(out_MYO)!=0: 
                  single_myo,single_hd_myo = calculate_metric_percase(out_MYO.detach().cpu().numpy(),gt[:,2,:].detach().cpu().numpy())
                if torch.sum(out_RV)!=0: 
                  single_rv,single_hd_rv = calculate_metric_percase(out_RV.detach().cpu().numpy(),gt[:,3,:].detach().cpu().numpy())
                
                Dice_LV+=single_lv
                HD_LV+=single_hd_lv
                
                Dice_MYO+=single_myo
                HD_MYO+=single_hd_myo
                
                Dice_RV+=single_rv
                HD_RV+=single_hd_rv
                
                img = img.detach().cpu().numpy()
                gt = gt.detach().cpu().numpy()
                
                out_LV = out_LV.detach().cpu().numpy()
                out_MYO = out_MYO.detach().cpu().numpy()
                out_RV = out_RV.detach().cpu().numpy()
                   
                pred_blend = blend(img[0,0,:],out_LV[0,:],out_MYO[0,:],out_RV[0,:],1-gt[0,0,:])
                plt.imsave(viz_pred_path +  name[0]  + '.png', pred_blend)
                

        print("for  fold -->", fold)
        print(' :: Dice Scores ::')
        print(f"Dice_LV  : {Dice_LV/len(loader)}")
        print(f"Dice_MYO  : {Dice_MYO/len(loader)}")
        print(f"Dice_RV  : {Dice_RV/len(loader)}")
        print(' :: HD Scores :: ')
        print(f"HD_LV  : {HD_LV/len(loader)}")
        print(f"HD_MYO  : {HD_MYO/len(loader)}")
        print(f"HD_RV  : {HD_RV/len(loader)}")
        print("                   ")
        
        return Dice_LV/len(loader), Dice_MYO/len(loader),Dice_RV/len(loader), HD_LV/len(loader),HD_MYO/len(loader),HD_RV/len(loader)
for fold in range(1,6):
    
    fold = str(fold)  ## training fold number     
    
    path_to_checkpoints  = "/data/scratch/acw676/Mamba_Data/NEW_DATA/Data_Aug/MNM_OTHER/F"+fold+_"Mamba_1.pth.tar"
    
    viz_gt_path = '/data/scratch/acw676/Mamba_Data/NEW_DATA/Visual_Res/swinunet/gts/F'+fold+'/'
    viz_pred_path =  '/data/scratch/acw676/Mamba_Data/NEW_DATA/Data_Aug/MNM_OTHER/Visual_R_MNM/Mamba_UNet/F'+fold+'/'
    val_imgs  =  "/data/scratch/acw676/Mamba_Data/NEW_DATA/Aug_MnM2/new_split_mix/F"+fold+"/val/imgs/"
    
    Batch_Size = 1
    val_loader = Data_Loader_val(val_imgs,batch_size = Batch_Size)
    print(len(val_loader))   ### same here
      
    from pre1 import EncoderDecoder,CustomEncoder
    encoder = CustomEncoder()
    model_1 = EncoderDecoder(encoder)


    def eval_():
        model = model_1.to(device=DEVICE,dtype=torch.float)
        checkpoint = torch.load(path_to_checkpoints,map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        
        Dice_LV, Dice_MYO,Dice_RV,HD_LV,HD_MYO,HD_RV= check_Dice_Score(val_loader, model, device=DEVICE)
        
        Dice_LV_LA.append(Dice_LV)
        Dice_MYO_LA.append(Dice_MYO)
        Dice_RV_LA.append(Dice_RV)
        
        HD_LV_LA.append(HD_LV)
        HD_MYO_LA.append(HD_MYO)
        HD_RV_LA.append(HD_RV)
    
    
    if __name__ == "__main__":
        eval_()
        
print("Average Five Fold Dice_LV_LA  --> ", Average(Dice_LV_LA))
print("Average Five Fold Dice_MYO_LA  --> ", Average(Dice_MYO_LA))
print("Average Five Fold Dice_RV_LA  --> ", Average(Dice_RV_LA))

print("Average Five Fold HD_LV_LA  --> ", Average(HD_LV_LA))
print("Average Five Fold HD_MYO_LA  --> ", Average(HD_MYO_LA))
print("Average Five Fold HD_RV_LA  --> ", Average(HD_RV_LA))
