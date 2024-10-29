import torch
import torch.optim as optim
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def check_Dice_Score(loader, model1, device=DEVICE):
    
    Dice_score_LA = 0
    Dice_score_RA = 0
    Dice_score_LV = 0
    Dice_score_RV = 0
        
    loop = tqdm(loader)
    model1.eval()
    
    for batch_idx, (img,gt) in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
        
        with torch.no_grad(): 
            pre_2d= model1(img) 
            
            ## segemntaiton ##
            
            pred = torch.argmax(pre_2d, dim=1)

            out_LA = torch.zeros_like(pred)
            out_LA[torch.where(pred==1)] = 1
                
            out_RA = torch.zeros_like(pred)
            out_RA[torch.where(pred==2)] = 1
                
            out_LV = torch.zeros_like(pred)
            out_LV[torch.where(pred==3)] = 1
            
            out_RV = torch.zeros_like(pred)
            out_RV[torch.where(pred==4)] = 1
            
                        
            single_LA = (2 * (out_LA * gt[:,1,:]).sum()) / (
               (out_LA + gt[:,1,:]).sum() + 1e-8)
            
            Dice_score_LA +=single_LA
            
            single_RA = (2 * (out_RA * gt[:,2,:]).sum()) / (
               (out_RA + gt[:,2,:]).sum() + 1e-8)
            
            Dice_score_RA +=single_RA
            
            
            single_LV = (2 * (out_LV * gt[:,3,:]).sum()) / (
               (out_LV + gt[:,3,:]).sum() + 1e-8)
            
            Dice_score_LV +=single_LV
            
            single_RV = (2 * (out_RV * gt[:,4,:]).sum()) / (
               (out_RV + gt[:,4,:]).sum() + 1e-8)
            
            Dice_score_RV +=single_RV
            

    ## segemntaiton ##
    print(f"Dice_score_LA  : {Dice_score_LA/len(loader)}")
    print(f"Dice_score_RA  : {Dice_score_RA/len(loader)}")
    print(f"Dice_score_LV  : {Dice_score_LV/len(loader)}")
    print(f"Dice_score_RV  : {Dice_score_RV/len(loader)}")

    Overall_Dicescore_LA = (Dice_score_LA + Dice_score_RA + Dice_score_LV + Dice_score_RV )/4
    
    print(f"Overall_Dicescore_LA  : {Overall_Dicescore_LA/len(loader)}")
    
    return Overall_Dicescore_LA/len(loader)
    

def train_fn(loader_train1,loader_valid1,model1, optimizer1, scaler1,loss_fn_DC1): ### Loader_1--> ED and Loader2-->ES

    train_losses1_seg  = [] # loss of each batch
    valid_losses1_seg  = []  # loss of each batch
    
    
    loop = tqdm(loader_train1)
    model1.train()
    
    for param_group in optimizer1.param_groups:
      print(f"Current learning rate: {param_group['lr']}")
    
    
    for batch_idx,(img,gt)  in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
       

        with torch.cuda.amp.autocast():
            pre_2d = model1(img)    ## loss1 is for 4 classes
            gt = torch.argmax(gt, dim=1)  ## used for Loss1
            ## segmentation losses ##
            loss = loss_fn_DC1(pre_2d,gt)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), 2.0)
            optimizer1.first_step(zero_grad=True)

            loss_fn_DC1(model1(img),gt).backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), 2.0)
            optimizer1.second_step(zero_grad=True)
                    
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        
        train_losses1_seg.append(float(loss))
        
    loop_v = tqdm(loader_valid1)
    model1.eval() 
    for batch_idx,(img,gt) in enumerate(loop_v):
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
       
        with torch.no_grad(): 
            pre_2d  =  model1(img)    ## loss1 is for 4 classes
            ## segmentation losses ##
            gt = torch.argmax(gt, dim=1)  ## used for Loss1
            loss = loss_fn_DC1(pre_2d,gt)
            
        # backward
        loop_v.set_postfix(loss = loss.item())
        valid_losses1_seg.append(float(loss))

    train_loss_per_epoch1_seg = np.average(train_losses1_seg)
    valid_loss_per_epoch1_seg  = np.average(valid_losses1_seg)
    
    avg_train_losses1_seg.append(train_loss_per_epoch1_seg)
    avg_valid_losses1_seg.append(valid_loss_per_epoch1_seg)
        
    
    return train_loss_per_epoch1_seg,valid_loss_per_epoch1_seg

  
from Loss3 import DiceCELoss
loss_fn_DC1 = DiceCELoss()


from loader import Data_Loader_val,Data_Loader_train
for fold in range(1,6):

  from pre2 import EncoderDecoder,CustomEncoder


  fold = str(fold)  ## training fold number 
  
  ### Data is arranged as follows;
# Data_CMR/F1/train/imgs 
#               /gts

# Data_CMR/F2/train/imgs 
#               /gts

# Data_CMR/F3/train/imgs 
#               /gts

# Data_CMR/F4/train/imgs 
#               /gts

# Data_CMR/F5/train/imgs 
#               /gts



  
  train_imgs = "Data_CMR/F"+fold+"/train/imgs/"  ## ABSOLUTE PATHs
  val_imgs  = "Data_CMR/F"+fold+"/val/imgs/"
  
  Batch_Size = 16
  Max_Epochs = 500
  
  train_loader = Data_Loader_train(train_imgs,batch_size = Batch_Size) # Data_Loader_io_transforms
  val_loader = Data_Loader_val(val_imgs,batch_size = 1)
  
  
  print(len(train_loader)) ### this shoud be = Total_images/ batch size
  print(len(val_loader))   ### same here
  #print(len(test_loader))   ### same here
  
  avg_train_losses1_seg = []   # losses of all training epochs
  avg_valid_losses1_seg = []  #losses of all training epochs
    
  avg_valid_DS_ValSet_seg = []  # all training epochs
  avg_valid_DS_TrainSet_seg = []  # all training epochs
    
  path_to_save_Learning_Curve = '/data/scratch/acw676/Mamba_Data/NEW_DATA/Data_Aug/DC1_CMR/'+'/F'+fold+'Mamba_1'
  path_to_save_check_points = '/data/scratch/acw676/Mamba_Data/NEW_DATA/Data_Aug/DC1_CMR/'+'/F'+fold+'Mamba_1'
  
  ### 3 - this function will save the check-points 
  def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
      print("=> Saving checkpoint")
      torch.save(state, filename)

  encoder = CustomEncoder()
  
  ### Freeze Or Unfreeze this part to use the pre-trained weights  ####
  
#  pretrained_weights_path = "/data/scratch/acw676/Mamba_Data/imageNet_weights/Mamba_160.pth.tar"
#  checkpoint = torch.load(pretrained_weights_path, map_location=DEVICE)
#  encoder.load_state_dict(checkpoint['state_dict'])
  
  Mamba_Model = EncoderDecoder(encoder)    
      
  model_1 =  Mamba_Model  # SwinUNET_R 
  epoch_len = len(str(Max_Epochs))
  
    # Variable to keep track of maximum Dice validation score

  def main():
      max_dice_val = 0.0
      model1 = model_1.to(device=DEVICE,dtype=torch.float)
      scaler1 = torch.cuda.amp.GradScaler()
      
      optimizer1 = optim.AdamW(model1.parameters(),betas=(0.5, 0.55),lr=0.001) #  0.00005
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer1,milestones=[100,200,300,400,500], gamma=0.5)

      for epoch in range(Max_Epochs):
          
          train_loss_seg ,valid_loss_seg = train_fn(train_loader,val_loader, model1, optimizer1,scaler1,loss_fn_DC1)
          scheduler.step()
          
          print_msg1 = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss_seg: {train_loss_seg:.5f} ' +
                     f'valid_loss_seg: {valid_loss_seg:.5f}')
        
          
          print(print_msg1)

          Dice_val = check_Dice_Score(val_loader, model1, device=DEVICE)
          avg_valid_DS_ValSet_seg.append(Dice_val.detach().cpu().numpy()) 
          
          
          if Dice_val > max_dice_val:
              max_dice_val = Dice_val
            # Save the checkpoint
              checkpoint = {
                  "state_dict": model1.state_dict(),
                  "optimizer": optimizer1.state_dict(),
                  }
              save_checkpoint(checkpoint)
                        
  if __name__ == "__main__":
      main()
  
  fig = plt.figure(figsize=(10,8))
    
  plt.plot(range(1,len(avg_train_losses1_seg)+1),avg_train_losses1_seg, label='Training Segmentation Loss')
  plt.plot(range(1,len(avg_valid_losses1_seg)+1),avg_valid_losses1_seg,label='Validation Segmentation Loss')
  
  plt.plot(range(1,len(avg_valid_DS_ValSet_seg)+1),avg_valid_DS_ValSet_seg,label='Validation DS')

    # find position of lowest validation loss
  minposs = avg_valid_losses1_seg.index(min(avg_valid_losses1_seg))+1 
  plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')
  font1 = {'size':20}
  plt.title("Learning Curve Graph",fontdict = font1)
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.ylim(-1, 1) # consistent scale
  plt.xlim(0, len(avg_train_losses1_seg)+1) # consistent scale
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')
