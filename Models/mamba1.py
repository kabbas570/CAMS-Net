import torch
import torch.nn as nn
import torch 
import torch.nn as nn
from mamba_ssm import Mamba
import math
import numpy as np

def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_state = 2
Expand = 1
drop_rate = 0.1

def flip_Dim(x,dim):
    if dim == '98':
        x = torch.flip(x,[2,3])
    return x

def flip_Dim_back(x,dim):
    if dim == '98':
        x = torch.flip(x,[2,3])
    return x

class Linear_Layer(nn.Module):
    def __init__(self, n_channels, out_channels, Apply_Norm=False,Apply_Act=True, Apply_Dropout=True, bias=False):
        super(Linear_Layer, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(self.n_channels, self.out_channels, bias=bias)
        self.norm = nn.LayerNorm(normalized_shape=self.out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=drop_rate)
        
        self.Apply_Norm = Apply_Norm
        self.Apply_Act = Apply_Act
        self.Apply_Dropout = Apply_Dropout
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # [B, H*W, C]
        x = self.linear(x)
        
        if self.Apply_Norm:
            x = self.norm(x)
        if self.Apply_Act:
            x = self.act(x)
        if self.Apply_Dropout:
            x = self.dropout(x)
        x = x.view(b, h, w, self.out_channels).permute(0, 3, 1, 2)
        return x
    

class Linear_Layer_SP_Res(nn.Module):
    def __init__(self, sp_dim1,sp_dim2,in_channels,out_channels,Apply_Norm=False,Apply_Act=True, Apply_Dropout=True, bias=False):
        super(Linear_Layer_SP_Res, self).__init__()
        self.sp_dim1 = sp_dim1
        self.sp_dim2 = sp_dim2
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear_spa = nn.Linear(sp_dim1*sp_dim1,sp_dim2*sp_dim2)
        self.lin_chan = Linear_Layer(self.in_channels,self.out_channels)
        
        self.norm = nn.LayerNorm(normalized_shape=sp_dim2*sp_dim2)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=drop_rate)
        
        
        self.Apply_Norm = Apply_Norm
        self.Apply_Act = Apply_Act
        self.Apply_Dropout = Apply_Dropout

    def forward(self, x):
    
        b,c,h,w = x.shape
        x = x.flatten(start_dim=2,end_dim=3)
        
        x = self.linear_spa(x)
        
        if self.Apply_Norm:
            x = self.norm(x)
        if self.Apply_Act:
            x = self.act(x)
        if self.Apply_Dropout:
            x = self.dropout(x)
         
        x = x.view(b,c,h,w)
        x = self.lin_chan(x)
        return x
    
    
    
class SSM_spa(nn.Module):
    def __init__(self, sp_dim1,sp_dim2, Norm_Apply=True):
        super(SSM_spa, self).__init__()
        self.sp_dim2 = sp_dim2
        self.ssm = Mamba(
          d_model = sp_dim1*sp_dim1,
          out_c = sp_dim2*sp_dim2,
          d_state = D_state,  
          expand=Expand)
        self.norm = nn.LayerNorm(normalized_shape=sp_dim2*sp_dim2)
        self.Norm_Apply = Norm_Apply
    def forward(self, x):
        b,c,h,w = x.shape
        x = x.flatten(start_dim=2,end_dim=3) ##  [B,C,H*W]   

        x = self.ssm(x)
        if self.Norm_Apply:
            x = self.norm(x)
          
        x = x.view(b,c,self.sp_dim2,self.sp_dim2) 
        return x
    
class SSM_cha(nn.Module):
    def __init__(self, n_channels,out_channels, Norm_Apply=True):
        super(SSM_cha, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.ssm = Mamba(
          d_model = self.n_channels,
          out_c = self.out_channels,
          d_state=D_state,  
          expand=Expand,
      )
        self.norm = nn.LayerNorm(normalized_shape=self.out_channels)
        self.Norm_Apply = Norm_Apply

    def forward(self, x):
    
        b,c,h,w = x.shape        
        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # [B, H*W, C]
        x = self.ssm(x)
        
        if self.Norm_Apply:
            x = self.norm(x)
            
        x = x.view(b, h, w, self.out_channels).permute(0, 3, 1, 2)
        return x
    

class MSA(nn.Module): ## Branch_3 --> MSA
    def __init__(self,in_channels, out_channels,sp_dim1,sp_dim2, Apply_Act=True,Apply_Dropout=False):
        super().__init__()
        self.LIFM_SPA = nn.Sequential(
            SSM_spa(sp_dim1,sp_dim2),
            Linear_Layer(in_channels, out_channels),
            SSM_spa(sp_dim2,sp_dim2),
        )
        self.linear_res = Linear_Layer_SP_Res(sp_dim1, sp_dim2,in_channels, out_channels)
        
        self.Apply_Act = Apply_Act
        self.Apply_Dropout = Apply_Dropout
        self.dropout = nn.Dropout(p=drop_rate)
        self.act = nn.SiLU()
        
    def forward(self, x):
        x = self.LIFM_SPA(x) + self.linear_res(x)
        if self.Apply_Act:
            x = self.act(x)
        if self.Apply_Dropout:
            x = self.dropout(x)
        return x
    
    
class MCA(nn.Module): # Branch12--> MCA
    def __init__(self, in_channels, out_channels, Apply_Act=True,Apply_Dropout=False):
        super().__init__()
        self.linear_res = Linear_Layer(in_channels, out_channels)
        self.LIFM_CH = nn.Sequential(
            SSM_cha(in_channels, out_channels),
            Linear_Layer(out_channels, out_channels),
            SSM_cha(out_channels, out_channels), 
        )
        self.Apply_Act = Apply_Act
        self.Apply_Dropout = Apply_Dropout
        self.dropout = nn.Dropout(p=drop_rate)
        self.act = nn.SiLU()
        
   
    def forward(self, x):
        x = self.linear_res(x)+self.LIFM_CH(x)
        if self.Apply_Act:
            x = self.act(x)
        if self.Apply_Dropout:
            x = self.dropout(x)
        return x
    
class CS_IF(nn.Module):
    def __init__(self, in_channels, out_channels, sp_dim1,sp_dim2, Apply_Act=True,Apply_Dropout=False):
        super().__init__()
        self.mca = MCA(in_channels, out_channels)
        self.msa = MSA(in_channels, out_channels, sp_dim1,sp_dim2)
        
        self.Apply_Act = Apply_Act
        self.Apply_Dropout = Apply_Dropout
        self.dropout = nn.Dropout(p=drop_rate)
        self.act = nn.SiLU()
           
    def forward(self, x):
        x = self.mca(x)+self.msa(x)
        if self.Apply_Act:
            x = self.act(x)
        if self.Apply_Dropout:
            x = self.dropout(x)
        return x
    


class Down(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.AvgPool2d(2),
            MCA(in_channels, out_channels)
        )
    def forward(self, x):
        return self.down(x)
    
class Down_1(nn.Module):
    def __init__(self, in_channels, out_channels, sp_dim1,sp_dim2):
        super().__init__()
        self.down_1 = nn.Sequential(
            nn.AvgPool2d(2),
            CS_IF(in_channels, out_channels, sp_dim1,sp_dim2)
        )
    def forward(self, x):
        return self.down_1(x)
    
class Up_1(nn.Module):
    def __init__(self, in_channels, out_channels,sp_dim1,sp_dim2):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cs_if = CS_IF(in_channels+in_channels//2, out_channels,sp_dim1,sp_dim2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.cs_if(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels,last=None):
        super().__init__()
        
        self.last = last
        if self.last is None:
            in_channels = in_channels+in_channels//2
        if self.last is not None:
            in_channels = in_channels + in_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.mca = MCA(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.last is not None:
            x2 = self.up(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.mca(x)
    

class MCA_Last(nn.Module): # Branch12--> MCA
    def __init__(self, in_channels, out_channels, Apply_Act = False,Apply_Dropout = False):
        super().__init__()
        self.linear_res = Linear_Layer(in_channels, out_channels)
        self.LIFM_CH = nn.Sequential(
            SSM_cha(in_channels, out_channels),
            Linear_Layer(out_channels, out_channels, Apply_Norm=False,Apply_Act=False, Apply_Dropout = False, bias=False),
            SSM_cha(out_channels, out_channels), 
        )
        self.Apply_Act = Apply_Act
        self.Apply_Dropout = Apply_Dropout
        self.dropout = nn.Dropout(p=drop_rate)
        self.act = nn.SiLU()
        
    def forward(self, x):
        x = self.linear_res(x)+self.LIFM_CH(x)
        if self.Apply_Act:
            x = self.act(x)
        if self.Apply_Dropout:
            x = self.dropout(x)
        return x
    
#BN1 = 16  
#BN2 = 8   
#image_size =  256
#Num_Classes = 4

BN1 = 10
BN2 = 5
image_size =  160
Num_Classes = 5


Base = 64
from timm.models.layers import to_2tuple
class PatchEmbed(nn.Module): # [2,1,160,160] -->[2,1600,96]
    def __init__(self, img_size=image_size, patch_size=2, in_chans=3, embed_dim=Base, Apply_Norm=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
            
        self.norm = nn.LayerNorm(embed_dim)
        self.Apply_Norm = Apply_Norm

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.Apply_Norm:
            x = self.norm(x)
        x = x.transpose(1, 2).view(B, self.embed_dim, H//2, W//2) 
        return x

class UNet(nn.Module):
    def __init__(self, n_channels=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        #self.pm = PatchMerging(3)
        self.pm = PatchEmbed()
        self.inc = MCA(Base,Base)
        self.down1 = Down(Base,2*Base)
        self.down2 = Down(2*Base,4*Base)
        self.down3 = Down_1(4*Base,8*Base,BN1,BN1)
        self.down4 = Down_1(8*Base,16*Base,BN2,BN2)
        self.up1 = Up_1(16*Base,8*Base,BN1,BN1)
        
        self.up2 = Up(8*Base,4*Base)
        self.up3 = Up(4*Base,2*Base)
        self.up4 = Up(2*Base,Base)
        self.up5 = Up(Base,Base,last='yes')
        self.outc = MCA_Last(Base, Num_Classes)
        #self.outc = nn.Conv2d(Base, Num_Classes, kernel_size=1)
        
        self.act = nn.SiLU()
        
        self.pos_embed =  positionalencoding2d(Base,image_size//2,image_size//2).to(DEVICE)
        
        self.apply_act = True
        
        ### also check adding pos-embd before first birecdio

    def forward(self, inp):
        
        inp = inp.repeat(1,3,1,1)
        inp = self.pm(inp)
        
        ## Adding Positional Embeddings Here ###
        inp = inp  + self.pos_embed
        
        x1_12 = self.inc(inp)   
        x1_98 = flip_Dim(inp,'98')
        x1_98 = self.inc(x1_98) 
        x1_98 = flip_Dim_back(x1_98,'98')
        
        x1 = x1_12 + x1_98
        if self.apply_act:
            x1 = self.act(x1)
                
        x2_12 = self.down1(x1)
        x2_98 = flip_Dim(x1,'98')
        x2_98 = self.down1(x2_98) 
        x2_98 = flip_Dim_back(x2_98,'98')
        
        x2 = x2_12 + x2_98
        if self.apply_act:
            x2 = self.act(x2)
        
        x3_12 = self.down2(x2)
        x3_98 = flip_Dim(x2,'98')
        x3_98 = self.down2(x3_98) 
        x3_98 = flip_Dim_back(x3_98,'98')
        
        x3 = x3_12 + x3_98
        if self.apply_act:
            x3 = self.act(x3)
        
        x4_12 = self.down3(x3)
        x4_98 = flip_Dim(x3,'98')
        x4_98 = self.down3(x4_98) 
        x4_98 = flip_Dim_back(x4_98,'98')
        
        x4 = x4_12 + x4_98
        if self.apply_act:
            x4 = self.act(x4)
        
        x5_12 = self.down4(x4)
        x5_98 = flip_Dim(x4,'98')
        x5_98 = self.down4(x5_98) 
        x5_98 = flip_Dim_back(x5_98,'98')
        
        x5 = x5_12 + x5_98
        if self.apply_act:
            x5 = self.act(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, inp)
        x = self.outc(x)
        return x
        
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#def model() -> UNet:
#    model = UNet()
#    model.to(device=DEVICE,dtype=torch.float)
#    return model
#from torchsummary import summary
#model = model()
#summary(model, [(1, 160,160)])
