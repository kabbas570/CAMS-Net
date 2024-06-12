#from Mamba_UNet.code.networks.vision_mamba import MambaUnet
import torch
#from Mamba_UNet.code.networks.vision_mamba import MambaUnet

#import importlib.util
#
#module_path = "/data/home/acw676/mamba_work/Mamba_UNet/code/networks/vision_mamba.py"
#spec = importlib.util.spec_from_file_location("vision_mamba", module_path)
#vision_mamba = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(vision_mamba)
#from vision_mamba import MambaUnet

import sys
# Add the directory containing your module to the Python path
sys.path.append( "/data/home/acw676/mamba_work/Mamba_UNet/code/networks/")
print(sys.path)
from vision_mamba import MambaUnet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def model() -> MambaUnet:
    model = MambaUnet()
    model.to(device=DEVICE,dtype=torch.float)
    return model
from torchsummary import summary
model = model()
summary(model, [(1, 256,256)])