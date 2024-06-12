import torch
import torch.nn as nn

class ResConvBlock(nn.Module):
    def __init__(self, input_features, output_features, dropout, kernel_size=3 ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(output_features, output_features, kernel_size, padding='same')
        self.batch_norm = nn.BatchNorm2d(output_features)
        
        self.res_conv = nn.Conv2d(input_features, output_features, 1, padding='same')
        self.res_batch_norm = nn.BatchNorm2d(output_features)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.con_batch_norm = nn.BatchNorm2d(output_features)
        
    def forward(self, x):
        res_x = x
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        
        res_x = self.res_conv(res_x)
        x = self.relu(x)
        res_x = self.res_batch_norm(res_x)
        
        x = torch.add(x, res_x)
        x = self.con_batch_norm(x)
        
        del res_x
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, input_features, output_features, dropout, kernel_size=3):
        super().__init__()
        self.conv_block = ResConvBlock(input_features, output_features, dropout, kernel_size)
        self.downsampler = nn.MaxPool2d((2, 2))
    
    def forward(self, x):
        c = self.conv_block(x)
        x = self.downsampler(c)
        
        return x, c
    
class Decoder(nn.Module):
    def __init__(self, input_features, output_features, dropout, kernel_size=3):
        super().__init__()
        self.upsampler = nn.ConvTranspose2d(input_features, input_features, 2, stride=2)
        self.conv_block = ResConvBlock(2*input_features, output_features, dropout, kernel_size)
   
    def forward(self, x, c):
        x = self.upsampler(x)
        x = torch.concat([x, c], dim=1)
        x = self.conv_block(x)
        
        return x
    
stages = 4

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([Encoder(1, 64, 0.1)]+[Encoder(64*(2**i), 64*(2**(i+1)), 0.1*i) for i in range(stages)])
        
        self.pointwise1 = nn.Conv2d(64*2**stages, 2**(stages+1)*64, 1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(2**(stages+1)*64)
        
        self.pointwise2 = nn.Conv2d(2**(stages+1)*64, 64*2**(stages), 1)
        self.relu = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm2d(64*2**(stages))
        
        self.decoders = nn.ModuleList([Decoder(64*(2**(stages-i)), 64*(2**((stages-1)-i)), 0.1*(stages-i)) for i in range(stages)]+[Decoder(64, 5, 0.1)])
                
    def forward(self, x):
        conv_outputs = []
        
        for layer in self.encoders:
            x , c = layer(x)
            conv_outputs.append(c)
                    
        x = self.pointwise1(x)
        x = self.relu(x)
        x = self.batch_norm1(x)
        
        x = self.pointwise2(x)
        x = self.relu(x)
        x = self.batch_norm2(x)
        
        for i, layer in enumerate(self.decoders):
            c = conv_outputs[stages - i]
            x = layer(x, c)
        
        del conv_outputs
        
        return x

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# def model() -> UNet:
#     model = UNet()
#     model.to(device=DEVICE,dtype=torch.float)
#     return model
# from torchsummary import summary
# model = model()
# summary(model, [(1, 160,160)])
