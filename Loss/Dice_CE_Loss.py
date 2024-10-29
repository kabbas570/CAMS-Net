import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure the weight tensor matches the number of classes
W_1 = torch.tensor([1, 1, 1, 1, 2], device=DEVICE, dtype=torch.float)

def to_one_hot(targets, num_classes):
    batch_size, height, width = targets.size()
    one_hot = torch.zeros(batch_size, num_classes, height, width, device=DEVICE)
    return one_hot.scatter_(1, targets.unsqueeze(1), 1)

class DiceCELoss(nn.Module):
    def __init__(self, weight=W_1, size_average=True):
        super(DiceCELoss, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets, smooth=1):
        num_classes = inputs.size(1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = to_one_hot(targets, num_classes)
        
        # Apply softmax to inputs to get probabilities for Dice loss calculation
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Dice Loss with class weights
        dice_loss = 0
        for i in range(num_classes):
            input_flat = inputs_softmax[:, i].contiguous().view(-1)
            target_flat = targets_one_hot[:, i].contiguous().view(-1)
            intersection = (input_flat * target_flat).sum()
            # Incorporate class weights into Dice loss calculation
            dice_loss += self.weight[i] * (1 - ((2. * intersection + smooth) / 
                                                (input_flat.sum() + target_flat.sum() + smooth)))
        dice_loss /= self.weight.sum()  # Normalize by sum of weights
        
        # Cross-Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight)
        
        # Combine losses
        loss = dice_loss + ce_loss
        
        return loss.mean() if self.size_average else loss.sum()

# Example usage
# Assuming inputs and targets are defined elsewhere with appropriate shapes
# loss_fn = DiceCELoss()
# loss = loss_fn(inputs, targets)
