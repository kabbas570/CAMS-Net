from UNet import UNet
import torch
import time
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_ = UNet()
model = model_.to(device=DEVICE,dtype=torch.float)

warm_up = 100
epochs = 10000
spatial_dim = 1024
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def evaluate_timing(model, warmup_epochs, eval_epochs):
    execution_time = AverageMeter()
    model.eval()
    # warm-up
    with torch.no_grad():
        for idx in range(warmup_epochs + eval_epochs):
            data_lowres = torch.randn(1,1,spatial_dim,spatial_dim,dtype=torch.float).to(DEVICE)
            tic = time.time()
            _ = model(data_lowres)
            toc = time.time()

            if idx > warmup_epochs:
                execution_time.update(toc-tic, data_lowres.shape[0])


    print("Execution: {}".format(execution_time.avg * 1000))

    avg_time = execution_time.avg*1000
    return avg_time

eval_time = evaluate_timing(model,warm_up,epochs)


dummy_input = torch.randn(1,1,spatial_dim,spatial_dim,dtype=torch.float).to(DEVICE)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = epochs
timings = np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(warm_up):
   _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     starter.record()
     _ = model(dummy_input)
     ender.record()
     # WAIT FOR GPU SYNC
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)
     timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print('my method -->', mean_syn)
