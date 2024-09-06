from UNet import UNet
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_ = UNet()
model = model_.to(device=DEVICE,dtype=torch.float)

dummy_input = torch.randn(1, 1,160,160,dtype=torch.float).to(DEVICE)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 500
timings = np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
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
print(mean_syn)
print(std_syn)
