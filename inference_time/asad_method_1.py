from UNet import UNet
import torch
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_ = UNet()
model = model_.to(device=DEVICE,dtype=torch.float)

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
        
def evaluate_timing(model, input_shape, warmup_epochs=50, eval_epochs=200):
    execution_time = AverageMeter()
    model.eval()
    # warm-up
    with torch.no_grad():
        for idx in range(warmup_epochs + eval_epochs):
            data_lowres = torch.rand(input_shape, dtype=torch.float32).to(DEVICE)
            tic = time.time()
            output = model(data_lowres)
            toc = time.time()

            if idx > warmup_epochs:
                execution_time.update(toc-tic, data_lowres.shape[0])


    print("Execution: {}".format(execution_time.avg * 1000))

    avg_time = execution_time.avg * 1000
    return avg_time

eval_time = evaluate_timing(model, input_shape=(1,1,160,160), warmup_epochs=10, eval_epochs=100)
