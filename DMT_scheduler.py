WARMUP_EPOCH = 10
DECAY_FACTOR = 0.1
from torch.optim.lr_scheduler import LambdaLR

def warmup_schedule(epoch):
    if epoch < WARMUP_EPOCH:
        return epoch / WARMUP_EPOCH
    else:
        return 1

def decay_schedule(epoch):
    if epoch < 30: 
        return 1
    if 30 <= epoch < 40:
        return  DECAY_FACTOR 

    return DECAY_FACTOR ** 2 

def combined_schedule(epoch):
    return warmup_schedule(epoch) * decay_schedule(epoch)


def get_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=combined_schedule)
