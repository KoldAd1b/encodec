import torch
from accelerate import Accelerator

accelerator = Accelerator()
rand = torch.randn(2,).to(accelerator.device)

torch.cuda.set_device(accelerator.device)

accelerator.print("BEFORE")
print(accelerator.device, rand)
accelerator.wait_for_everyone()

accelerator.print("AFTER")
torch.distributed.broadcast(rand.data, src=0)
print(accelerator.device, rand)

accelerator.end_training()
