import torch, torchvision
print ("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print ("npm gpus: ", torch.cuda.device_count())