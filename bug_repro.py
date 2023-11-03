import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device={device}")

cfg = {
    "vocabSize": 65,
    "blockSize": 32,
    "batchSize": 4,
    "encodingSize": 80,
    "nHeads": 4,
    "dropout": 0.1,
    "bias": False
}

model = torch.jit.load("shakespeare.pt.zip").to(device)

print("Model parameters:")
for name, param in model.named_parameters():
    print(f"{name.ljust(36)}: shape={param.shape} device={param.device}")

xs = torch.tensor([random.randint(0, cfg["vocabSize"] - 1) for _ in range(cfg["blockSize"])], device=device, dtype=torch.int64).unsqueeze(0)

ys = model(xs)

print(f"shapes: xs={xs.shape} ys={ys.shape}")