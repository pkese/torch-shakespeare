

python: `pipenv shell`



Train time:
  - no optimizations:
    - F#: 6:46 ()
    - F#: 5:19-5:22-5:23 (allow_tf32)

    - Py: 4.39 (Python 3.11, Torch 2.1, Cuda 11.8)
    - Py: 4.28 (Python 3.11, Torch 2.0.1, Cuda 11.7)


Python 3.10, Torch 2.0.1, Cuda 11.7
    - 5:31 ()
    - 5:27 (allow_tf32)
    - 4.28 (autocast bfloaf16)
    - 4.04 (compile)
    - 3.57 (compile + allow_tf32)
    - 2.26 (compile + autocast bfloaf16)

Python 3.11, Torch 2.1.0, Cuda 11.8
    - 5:27 ()
    - 5:39 (script)
    - 5:24 (allow_tf32)
    - 4.41 (autocast bfloaf16)
    - 4.40 (allow_tf32 + autocast bfloaf16)
    - 4.04 (compile)
    - 3.57 (compile + allow_tf32)
    - 2.30 (compile + autocast bfloaf16)
    - 2.30 (compile + allow_tf32 + autocast bfloaf16)


I made a small language model inspired by https://github.com/karpathy/nanoGPT in both PyTorch and TorchSharp.  
The model has 2 layers of transformers totalling 150k parameters and is trained on Shakespeare's text.

I found out that going to smaller data types, improves training time, as does PyTorch's jit.compile, which is not available in TorchSharp.   

Here are some benchmarks of model training times with CUDA on a small GPU (RTX 3070).

Cuda 11.7

|                     | default | tf32 | bf16 |
| ------------------- | ----- | ---- | ---- |
| TorchSharp 0.100.7  | 6:46  | 5:20 | N/A  |
| PyTorch 2.0.1       | 5:31  | 5:27 | 4:28 |
| PyTorch+jit.compile | 4:04  | 3:57 | 2:26 |

Cuda 12.1

|                     | default | tf32 | bf16 |
| ------------------- | ----- | ---- | ---- |
| TorchSharp 0.100.7  |  4:53 | 4.51 | N/A  |
| PyTorch 2.1.0       |  4:22 | 4.19 | 3:48 |
| PyTorch+script      |  4:30 |  |  |
| PyTorch+jit.compile |  4:02 | 3:57 | 2:30 |



For `bf16` I used:
```python
from torch.cuda.amp import autocast
with autocast(dtype=torch.bfloat16):
    <train code>
```
I couldn't achieve the same functionality with TorchSharp.

    - Py: 6:10
    - Py: 5:17 (compile submodules)
    - Py: 5:13 (compile)
    - Py: 4:56 (compile + tf32)
    - Py: 4:32 (compile + amp=bf16)




flash sdp enabled = True
/home/peter/.local/share/virtualenvs/shakespere-7XL7FsfX/lib/python3.11/site-packages/torch/overrides.py:110: UserWarning: 'has_cuda' is deprecated, please use 'torch.backends.cuda.is_built()'
  torch.has_cuda,
/home/peter/.local/share/virtualenvs/shakespere-7XL7FsfX/lib/python3.11/site-packages/torch/overrides.py:111: UserWarning: 'has_cudnn' is deprecated, please use 'torch.backends.cudnn.is_available()'
  torch.has_cudnn,
/home/peter/.local/share/virtualenvs/shakespere-7XL7FsfX/lib/python3.11/site-packages/torch/overrides.py:117: UserWarning: 'has_mps' is deprecated, please use 'torch.backends.mps.is_built()'
  torch.has_mps,
/home/peter/.local/share/virtualenvs/shakespere-7XL7FsfX/lib/python3.11/site-packages/torch/overrides.py:118: UserWarning: 'has_mkldnn' is deprecated, please use 'torch.backends.mkldnn.is_available()'
  torch.has_mkldnn,
Step     0: loss=4.380359 train=4.359313 valid=4.357822 epoch=0.01