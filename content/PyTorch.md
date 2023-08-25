## CUDA

验证是否支持CUDA

```Python
import torch

print("CUDA:", torch.cuda.is_available())
```

如果输出为 CUDA：True ，则表明支持 CUDA 加速

如果输出为 CUDA：False，则表明不支持 CUDA 加速，需要重新进行 torch 环境的配置和修改，直到出现 CUDA：True 为止