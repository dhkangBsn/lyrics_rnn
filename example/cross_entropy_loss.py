import torch
import torch.nn as nn
import numpy as np
output = torch.Tensor(
    [
        [0.8982, 0.805, 0.6393, 0.9983, 0.5731, 0.0469, 0.556, 0.1476, 0.8404, 0.5544],
        [0.9457, 0.0195, 0.9846, 0.3231, 0.1605, 0.3143, 0.9508, 0.2762, 0.7276, 0.4332]
    ]
)
target = torch.LongTensor([1, 5])
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
print(loss) # tensor(2.3519)