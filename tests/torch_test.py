import torch

a = torch.zeros(10, 21)

a[0] = torch.arange(0, 0.21, 0.01)
a[3] = 1.0

a = a.unsqueeze(0).repeat(5, 1, 1)

b = torch.zeros(5, 10)
b[0] = 0.5
b[2] = 1.0
b[3] = 1.0

c = a.clone()

# Add only the first three indices (0, 1, 2) of the second dimension
c[:, :3, :] += b[:, :3].unsqueeze(-1)

print(c)  # Output: torch.Size([5, 10, 21])


import numpy as np

# Initialize the array
a = np.zeros((10, 21))

# Assign values to the first and fourth rows
a[0] = np.arange(0, 0.21, 0.01)
a[3] = 1.0

# Replicate the array 5 times along a new axis
a = np.expand_dims(a, axis=0).repeat(5, axis=0)

print(a)  # Output: (5, 10, 21)