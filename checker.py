from datasets import load_dataset
ds = load_dataset("Teklia/IAM-line", split="test")
sample = ds[0]
img = sample["test"]
print(type(img), getattr(img, "mode", None), getattr(img, "size", None))

import numpy as np
a = np.array(img)
print(a.shape, a.dtype, a.min(), a.max(), a.mean())
