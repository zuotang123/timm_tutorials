import numpy as np
import torch
from PIL import Image
from timm.data.transforms_factory import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation

tfm = RandomResizedCropAndInterpolation(size=350, interpolation='random')
import matplotlib.pyplot as plt

a = create_transform(224, is_training=True)
print(a)

pets_image_paths = './download.png'
image = Image.open(pets_image_paths)

# We can convert this into a tensor, and transpose the channels into the format that PyTorch expects:
# np_image = np.array(image, dtype=np.float32)
# image = torch.as_tensor(np_image).transpose(2, 0)[None]


fig, ax = plt.subplots(2, 4, figsize=(10, 5))

for idx, im in enumerate([tfm(image) for i in range(4)]):
    ax[0, idx].imshow(im)

for idx, im in enumerate([tfm(image) for i in range(4)]):
    ax[1, idx].imshow(im)

fig.tight_layout()
plt.show()
