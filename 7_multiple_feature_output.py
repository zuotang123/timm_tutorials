import numpy as np
import timm
import torch
from PIL import Image
from matplotlib import pyplot as plt

"""Create model"""
model = timm.create_model('resnet50d', pretrained=False, features_only=True)

"""We can get more information about the features that are returned, such as the specific module names, the reduction 
in features and the number of channels: """
print(model.feature_info.module_name())

"""Feature reduction"""
print(model.feature_info.reduction())

"""Feature channels"""
print(model.feature_info.channels())

"""Now, lets pass an image through our feature extractor and explore the output."""
pets_image_paths = './download.png'
image = Image.open(pets_image_paths)
np_image = np.array(image, dtype=np.float32)
image = torch.as_tensor(np_image).transpose(2, 0)[None]
out = model(image)

print(len(out))

"""As expected, 5 feature maps have been returned. Inspecting the shape, we can see that the number of channels is 
consistent with what we expect: """
for o in out:
    print(o.shape)

"""Visualising each feature map, we can see that the image is gradually downsampled, as we would expect."""
for o in out:
    plt.imshow(o[0].transpose(0, 2).sum(-1).detach().numpy())
    plt.show()




