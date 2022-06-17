from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import timm

pets_image_paths = './download.png'
image = Image.open(pets_image_paths)
print(image.size)
h, w = image.size
print(h, w)

# We can convert this into a tensor, and transpose the channels into the format that PyTorch expects:
np_image = np.array(image, dtype=np.float32)
image = torch.as_tensor(np_image).transpose(2, 0)[None]
print(image.shape)

model = timm.create_model('resnet50d', pretrained=True)
feature_output = model.forward_features(image)


# Visulize
def visualise_feature_output(t):
    plt.imshow(feature_output[0].transpose(0, 2).sum(-1).detach().numpy())
    plt.show()


visualise_feature_output(feature_output)
