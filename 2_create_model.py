from pprint import pprint
import torch

import timm

# create timm model with resnet 50d
model = timm.create_model('resnet50d', pretrained=False)
print(model)

# To know the statistics of above model
print("Model statistics:")
pprint(model.default_cfg)

# We can specify the number of channels for our input images by passing the in_chans argument to create_model.
model = timm.create_model('resnet50d', pretrained=True, in_chans=1)

# single channel image
x = torch.randn(1, 1, 224, 224)
print(model(x).shape)


