import timm
import torch
from torch import nn
model = timm.create_model('resnet50d', pretrained=False, num_classes=10, global_pool='catavgmax')

# From the existing classifier, we can get the number of input features:
num_in_features = model.get_classifier().in_features; num_in_features
print(num_in_features)

# we can replace the final layer with our modified classification head by accessing the classifier directly.
# Here, the classfication head has been chosen somewhat arbitrarily.
model.fc = nn.Sequential(
    nn.BatchNorm1d(num_in_features),
    nn.Linear(in_features=num_in_features, out_features=512, bias=False),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(in_features=512, out_features=10, bias=False))

# Testing the model with a dummy input, we get an output of the expected shape.
# Now, our modified model is ready to train!

model.eval()
print(model(torch.randn(1, 3, 224, 224)).shape)
