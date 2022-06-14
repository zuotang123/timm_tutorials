from pprint import pprint
import torch

import timm

# create timm model with resnet 50d and view the classification head FC
model = timm.create_model('resnet50d', pretrained=False)
print("model.fc:", model.fc)

# However, this model.fc name is likely to change depending on the model architecture used. timm models have the
# get_classifier method, which we can use to retrieve the classification head without having to lookup the module name.
print("model.get_classifier():", model.get_classifier())

# we can see that the final layer outputs 1000 classes. We can change this with the num_classes argument
model_class_change = timm.create_model('resnet50d', pretrained=False, num_classes=10).get_classifier()
print("Changes in model classes/out_features:", model_class_change)




