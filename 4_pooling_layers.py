from pprint import pprint
import torch

import timm

# create timm model with resnet 50d and view the global pooling layers
'''' If we would like to avoid creating the last layer completely, we can set the number of classes equal to 0, 
which will create a model with the identity function as the final layer; this can be useful for inspecting the output 
of penultimate layer. '''
model = timm.create_model('resnet50d', pretrained=False,num_classes=0)
print(model.global_pool)

'''An instance of SelectAdaptivePool2d, which is a custom layer provided by timm, 
which supports different pooling and flattening configurations: 
avg: Average pooling
max: Max pooling
avgmax: the sum of average and max pooling, re-scaled by 0.5
catavgmax: a concatenation of the outputs of average and max pooling along feature dimension. Note that this will double the feature dimension.
'': No pooling is used, the pooling layer is replaced by an Identity operation
We can visualise the output shapes of the different pooling options as demonstrated below
'''

# We can visualise the output shapes of the different pooling options as demonstrated below
pool_types = ['avg', 'max', 'avgmax', 'catavgmax', '']

for pool in pool_types:
    model = timm.create_model('resnet50d', pretrained=False, num_classes=0, global_pool=pool)
    model.eval()
    feature_output = model(torch.randn(1, 3, 224, 224))
    print(feature_output.shape)

# Modifying an existing model
# We can also modify the classifier and pooling layers of an existing model, using the reset_classifier method:
model = timm.create_model('resnet50d', pretrained=False)

print(f'Original pooling: {model.global_pool}')
print(f'Original classifier: {model.get_classifier()}')
print('--------------------')

model.reset_classifier(10, 'max')

print(f'Modified pooling: {model.global_pool}')
print(f'Modified classifier: {model.get_classifier()}')

