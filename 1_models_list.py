from pprint import pprint
import timm

# We can list, and query, the collection available models as demonstrated below:
print("Total models in timm:", len(timm.list_models('*')))

# We can also use the pretrained argument to filter this selection to the models with pretrained weights:
print("Total pretrained models in timm:", len(timm.list_models(pretrained=True)))

# We can list the different ResNet variants available by providing a wildcard string
print("ResNet family in timm:", timm.list_models('resnet*', pretrained=True))
