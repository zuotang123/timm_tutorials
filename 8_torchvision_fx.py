import timm
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

model = timm.create_model('resnet50d', pretrained=True, exportable=True)

nodes, _ = get_graph_node_names(model)

print(nodes)
