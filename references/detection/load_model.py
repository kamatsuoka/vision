import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_model(filename: str):
    """Load a model from disk"""
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # TODO: change to more obvious name
#    model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model'])

    args = checkpoint['args']
    epoch = checkpoint['epoch']
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # TODO: change to more obvious name
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return [model, optimizer, epoch]
