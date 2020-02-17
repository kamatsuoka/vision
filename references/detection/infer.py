#!/usr/bin/env python

from torchvision.transforms import functional as F
from PIL import Image
from load_model import load_model
import sys


if __name__ == '__main__':
    model_file = sys.argv[1]
    image_files = sys.argv[2:]
    model, _, _ = load_model(model_file)
    model.eval()
    images = [F.to_tensor(Image.open(img_file).convert('RGB')) for img_file in image_files]
    infs = model(images)
    good_boxes = []
    for i in range(len(image_files)):
        boxes = infs[i]['boxes'].tolist()
        scores = infs[i]['scores'].tolist()
        for j in range(len(scores)):
            if scores[j] > 0.9:
                x = boxes[j]
                print(f'rectangle {int(x[0])},{int(x[1])} {int(x[2])},{int(x[3])}')
