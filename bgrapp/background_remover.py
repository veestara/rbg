import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
import os
import cv2
import numpy as np
from PIL import Image

os.environ['TORCH_HOME'] = 'D:/DyroDev/bgremover_project/cache/torch'

def load_model():
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    return model

def remove_background(image_path, model, threshold=0.1):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    print(f"Image loaded: {np.array(image).shape}")

    # Define the transformation
    transform = T.Compose([T.ToTensor()])

    # Transform the image
    transformed_image = transform(image).unsqueeze(0)
    print(f"Image transformed: {transformed_image.shape}")

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    transformed_image = transformed_image.to(device)

    # Get the model outputs
    with torch.no_grad():
        outputs = model(transformed_image)

    # Extract the masks, boxes, and scores
    masks = outputs[0]['masks'].cpu()
    scores = outputs[0]['scores'].cpu()
    print(f"Model outputs: {outputs}")

    # Check if there are masks with scores above the threshold
    if scores[scores > threshold].size(0) == 0:
        raise ValueError("No masks with scores above the threshold were found.")

    # Select the first mask with score > threshold
    mask = masks[scores > threshold][0]

    # Convert the mask to a binary mask
    binary_mask = mask.squeeze().numpy() > 0.5

    # Apply the mask to the image
    np_image = np.array(image)
    np_image[~binary_mask] = 0

    # Convert back to PIL Image
    result_image = Image.fromarray(np_image)

    return result_image
