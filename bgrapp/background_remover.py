import torch
import torchvision.transforms as T
import cv2
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
import os

os.environ['TORCH_HOME'] = 'D:/DyroDev/bgremover_project/cache/torch'

def load_model():
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    return model

def remove_background(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    print(f"Image loaded: {image.shape}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform the image
    transform = T.Compose([T.ToTensor()])
    try:
        image_tensor = transform(image_rgb).unsqueeze(0)
        print(f"Image transformed: {image_tensor.shape}")
    except Exception as e:
        print(f"Error during transformation: {e}")
        raise

    # Perform object detection
    with torch.no_grad():
        outputs = model(image_tensor)
        print(f"Model outputs: {outputs}")

    # Process the results
    masks = outputs[0]['masks']
    scores = outputs[0]['scores']
    threshold = 0.5
    mask = masks[scores > threshold][0]  # Select the first mask with score > threshold
    mask = mask.squeeze().mul(255).byte().cpu().numpy()  # Ensure mask is 2-dimensional
    print(f"Mask processed: {mask.shape}")

    # Create an alpha channel based on the mask
    alpha = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    alpha[mask > 127] = 255  # Set alpha channel values based on mask

    # Merge the image and the alpha channel
    rgba = cv2.merge((image_rgb, alpha))
    return rgba




# import torch
# import torchvision.transforms as T
# import cv2
# import numpy as np
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
# import os
# os.environ['TORCH_HOME'] = 'D:/DyroDev/bgremover_project/cache/torch'
# def load_model():
#     model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
#     model.eval()
#     return model

# def remove_background(image_path, model):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image not found or unable to load: {image_path}")
#     print(f"Image loaded: {image.shape}")

#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Transform the image
#     transform = T.Compose([T.ToTensor()])
#     try:
#         image_tensor = transform(image_rgb).unsqueeze(0)
#         print(f"Image transformed: {image_tensor.shape}")
#     except Exception as e:
#         print(f"Error during transformation: {e}")
#         raise

#     # Perform object detection
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         print(f"Model outputs: {outputs}")

#     # Process the results
#     masks = outputs[0]['masks']
#     scores = outputs[0]['scores']
#     threshold = 0.5
#     mask = masks[scores > threshold][0]
#     mask = mask.mul(255).byte().cpu().numpy()
#     print(f"Mask processed: {mask.shape}")

#     # Create an alpha channel based on the mask
#     alpha = np.zeros_like(image_rgb[:, :, 0])
#     alpha[mask > 127] = 255

#     # Merge the image and the alpha channel
#     rgba = cv2.merge((image_rgb, alpha))
#     return rgba
