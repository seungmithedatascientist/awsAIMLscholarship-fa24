from PIL import Image
import numpy as np
import torch

def process_image(image_path):
    """Process an image path into a PyTorch-compatible tensor for model inference."""
    
    image = Image.open(image_path)

    # Resize: Maintain aspect ratio, shortest side = 256px
    aspect_ratio = image.width / image.height
    if image.width < image.height:
        image = image.resize((256, int(256 / aspect_ratio)))
    else:
        image = image.resize((int(256 * aspect_ratio), 256))

    # Center crop to 224x224
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))

    # Convert to NumPy and normalize
    np_image = np.array(image) / 255.0 

    # Reorder dimensions to (channels, height, width)
    np_image = np_image.transpose((2, 0, 1))

    # Normalize using ImageNet mean & std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(np_image).float()

    return image_tensor
