import numpy as np
import cv2
from typing import List, Tuple, Optional
import os
import matplotlib.pyplot as plt

def _compute_single_channel_gradient(channel: np.ndarray, method: str) -> np.ndarray:
    """
    Calcule le gradient pour un canal unique.
    """
    # Normalisation
    channel = channel.astype(np.float32) / 255.0
    
    if method == 'sobel':
        grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    elif method == 'scharr':
        grad_x = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    elif method == 'laplacian':
        gradient_magnitude = cv2.Laplacian(channel, cv2.CV_32F)
        gradient_magnitude = np.abs(gradient_magnitude)
    
    else:
        raise ValueError("Method must be 'sobel', 'scharr', or 'laplacian'")
    
    return gradient_magnitude

def get_gradient(image: np.ndarray, ker=5, method: str = 'sobel') -> np.ndarray:
    ker = 2*ker+1
    image_blur = cv2.GaussianBlur(image, (ker, ker), 0)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_blur
    gradient = _compute_single_channel_gradient(gray, method)
    vmin = np.min(gradient)
    vmax = np.max(gradient)
    gradient = (gradient - vmin) / (vmax - vmin) * 255.0
    gradient = gradient.astype(np.uint8)
    return gradient,image_blur

def get_sorted_gradient_patches(img: np.ndarray, patch_size: int, stride: int, ker=5):
    gradient_img = get_gradient(img, ker=ker)[0]
    h, w = gradient_img.shape
    gradient_patches = []
    original_patches = []
    sums = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            grad_patch = gradient_img[y:y+patch_size, x:x+patch_size]
            orig_patch = img[y:y+patch_size, x:x+patch_size]
            gradient_patches.append(grad_patch)
            original_patches.append(orig_patch)
            sums.append(np.sum(grad_patch))

    # Sort patches by descending gradient sum
    sorted_indices = np.argsort(sums)[::-1]
    gradient_patches_sorted = np.array([gradient_patches[i] for i in sorted_indices])
    original_patches_sorted = np.array([original_patches[i] for i in sorted_indices])

    return gradient_patches_sorted, original_patches_sorted

if __name__ == "__main__":
    patch_size = 64
    stride = patch_size // 2

    data_dir = "Data"
    output_dir = "data_contour_patch"
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png")

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(root, file)
                print(f"Found image: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read image: {image_path}")
                    continue
                
                print(f"Image min: {image.min()}, max: {image.max()}")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Subfolder for each image's top patches
                base_name = os.path.splitext(os.path.basename(file))[0]
                subfolder = os.path.join(output_dir, base_name)
                os.makedirs(subfolder, exist_ok=True)

                
                
                top_patches, blurs = get_sorted_gradient_patches(image, patch_size, stride, ker=10)

                n_top = int(len(top_patches) * 0.2)
                top_patches = top_patches[:n_top]

                # 4. Save patches
                for i, patch in enumerate(top_patches):
                    patch_filename = f"{base_name}_top_{i}.jpg"
                    patch_path = os.path.join(subfolder, patch_filename)
                    cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    

                

