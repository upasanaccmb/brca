#This code is used to preprocess the histology whole slide images with tiling, normalization and augmentation.


import numpy as np
import openslide
import cv2
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, 
    RandomBrightnessContrast, GaussNoise
)

def preprocess_histology_slides(slide_path, output_path, patch_size=256, 
                              magnification_level=0, augment=True):
    """
    Preprocess histology whole slide images with tiling, normalization and augmentation.
    """
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    
    def extract_patches(slide, coords):
        """Extract tiles at given coordinates"""
        patches = []
        for x, y in coords:
            patch = slide.read_region(
                (x, y), 
                magnification_level,
                (patch_size, patch_size)
            ).convert('RGB')
            patches.append(np.array(patch))
        return np.array(patches)
    
    def filter_background(patches, threshold=0.8):
        """Remove background tiles"""
        filtered = []
        for patch in patches:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            if np.mean(gray > 240) < threshold:
                filtered.append(patch)
        return np.array(filtered)
    
    def normalize_staining(patches):
        """Normalize H&E staining"""
        normalized = []
        for patch in patches:
            lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
            lab_scaled = StandardScaler().fit_transform(
                lab.reshape(-1, 3)
            ).reshape(patch_size, patch_size, 3)
            normalized.append(
                cv2.cvtColor(lab_scaled.astype(np.uint8), 
                            cv2.COLOR_LAB2RGB)
            )
        return np.array(normalized)
    
    def augment_patches(patches):
        """Apply data augmentation"""
        transform = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=90, p=0.5),
            RandomBrightnessContrast(p=0.5),
            GaussNoise(p=0.3)
        ])
        
        augmented = []
        for patch in patches:
            # Original patch
            augmented.append(patch)
            # Augmented versions
            for _ in range(3):  # Create 3 augmented versions
                aug_patch = transform(image=patch)['image']
                augmented.append(aug_patch)
        return np.array(augmented)

    # Generate patch coordinates
    coords = [(x, y) 
             for x in range(0, width - patch_size, patch_size)
             for y in range(0, height - patch_size, patch_size)]
    
    # Extract patches
    patches = extract_patches(slide, coords)
    
    # Filter background
    filtered_patches = filter_background(patches)
    
    # Normalize staining
    normalized_patches = normalize_staining(filtered_patches)
    
    # Apply augmentation if requested
    if augment:
        final_patches = augment_patches(normalized_patches)
    else:
        final_patches = normalized_patches
    
    # Save processed patches
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, patch in enumerate(final_patches):
        cv2.imwrite(str(output_path / f'patch_{i}.png'), 
                   cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    
    return final_patches