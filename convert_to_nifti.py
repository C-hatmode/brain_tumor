import os
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Define input and output directories
input_dir = "./tumor_category/Training"   # change if needed
output_dir = "./tumor_category_nifti"

# Create output folders
os.makedirs(output_dir, exist_ok=True)

# Loop through all subfolders (glioma, meningioma, etc.)
for category in os.listdir(input_dir):
    category_path = os.path.join(input_dir, category)
    if not os.path.isdir(category_path):
        continue

    output_category_path = os.path.join(output_dir, category)
    os.makedirs(output_category_path, exist_ok=True)

    print(f"\nConverting {category} images to NIfTI format...")
    for img_name in tqdm(os.listdir(category_path)):
        img_path = os.path.join(category_path, img_name)
        try:
            # Read image and convert to grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Add an extra axis to make it 3D (as NIfTI expects volumetric data)
            img_3d = np.expand_dims(img, axis=-1)

            # Create a NIfTI image
            nifti_img = nib.Nifti1Image(img_3d, affine=np.eye(4))

            # Save as .nii file
            base_name = os.path.splitext(img_name)[0]
            output_path = os.path.join(output_category_path, f"{base_name}.nii")
            nib.save(nifti_img, output_path)
        except Exception as e:
            print(f"Error converting {img_name}: {e}")
