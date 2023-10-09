import os
import shutil
from PIL import Image

def rgb_to_gray(src_path, dst_path):
    # Create the target folder if it doesn't exist
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    # Iterate through the sub-folders in source folder
    for subfolder in os.listdir(src_path):
        subfolder_path = os.path.join(src_path, subfolder)
        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # Create the corresponding subfolder in target folder
            dst_subfolder_path = os.path.join(dst_path, subfolder)
            if not os.path.exists(dst_subfolder_path):
                os.mkdir(dst_subfolder_path)

            # Iterate through the images in the subfolder
            for image_file in os.listdir(subfolder_path):
                # Open the image
                img = Image.open(os.path.join(subfolder_path, image_file))
                # Convert the image to grayscale
                img = img.convert("L")
                # Get the name of the image file and change the extension to jpeg
                file_name, file_ext = os.path.splitext(image_file)
                new_file_name = file_name + ".jpeg"
                # Save the image in the corresponding subfolder in target folder
                img.save(os.path.join(dst_subfolder_path, new_file_name), "JPEG")