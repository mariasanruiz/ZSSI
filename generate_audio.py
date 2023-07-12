import numpy as np
from audioLDM import generate_audio
from CoCa import compute_coca
import os

# AUDIO GENERATION PIPELINE (images -> captions -> audios)

# def compute_coca(image, n_captions) -> np.array:
#     pass
#
# def generate_audio(caption) -> np.array:
#     pass

# COMPUTE COCA
while True:
    obj_image_folder = input("Enter the reference image folder path here: ")
    # Check if the path exists
    if os.path.exists(obj_image_folder):
        print("Path exists!")
        break
    else:
        print("Path does not exist. Please try again.")
while True:
    n_captions = input("Enter the number (integer) of captions you want to generate here: ")
    # Check if the path exists
    if n_captions.isdigit():
        n_captions = int(n_captions)
        break
    else:
        print("Please insert an integer.")
captions_list = compute_coca(obj_image_folder, n_captions)

# COMPUTE AUDIOLDM
n, output_folder = generate_audio(captions_list)
print(f"{n} audios generated successfully in: {output_folder}")