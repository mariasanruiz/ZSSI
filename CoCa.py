from pathlib import Path
from sys import argv
import os
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm


def compute_coca(image_directory, n_captions):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    coca_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k",
        device=device,
    )


    def generate_caption_coca(image):
        im = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type):
            generated = coca_model.generate(im,
                                            seq_len=50,
                                            generation_type='top_p',
                                            temperature=1,
                                            top_p=1,
                                            repetition_penalty=5.0)
        return (
            open_clip.decode(generated[0].detach())
            .split("<end_of_text>")[0]
            .replace("<start_of_text>", "")
        )

    caption_list = []

    # Get list of image files in the directory
    image_files = [file for file in os.listdir(image_directory) if file.endswith((".jpg", ".png"))]

    # Generate captions for each image
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_directory, image_file)
        im = Image.open(image_path).convert("RGB")
        for _ in range(n_captions):
            caption_list.append(generate_caption_coca(im))

    return caption_list

