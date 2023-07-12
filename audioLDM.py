from diffusers import AudioLDMPipeline
import torch
import scipy
import os


def generate_audio(captions_list):
    output_folder = "generated_audios"  # Name of the output folder
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    repo_id = "cvssp/audioldm-s-full-v2"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, device=device)
    #pipe = pipe.to("cuda")
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for i, caption in enumerate(captions_list):
        audio = pipe(caption, num_inference_steps=10, audio_length_in_s=5.0).audios[0]
        file_name = f"audio_generated_{i}.wav"
        file_path = os.path.join(output_folder, file_name)

        scipy.io.wavfile.write(file_path, rate=16000, data=audio)

    return len(captions_list), output_folder


