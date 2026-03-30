import argparse
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from project_paths import instructblip_cache_dir, resolve_data_path

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default="Greece_4.jpg")
args = parser.parse_args()

MODEL_CACHE_DIR = instructblip_cache_dir()

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl", cache_dir=str(MODEL_CACHE_DIR))
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl", cache_dir=str(MODEL_CACHE_DIR))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

image = Image.open(resolve_data_path(args.image)).convert("RGB")
prompt = "A short image description:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=0.9,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)

# a painting of a house with flowers on the steps

# a painting of a traditional Chinese courtyard with blossoming peonies
# a painting of a traditional Indian home with colorful marigolds on the staircase
# a painting of a typical American house with a flower-filled front porch
