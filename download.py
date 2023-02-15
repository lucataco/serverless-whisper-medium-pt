# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import torch
import whisper
from multiple_datasets.hub_default_utils import convert_hf_whisper

def download_model():

    convert_hf_whisper("jlondonobo/whisper-medium-pt", "local_whisper_model.pt")
    model = whisper.load_model("local_whisper_model.pt")

if __name__ == "__main__":
    download_model()