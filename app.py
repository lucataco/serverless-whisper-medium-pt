import os
import torch
import base64
import whisper
from io import BytesIO
from multiple_datasets.hub_default_utils import convert_hf_whisper

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    # Write HF model to local whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    convert_hf_whisper("jlondonobo/whisper-medium-pt", "local_whisper_model.pt")
    # Load the whisper model
    model = whisper.load_model("local_whisper_model.pt", device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    if mp3BytesString == None:
        return {'message': "No input provided"}
    
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    # Run the model
    result = model.transcribe("input.mp3", language="pt")["text"]
    output = {"text":result}
    os.remove("input.mp3")
    # Return the results as a dictionary
    return output
