""" Example handler file. """

import runpod
from TTS.api import TTS
import torch

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device = ", device)

tts = TTS(
    model_path="data/06_models/tts_api/XTTS-v2",
    config_path="data/06_models/tts_api/XTTS-v2/config.json"
).to(device)

print("Loaded model tts !!")


def handler0(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    name = job_input.get('name', 'World')
    return f"Hello, {name}!"

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    text = job_input.get('text')
    
    wav = tts.tts(
        text=text,
        speaker_wav="data/06_models/tts_api/XTTS-v2/en_sample.wav",
        language="en",
    )

    print("ran inference successfully !!!!!!")
    print("type(wav): ", type(wav))

    json_data = {
        "text": text, 
        "wav": [float(x) for x in wav],  # FastAPI does not support numpy types, only support native python types
        "output_sample_rate": tts.config.audio.output_sample_rate,
        "sample_rate": tts.config.audio.sample_rate,
    }

    return json_data


runpod.serverless.start({"handler": handler})
