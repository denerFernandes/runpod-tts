""" Example handler file. """

import runpod
from TTS.api import TTS
import torch
import math

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
    print("len(wav): ", len(wav))

    # Filter out NaN and Infinity values
    # TODO: debug error ''Failed to return job results. | 400, message='Bad Request'''
    # -> https://www.answeroverflow.com/m/1315032298369974412
    # -> https://www.answeroverflow.com/m/1208972636777095218
    # -> initially, it seemed like error was due to some NaNs in output
    # -> but, non-NaN outputs still yield an error
    # -> so the error might be due to ouput size
    wav_filtered = [float(x) for x in wav if not (math.isnan(x) or math.isinf(x))]
    print("type(wav_filtered): ", type(wav_filtered))
    print("len(wav_filtered): ", len(wav_filtered))

    json_data = {
        "text": text, 
        # "wav": [float(x) for x in wav],  # FastAPI does not support numpy types, only support native python types
        "wav": wav_filtered,
        "output_sample_rate": tts.config.audio.output_sample_rate,
        "sample_rate": tts.config.audio.sample_rate,
    }
    print("json_data['wav'][:5] = ", json_data['wav'][:5])

    return json_data


runpod.serverless.start({"handler": handler})
