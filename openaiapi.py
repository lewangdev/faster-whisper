from typing import Annotated
from fastapi import FastAPI, Form, File

from faster_whisper import WhisperModel
import torch
import sys
from io import BytesIO

device = torch.device('cuda:0')
print('Using device:', device, file=sys.stderr)
model_size = 'large-v2'
# model_size = 'tiny'
device_type = "cuda"
compute_type = "float16"
whisper_engine = WhisperModel(model_size, device=device_type, compute_type=compute_type)

app = FastAPI()


@app.post("/v1/audio/transcriptions")
def create_transcription(file: Annotated[bytes, File()],
                         model: Annotated[str, Form()] = 'whipser-1',
                         language: Annotated[str | None, Form()] = None,
                         prompt: Annotated[str | None, Form()] = None):
    segments, _ = whisper_engine.transcribe(BytesIO(file), beam_size=5,
                                            language=language,
                                            initial_prompt=prompt,
                                            word_timestamps=False,
                                            vad_filter=True,
                                            vad_parameters=dict(min_silence_duration_ms=50))
    sentences = []
    for segment in segments:
        sentences.append(segment.text)

    return {
        "text": "".join(sentences)
    }
