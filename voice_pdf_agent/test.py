import numpy as np
import torch
import av
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_path = r"C:\tmp_upload\indicwhisper_bhajan"
audio_file = r"C:\tmp_upload\Recording2.m4a"

def load_audio(path, target_sr=16000):
    container = av.open(path)
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    samples = []
    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        if not isinstance(resampled, list):
            resampled = [resampled]
        for f in resampled:
            samples.append(f.to_ndarray()[0])
    return np.concatenate(samples)

print("Loading model...")
proc = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
model.eval()
print("Model loaded!")

print("Loading audio...")
audio = load_audio(audio_file)
print(f"Audio duration: {len(audio)/16000:.1f}s")
print(f"Audio stats — min: {audio.min():.4f}, max: {audio.max():.4f}, rms: {np.sqrt(np.mean(audio**2)):.4f}")

# Whisper processes max 30s at a time
audio_30s = audio[:30 * 16000]
inputs = proc(audio_30s.astype("float32"), sampling_rate=16000, return_tensors="pt")

# Clear any conflicting settings from the checkpoint's generation_config
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []

print("\nTranscribing (auto-detect language)...")
with torch.no_grad():
    out = model.generate(inputs.input_features)
print(f"Auto-detected: '{proc.batch_decode(out, skip_special_tokens=True)[0]}'")

print("\nTranscribing (Hindi forced)...")
with torch.no_grad():
    out = model.generate(inputs.input_features, language="hi", task="transcribe")
print(f"Hindi: '{proc.batch_decode(out, skip_special_tokens=True)[0]}'")
