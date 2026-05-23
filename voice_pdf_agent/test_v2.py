import numpy as np
import torch
import av
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_path = r"C:\tmp_upload\indicwhisper_bhajan_v2"
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
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []

# --- 1. Check for NaN/Inf in weights ---
print("\n[1] Checking weights for NaN/Inf...")
nan_layers, inf_layers = [], []
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        nan_layers.append(name)
    if torch.isinf(param).any():
        inf_layers.append(name)
if nan_layers:
    print(f"  NaN found in: {nan_layers[:5]}")
elif inf_layers:
    print(f"  Inf found in: {inf_layers[:5]}")
else:
    print("  Weights look clean — no NaN or Inf")

# --- 2. Test with synthetic silence ---
print("\n[2] Testing with silence (3s)...")
silence = np.zeros(3 * 16000, dtype="float32")
inputs = proc(silence, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    out = model.generate(inputs.input_features, language="hi", task="transcribe")
print(f"  Raw tokens: {out[0].tolist()}")
print(f"  Text: '{proc.batch_decode(out, skip_special_tokens=True)[0]}'")

# --- 3. Test with actual recording ---
print("\n[3] Testing with Recording2.m4a...")
audio = load_audio(audio_file)
audio_30s = audio[:30 * 16000]
inputs = proc(audio_30s.astype("float32"), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    out = model.generate(inputs.input_features, language="hi", task="transcribe")
print(f"  Raw tokens: {out[0].tolist()}")
print(f"  With special tokens: '{proc.batch_decode(out, skip_special_tokens=False)[0]}'")
print(f"  Final text: '{proc.batch_decode(out, skip_special_tokens=True)[0]}'")
