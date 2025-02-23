import torch
import librosa
import numpy as np
import soundfile as sf
from torch.nn.parallel import DataParallel

from UNet import UNet
from feature_extract import extract_mel_spectrogram


# Load the noisy audio file
noisy_audio_path = 'noisy_test.wav'
noisy_audio, sr = librosa.load(noisy_audio_path, sr=44500, mono=True)  # same as training

# Convert audio to Mel spectrogram
mel_soec = extract_mel_spectrogram(noisy_audio, sr) 
print(f"Mel spectrogram shape: {mel_soec.shape}")
noisy_mel_spec = torch.tensor(mel_soec).unsqueeze(0)

# Load UNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet(in_channels=1, out_channels=1, init_channels=16).to(device) # same as training

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    unet = DataParallel(unet)

unet.load_state_dict(torch.load("saved_models/model_splits_lr1e3_b32_i16.pth")) # model path
unet.eval() 

# Pass the Mel spectrogram through the U-Net
noisy_mel_spec = noisy_mel_spec.to(device)
with torch.no_grad():
    denoised_mel_spec = unet(noisy_mel_spec.unsqueeze(1))

    
print("Model output shape:", denoised_mel_spec.shape)
denoised_mel_spec = denoised_mel_spec.squeeze(1).squeeze(0).detach().cpu().numpy()

print("Denoised Mel spectrogram shape:", denoised_mel_spec.shape)

# Convert the denoised Mel spectrogram to audio
output_audio = librosa.feature.inverse.mel_to_audio(denoised_mel_spec, sr=sr, n_fft=2048, hop_length=512, window='hamm') # same as training

# Save the denoised audio
output_path = 'model_results/denoised_audio.wav'
sf.write(output_path, output_audio, sr)
print(f"Denoised audio saved at {output_path}")


