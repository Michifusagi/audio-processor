
# Does the audio processing and the STT transcription using AssemblyAI.

import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from models import Transcript
from openai import OpenAI

class AudioManager:
    def __init__(self, s3_client=None, s3_bucket: str = None, openai_client: OpenAI = None):
        self.s3 = s3_client
        self.s3_bucket = s3_bucket
        self.openai_client = openai_client

    def preprocess(self, s3_key: str) -> str:
        """
        Download the audio file from S3, denoise it, save locally, and return the local path.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            input_path = temp_input.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            output_path = temp_output.name
        self.s3.download_file(self.s3_bucket, s3_key, input_path)
        self._reduce_noise_stereo(input_path, output_path)
        os.remove(input_path)
        return output_path  # Caller should remove after use

    def _reduce_noise_stereo(self, input_path: str, output_path: str):
        y, sr = sf.read(input_path)
        if y.ndim == 1:
            y_denoised = self._denoise_channel(y, sr)
            y_resampled = librosa.resample(y_denoised, orig_sr=sr, target_sr=16000)
            sf.write(output_path, y_resampled, 16000)
        else:
            left, right = y[:, 0], y[:, 1]
            left_denoised = self._denoise_channel(left, sr)
            right_denoised = self._denoise_channel(right, sr)
            left_resampled = librosa.resample(left_denoised, orig_sr=sr, target_sr=16000)
            right_resampled = librosa.resample(right_denoised, orig_sr=sr, target_sr=16000)
            min_len = min(len(left_resampled), len(right_resampled))
            stereo_resampled = np.stack([
                left_resampled[:min_len],
                right_resampled[:min_len]
            ], axis=1)
            sf.write(output_path, stereo_resampled, 16000)

    def _denoise_channel(self, channel_audio: np.ndarray, sr: int) -> np.ndarray:
        stft = librosa.stft(channel_audio)
        magnitude, phase = np.abs(stft), np.angle(stft)
        noise_profile = np.mean(magnitude[:, :30], axis=1, keepdims=True)
        magnitude_denoised = np.maximum(magnitude - 1.0 * noise_profile, 0)
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        return librosa.istft(stft_denoised)


    def transcribe(self, input_audio_path: str) -> Transcript:
        try:
            with open(input_audio_path, "rb") as audio_file:
                resp = self.openai_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio_file,
                    response_format="text",
                    language="ja"
                )
            return Transcript(text=resp)
        finally:
            if os.path.exists(input_audio_path):
                os.remove(input_audio_path)
