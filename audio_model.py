import speech_recognition
from pydub import AudioSegment
import os
import clip
import subprocess
import librosa
import torch
from resources.audio_feature_projector import AudioFeatureProjector
from pydub.utils import mediainfo
import numpy as np
import io
import wave

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


def transcribe_audio_from_array(audio_array: np.ndarray, sampling_rate: int) -> str:
    """
    Transcribe an audio array to text using the SpeechRecognition library.

    Parameters:
    - audio_array (np.ndarray): The audio data as a NumPy array.
    - sampling_rate (int): The sampling rate of the audio data.

    Returns:
    - str: The transcribed text.
    """

    # Ensure the audio is in int16 format
    if audio_array.dtype != np.int16:
        # Normalize the array to the range of int16
        audio_normalized = audio_array / np.max(np.abs(audio_array))
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
    else:
        audio_int16 = audio_array

    # Create an in-memory bytes buffer
    wav_io = io.BytesIO()

    # Define WAV file parameters
    num_channels = 1  # Mono
    sample_width = 2  # Bytes per sample for int16
    framerate = sampling_rate
    num_frames = len(audio_int16)

    # Write the WAV file to the in-memory buffer
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(framerate)
        wav_file.writeframes(audio_int16.tobytes())

    # Seek to the beginning of the BytesIO buffer
    wav_io.seek(0)

    # Initialize the recognizer
    recognizer = speech_recognition.Recognizer()

    # Use the in-memory WAV file as the audio source
    with speech_recognition.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)  # Read the entire audio file

    try:
        # Perform transcription using Google Web Speech API
        transcription = recognizer.recognize_google(audio_data)
        return transcription
    except speech_recognition.UnknownValueError:
        return "Google Speech Recognition could not understand the audio."
    except speech_recognition.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"


def truncate_text(text, max_words=75):
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

def convert_audio_to_16bit_pcm(audio_path):
    # Define the output path for the converted file
    wav_path = os.path.splitext(audio_path)[0] + "_16bit.wav"
    if os.path.exists(wav_path):
        os.remove(wav_path)
    
    # Use FFmpeg to convert the audio to PCM signed 16-bit
    command = [
        "ffmpeg", "-i", audio_path,
        "-acodec", "pcm_s16le",       # Set audio codec to PCM signed 16-bit little-endian
        "-ar", "16000",               # Set sample rate (e.g., 16kHz)
        "-ac", "1",                   # Set audio channels to mono
        wav_path
    ]
    
    # Run the FFmpeg command
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    return wav_path

def convert_to_wav(audio_path):
    original_audio = AudioSegment.from_file(audio_path)
    wav_path = audio_path.rsplit('.', 1)[0] + ".wav"
    wav_path = convert_audio_to_16bit_pcm(wav_path)

    return wav_path

def transcribe_audio(wav_file):
    recognizer = speech_recognition.Recognizer()
    
    with speech_recognition.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except speech_recognition.UnknownValueError:
            # print("Google Speech Recognition could not understand the audio")
            return None
        except speech_recognition.RequestError:
            # print("Could not request results from Google Speech Recognition service")
            return None
    # return pipe({'array': audio_data, 'sampling_rate': sampling_rate})['text']


def preprocess_audio_file(audio_path):

    if not audio_path.endswith('.wav'):
        print('Converting to wav file....')
        converted_path = convert_to_wav(audio_path)
        print('Transcribing the audio....')
        transcript = transcribe_audio(converted_path)
        print('Transcription completed!')
        print(f'Transcription: {transcript}')
        return transcript

    elif audio_path.endswith('.wav') and mediainfo(audio_path)['sample_fmt'] != 's16':
        wav_path = convert_audio_to_16bit_pcm(audio_path)
        transcript = transcribe_audio(wav_path)
        return transcript

    else:
        transcript = transcribe_audio(audio_path)
        return transcript


def audio_preprocess(audio_array, sampling_rate):
    # Load audio file
    # try:
    #     y, sr = librosa.load(audio_path, sr=None)
    # except Exception as e:
    #     print('Audio load error:', e)
    #     return None, None, None
    y, sr = audio_array, sampling_rate
    # Initialize lists to hold features
    features_list = []

    # Extract MFCCs
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfccs_flat = mfccs.flatten()
        features_list.append(mfccs_flat)
    except Exception as e:
        print('Error extracting MFCCs:', e)

    # Extract pitch using YIN
    try:
        pitch = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_flat = pitch.flatten()
        features_list.append(pitch_flat)
    except Exception as e:
        print('Error extracting pitch:', e)

    # Calculate RMS energy
    try:
        energy = librosa.feature.rms(y=y)
        energy_flat = energy.flatten()
        features_list.append(energy_flat)
    except Exception as e:
        print('Error calculating RMS energy:', e)

    # Calculate spectral centroid
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_flat = spectral_centroid.flatten()
        features_list.append(spectral_centroid_flat)
    except Exception as e:
        print('Error calculating spectral centroid:', e)

    # Concatenate all features
    if features_list:
        audio_features = np.concatenate(features_list)
    else:
        print('No audio features extracted.')
        return None, None, None

    # Convert features to tensor and move to device
    audio_features_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)
    projector = AudioFeatureProjector(input_dim=audio_features.shape[0], output_dim=512).to(device)

    # Apply the projector (ensure it's on the device)
    with torch.no_grad():
        projected_audio_features = projector(audio_features_tensor).to(device)  # Shape: (1, 512)

    # Transcribe audio
    extracted_text = truncate_text(transcribe_audio_from_array(audio_array, sampling_rate))

    # Process text
    if extracted_text:
        try:
            text_tokens = clip.tokenize([extracted_text]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
        except Exception as e:
            # print(f"Error during tokenization or encoding: {e}")
            text_features = torch.zeros((1, 512), dtype=torch.float32).to(device)
    else:
        text_features = torch.zeros((1, 512), dtype=torch.float32).to(device)

    # Placeholder for image features (modify as needed)
    image_features = torch.zeros((1, 512), dtype=torch.float32).to(device)

    return text_features, projected_audio_features, image_features