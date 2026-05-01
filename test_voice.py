import pyaudio
import wave
import requests
import tempfile
import os

def enregistrer_audio(duree=5):
    """Enregistre l'audio du micro pendant 'duree' secondes"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    
    print(f"🎙️ Enregistrement pendant {duree} secondes... Parlez maintenant !")
    frames = []
    for _ in range(0, int(RATE / CHUNK * duree)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Sauvegarder temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        wf = wave.open(tmp.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return tmp.name

# Enregistrer
temp_file = enregistrer_audio(5)

# Envoyer à l'API
with open(temp_file, 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/transcribe',
        files={'audio': f}
    )

print("📝 Résultat:", response.json())

# Nettoyer
os.unlink(temp_file)