import sys
import os
OS = sys.platform

## Wake word
import numpy as np
import openwakeword
from openwakeword.model import Model

## STT
import pyaudio
from faster_whisper import WhisperModel

## AI
import json
import ollama

## SpeechBrain
#import torch
#from speechbrain.inference.speaker import SpeakerRecognition

openwakeword.utils.download_models()
WAKE_WORD_MODEL = "Hey_Nexus.onnx"
owwModel = Model(wakeword_models=[WAKE_WORD_MODEL])

# Chargement de Whisper
model = WhisperModel("small", device="cpu", compute_type="int8")

# liste commandes dispo IA
COMMANDS_CATALOG = {
  "allumer_lumiere": "Allumer ou élever la luminosité des lampes",
  "couper_son": "Mettre en sourdine, couper le volume ou la musique",
  "donner_meteo": "Demander les prévisions météo ou la température extérieure",
  "lancer_musique": "Jouer de la musique ou un artiste spécifique",
  "inconnu": "Si aucune commande ne correspond à la demande",
}

# Paramètres audio Wake word
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
GAIN = 4

# Paramètres audio pyaudio 
pyCHUNK = 1024
pyFORMAT = pyaudio.paInt16
pyCHANNELS = 1 if sys.platform == 'darwin' else 2
pyRATE = 44100
pyRECORD_SECONDS = 5

# STT : code
def STT(file,lang):
  segments, info = model.transcribe(file, language=lang)
  text = "".join(segment.text for segment in segments).strip()
  return text

# AI : code
def interpreter_commande(text_user: str) -> str:
  prompt_system = f"""
    Tu es le module d'intelligence d'un assistant vocal. Ton unique rôle est d'analyser la phrase de l'utilisateur et de l'associer à l'une de ces commandes :
    {json.dumps(COMMANDS_CATALOG, ensure_ascii=False, indent=2)}
    
    Règles strictes :
    - Réponds UNIQUEMENT au format JSON valide, sans markdown (pas de ```json), avec cette structure exacte :
    {{"instruction": "nom_de_la_commande"}}
    - Si la phrase ne correspond à rien, renvoie "inconnu".
    """

  try:
    response = ollama.chat(
        model="qwen2.5:3b", 
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": text_user},
        ],
        options={
            "temperature": 0.0
        },  
    )

    #print(response)
    content = response["message"]["content"].strip()
    #print(content)
    data = json.loads(content)
    return data.get("instruction", "inconnu")

  except Exception as e:
    print(f"Erreur lors de l'interprétation : {e}")
    return "inconnu"

# Initialisation des commandes (dict de listes)
commands = {"stop_assistant" : ["","0"]}
print(commands["stop_assistant"][1])

audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

print(f"Awaiting for your command : '{WAKE_WORD_MODEL}'...")

try:
  while True:
    # Lecture
    data = mic_stream.read(CHUNK, exception_on_overflow=False)
    audio_frame = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    audio_frame = np.clip(audio_frame * GAIN, -32768, 32767).astype(np.int16)
    # Analyse
    prediction = owwModel.predict(audio_frame)

    # Récupération score
    score = prediction.get("Hey_Nexus", 0)

    # Affichage optionnel
    print(f"Score : {score:.2f}   ", end="\r")

    # score dépasse seuil déclenchement
    if score > 0.5:
      print(f"\nWake Word détecté ! (Score : {score:.2f})")

      # --- ICI : SUITE ---
      # 1. Couper//mettre en pause wake word
      # 2. Enregistrer phrase (ex: 5 secondes)
      STT("voice_demo.mp", "fr")
      print("Transcription :", transcription)

      commande = interpreter_commande(transcripted_text)
      print(f"Action à effectuer :{interpreter_commande(transcription)}")


except KeyboardInterrupt:
  print("\nArrêt de l'assistant par l'utilisateur.")
finally:
  mic_stream.stop_stream()
  mic_stream.close()
  audio.terminate()