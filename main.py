# -----------------------------------------------------------------------------
# Nexus.assist
# Copyright (C) 2026 DrNexus
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import sys
import os
import time
from random import randint

OS = sys.platform

## Wake word
import numpy as np
import openwakeword
from openwakeword.model import Model

## Record
import sounddevice as sd
import wave

## STT
import pyaudio
from faster_whisper import WhisperModel

## AI
import json
import ollama

## SpeechBrain
# import torch
# from speechbrain.inference.speaker import SpeakerRecognition

WAKE_WORD_MODEL = "ressources/Hey_Nexus.onnx"
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

# Liste extraits audio test IA
demo_audio = ["voice_demo.mp3", "voice_demo1.mp3", "voice_demo2.mp3", "voice_demo3.mp3", "voice_demo4.mp3",
              "voice_demo5.mp3", ]

# Paramètres audio Wake word
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
GAIN = 4
WAKE_THRESHOLD = 0.5
WAKE_RESET_THRESHOLD = 0.2

# Paramètres audio pyaudio 
pyCHUNK = 1024
pyFORMAT = pyaudio.paInt16
pyCHANNELS = 1
pyRATE = 44100


# Record : code
def record(rate, time):
    frame_count = int(time * rate)
    audio = sd.rec(frame_count,
                   samplerate=pyRATE,
                   channels=CHANNELS,
                   dtype="int16")
    sd.wait()
    peak = int(np.abs(audio).max())
    if peak == 0:
        print("WARNING: No sound recorded, microphone may not work.")
    with wave.open("record.wav", "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(audio.tobytes())


record(44100, 1)


# STT : code
def stt(file, lang):
    segments, info = model.transcribe(file, language=lang)
    text = "".join(segment.text for segment in segments).strip()
    os.remove(file)
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
        content = response["message"]["content"].strip()
        data = json.loads(content)
        return data.get("instruction", "inconnu")

    except Exception as e:
        print(f"Erreur lors de l'interprétation : {e}")
        return "inconnu"


# Initialisation des commandes (dict de listes)
commands = {"stop_assistant": ["", "0"]}
print(commands["stop_assistant"][1])

audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

print(f"Awaiting for your command : '{WAKE_WORD_MODEL}'...", flush=True)
count = 0
try:
    print("Boucle d'écoute démarrée.", flush=True)
    wake_word_armed = True
    while True:
        # Lecture
        data = mic_stream.read(CHUNK, exception_on_overflow=False)
        audio_frame = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        audio_frame = np.clip(audio_frame * GAIN, -32768, 32767).astype(np.int16)
        # Analyse
        prediction = owwModel.predict(audio_frame)

        # Recup score
        score = prediction.get("Hey_Nexus", 0)

        # Déclenche 1 seule fois jusqu'à score repasse sous seuil
        if wake_word_armed and score > WAKE_THRESHOLD:
            wake_word_armed = False
            print(f"\nWake Word détecté ! (Score : {score:.2f})")

            mic_stream.stop_stream()
            record(44100, 5)
            try:
                transcripted_text = stt(f"record.wav", "fr")
                # transcripted_text = stt(f"ressources/{demo_audio[randint(0, len(demo_audio))]}", "fr")
                print("Transcription :", transcripted_text)

                commande = interpreter_commande(transcripted_text)
                print(f"Action à effectuer : {commande}")
            finally:
                mic_stream.start_stream()

            time.sleep(0.2)

        elif not wake_word_armed and score < WAKE_RESET_THRESHOLD:
            wake_word_armed = True
            score = 0.0

except KeyboardInterrupt:
    print("\nArrêt de l'assistant par l'utilisateur.")
except Exception as e:
    print(f"Erreur d'assistant : {e}")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()
