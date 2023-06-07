from pydub import AudioSegment
from pydub.playback import play
import pyaudio
import requests
import json
import io
import simpleaudio as sa
import sys
import subprocess

def convert_text_to_speech(text, voice_id):

    print("Audiosegment 111: ",AudioSegment.ffmpeg)

        #sys.path.append('c:\python39\lib\site-packages\ffmpeg-python')
    file_probe = "C:/Users/willfrid boris/Documents/william/ffmpeg-6.0-essentials_build/bin/ffprobe.exe"

    file_mpeg = "C:/Users/willfrid boris/Documents/william/ffmpeg-6.0-essentials_build/bin/ffmpeg.exe"

    # pydub.utils.(file_pat)
    AudioSegment.converter = file_mpeg
    AudioSegment.ffmpeg = file_mpeg
    AudioSegment.ffprobe = file_probe

    print("Audiosegment : ",AudioSegment.ffmpeg)


    # Endpoint de l'API OpenAI Whisper
    endpoint = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}".format(voice_id=voice_id)

    # Headers pour l'authentification et le type de contenu
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": "f7896dcbff5fd933e2132a05d7fc5e6e"  # Remplacez par votre clé d'API OpenAI
    }

    # Corps de la requête avec le texte à synthétiser
    data = {
        "text": "je veux aller jouer au ballon."
    }

    # Envoyer la requête POST à l'API Whisper
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    # Vérifier le code de réponse
    if response.status_code == 200:
        print("Lecture de la reponse audio...")
        # Récupérer le contenu audio
        audio_content = response.content

        with open('audio_save3.wav', 'wb') as file:
            file.write(audio_content)
        
        # audio = pyaudio.PyAudio()

        # print(f"format = ", audio.get_format_from_width(2) )

        # stream = audio.open(format=audio.get_format_from_width(2),
        # channels=1, rate=16000, output=True)

        # stream.write(audio_content)

        # stream.stop_stream()
        # stream.close()
        # audio.terminate()
        
        # Charger les données audio en mémoire avec pydub
        audio = AudioSegment.from_mp3(io.BytesIO(audio_content))
        #subprocess.Popen(['mpg123', '-q', 'audio_save2.mp3']).wait()
        # wave_obj = sa.play_buffer(audio_content, 1, 2, 44100)
        # play_obj = wave_obj.play()
        # play_obj.wait_done()
        # audio.export('save_audio2.wav', format='wav')

        # Lire l'audio
        play(audio)
    else:
        print("Erreur lors de la synthèse vocale. Code de réponse :", response.status_code)
        print("Message d'erreur :", response.text)


convert_text_to_speech('text', '21m00Tcm4TlvDq8ikWAM')