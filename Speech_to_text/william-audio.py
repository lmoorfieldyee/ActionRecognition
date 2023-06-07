import speech_recognition as sr
import openai
import requests
import json
from pydub import AudioSegment
from pydub.playback import play
import io
import os

def generate_chat_response(input_text):
    # Initialiser OpenAI avec votre clé d'API
    openai.api_key = "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"

    # Appeler l'API de génération de ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input_text}]
        # prompt=input_text,
        # max_tokens=50,
        # temperature=0.7,
        # n=1,
        # stop=None,
        # temperature=0.7
    )

    # Récupérer la réponse générée
    generated_text = response['choices'][0]['message']['content']

    return generated_text

def transcribe_audio():
    # Créer un objet Recognizer
    r = sr.Recognizer()

    # Ouvrir le flux audio du microphone
    with sr.Microphone() as source:
        print("Say something...")
        # Régler le niveau de bruit ambiant pour une meilleure détection
        r.adjust_for_ambient_noise(source)

        # Capturer l'audio en continu
        audio_stream = r.listen(source)

    try:
        # Utiliser la reconnaissance vocale
        text = r.recognize_google(audio_stream, language="en-EN")
        return text
    except sr.UnknownValueError:
        print("La reconnaissance vocale n'a pas pu comprendre l'audio.")
    except sr.RequestError as e:
        print("Erreur lors de la récupération des résultats de la reconnaissance vocale ; {0}".format(e))

def convert_text_to_speech(text, voice_id):

    file_probe = "./ffprobe.exe"

    file_mpeg = "./ffmpeg.exe"

    # pydub.utils.(file_pat)
    AudioSegment.converter = file_mpeg
    AudioSegment.ffmpeg = file_mpeg
    AudioSegment.ffprobe = file_probe


    # Endpoint de l'API OpenAI Whisper
    endpoint = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}".format(voice_id=voice_id)

    # Headers pour l'authentification et le type de contenu
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": "f7896dcbff5fd933e2132a05d7fc5e6e"  # Remplacez par votre clé d'API OpenAI
    }

    # Corps de la requête avec le texte à synthétiser
    data = {
        "text": text
    }

    # Envoyer la requête POST à l'API Whisper
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    # Vérifier le code de réponse
    if response.status_code == 200:
        print("Lecture de la reponse audio...")
        # Récupérer le contenu audio
        audio_content = response.content
       
        with open('audio_save.wav', 'wb') as file:
            file.write(audio_content)


        # Charger les données audio en mémoire avec pydub
        audio = AudioSegment.from_mp3(io.BytesIO(audio_content))

        # Lire l'audio
        play(audio)
    else:
        print("Erreur lors de la synthèse vocale. Code de réponse :", response.status_code)
        print("Message d'erreur :", response.text)

# Appeler la fonction de transcription audio en continu
while True:
    result = transcribe_audio()
    if result:
        print("Texte transcrit :")
        print(result)

        # Envoyer le texte transcrit à ChatGPT pour obtenir une réponse
        response_text = generate_chat_response(result)

        # Envoyer la reponse de gpt a evenlabs pour generer un fichier audio
        print("'Enregistrement de l'audio")
        convert_text_to_speech(response_text, '21m00Tcm4TlvDq8ikWAM')
        
        # Afficher la réponse générée par ChatGPT
        print("Réponse de ChatGPT :")
        print(response_text)

# Obtenez le texte transcrit à partir de la reconnaissance vocale
transcribed_text = "Votre texte transcrit ici"
