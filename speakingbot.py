import azure.cognitiveservices.speech as speechsdk 
import threading 
from playsound import playsound 
import os 
from dotenv import load_dotenv

load_dotenv()
# Azure setup 
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SUBSCRIPTION_KEY'),endpoint=os.environ.get("REGION_ID"))
speech_config.speech_synthesis_voice_name = 'en-US-JennyNeural'
AUDIO_FILE="output.wav"
audio_config = speechsdk.audio.AudioOutputConfig(filename=AUDIO_FILE)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config,audio_config=audio_config)

## text to speech 
def speak_async(text):
    def run():
        speech_synthesizer.speak_text_async(text).get()
        playsound(AUDIO_FILE)
    threading.Thread(target=run, daemon=True).start()

  
