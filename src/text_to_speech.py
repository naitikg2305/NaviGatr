# import pyttsx3

# engine = pyttsx3.init()

# # List available voices
# voices = engine.getProperty('voices')

# # Choose a natural-sounding voice (try different indexes)
# engine.setProperty('voice', voices[1].id)  # Change index for different voices

# # Adjust rate and volume
# engine.setProperty('rate', 180)  # Speech speed
# engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# # Speak text
# engine.say("Hello! This is a more natural voice using Microsoft SAPI.")
# engine.runAndWait()


import asyncio
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
import io

async def speak(text):
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")  # Change voice if needed
    audio_data = b""

    # Stream the audio into a buffer
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    # Convert to an audio segment and play
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
    play(audio)

# Run the async function
asyncio.run(speak("Hello! This is a real-time text-to-speech example using Edge TTS."))




