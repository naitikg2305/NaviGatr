import asyncio
import edge_tts
import sounddevice as sd
import io
import soundfile as sf
import threading


def play_audio(data, samplerate):
    sd.play(data, samplerate=samplerate)
    sd.wait()


async def speak(text):
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    audio_data = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    audio_io = io.BytesIO(audio_data)
    data, samplerate = sf.read(audio_io, dtype='float32')

    play_audio(data, samplerate)

def text_to_speech(text):
    asyncio.run(speak(text))

if __name__ == "__main__":
    print("Running text_to_speech.py directly")
    text_to_speech("Hello. This is a test of text-to-speech with no robotic voice.")

