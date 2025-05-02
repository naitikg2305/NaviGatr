import asyncio
import edge_tts
import sounddevice as sd
import io
import soundfile as sf
import threading


def play_audio(data, samplerate):
    """ Function to play the audio on the main thread """
    sd.play(data, samplerate=samplerate)
    sd.wait()

'''
async def speak(text):
    """ Async function to handle speech synthesis """
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    audio_data = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    # Decode MP3 data into PCM NumPy array
    audio_io = io.BytesIO(audio_data)
    data, samplerate = sf.read(audio_io, dtype='float32')

    # Use threading to play audio in the main thread
    threading.Thread(target=play_audio, args=(data, samplerate)).start()
'''

async def speak(text):
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    audio_data = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    # Decode and play audio synchronously
    audio_io = io.BytesIO(audio_data)
    data, samplerate = sf.read(audio_io, dtype='float32')

    play_audio(data, samplerate)  # This blocks until audio finishes

def text_to_speech(text):
    """ Function to trigger the TTS process """
    asyncio.run(speak(text))

if __name__ == "__main__":
    print("Running text_to_speech.py directly")
    text_to_speech("Hello. This is a test of text-to-speech with no robotic voice.")


# Call the function with the text you want to speak
# text_to_speech("Hello. This is a test of text-to-speech with no robotic voice.")
