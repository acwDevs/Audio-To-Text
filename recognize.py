import sys
import time
import speech_recognition as sr
import moviepy.editor as mp
from scipy.io import wavfile
import noisereduce as nr
from translate import Translator
from pytube import YouTube
from pyannote.audio import Pipeline
from os import path
from os import rename
from os import listdir
from os import remove
from os import environ


NOSPEEKDURATION = 0.5

TOKEN = ""

# Get token from token.txt
if path.isfile("token.txt"):
    with open("token.txt", "r") as f:
        TOKEN = f.read().strip()
else:
    print("Please provide a token in token.txt")
    sys.exit(1)

# Get the path to the current file
savePath = path.dirname(path.realpath(__file__))

# #link of the video to be downloaded 
if len(sys.argv) < 2:
    print("Please provide a link to the video.")
    sys.exit(1)

print("Downloading video from", sys.argv[1])
link = sys.argv[1]

try: 
    # object creation using YouTube 
    yt = YouTube(link) 
except: 
    #to handle exception 
    print("Connection Error.") 

# Get all streams and filter for audio only
webm_streams = yt.streams.filter(only_audio=True)

# get the video with the highest resolution
d_audio = webm_streams[-1]

try: 
    # downloading the video 
    d_audio.download(savePath)
    print('Video downloaded successfully!')
except: 
    print("Some Error.")

webmPath = path.join(savePath, "audio.webm")
# Rename the file to audio.wav
for file in listdir(savePath):
    if file.endswith('.webm'):
        rename(path.join(savePath, file), webmPath)
        break

# # Path to the audio file
audioPath = path.join(savePath, "audio.wav")


# Convert the audio file to wav format
if path.isfile(audioPath):
    remove(audioPath)
clip = mp.AudioFileClip(webmPath)
clip.write_audiofile("audio.wav")
remove("audio.webm")

# # # Reduce noise in the audio
# rate, data = wavfile.read(audioPath)
# data = nr.reduce_noise(data, rate)

# with sr.Microphone() as source:
#     r = sr.Recognizer()
#     audio = r.listen(source)
#     print(r.recognize_google(audio))
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=TOKEN)

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("audio.wav")

# print the result
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append((turn.start, turn.end, speaker))
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# Group consecutive segments with the same speaker
# i = 0
# while i < len(segments) - 1:
#     if segments[i][2] == segments[i + 1][2]:
#         segments[i] = (segments[i][0], segments[i + 1][1], segments[i][2])
#         segments.pop(i + 1)
#     else:
#         i += 1

# # Remove and clips that are too short
# i = 0
# while i < len(segments):
#     if segments[i][1] - segments[i][0] < NOSPEEKDURATION:
#         segments.pop(i)
#     else:
#         i += 1

# Ouput the segments
print("Segments:")
for i in range(len(segments)):
    print("Segment", i, ":", segments[i])


# Recognize the audio
with sr.AudioFile(audioPath) as source:
    tr = Translator(to_lang="es")
    length = source.DURATION
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    audio = r.record(source) 
    for i in range(len(segments)):
        start = abs((segments[i][0] * 1000) - 300)
        end = (segments[i][1] * 1000)
        speaker = segments[i][2]
        clip = audio.get_segment(start, end)
        try:
            content = r.recognize_google(clip)
            print("Speaker", speaker, ":", content)
        except sr.UnknownValueError:
            print("Could not understand the audio:", end-start)
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        time.sleep(5)
    # content = r.recognize_google(audio)
    # print("Translations:")
    # print(" English:", content)  # English
    # tr = Translator(to_lang="es")
    # print(" Spanish:",tr.translate(content))  # Spanish
    # tr = Translator(to_lang="de") # German
    # print(" German:",tr.translate(content))  # German
    # tr = Translator(to_lang="ja") # Japanese
    # print(" Japanese:",tr.translate(content))  # Japanese


    # segmentCount = source.DURATION / 1.2
    # timeStamp = 0.0
    # while timeStamp < source.DURATION:
    #     audio = r.record(source, duration=1.2)
    #     try:
    #         content = r.recognize_google(audio)
    #         print("Recognized:", content)
    #     except sr.UnknownValueError:
    #         print("Could not understand the audio")
    #     except sr.RequestError as e:
    #         print("Could not request results; {0}".format(e))
    #     timeStamp += 1.2
