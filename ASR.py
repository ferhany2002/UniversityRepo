#import whisper
import time
import json
import whisper_timestamped as whisper


#Original version start and endtime for the sample and then use the extractor with sample frequency 100
#Accelerometer data format is a matrix

start_time = time.time()
valid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]

model = whisper.load_model("medium")

audio = whisper.load_audio("11.wav")

result = whisper.transcribe(model, audio, language="nl", fp16=False, verbose=True)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# Write the result to a text file
with open('Transcription11.txt', 'w') as file:
    file.write(json.dumps(result, indent=4))

print("done")

end_time = time.time()

#
# import whisper
# from pydub import AudioSegment
# import numpy as np
#
# audio_py = AudioSegment.from_wav('1normalized.wav')
#
# # Load RTTM file
# with open('1.rttm', 'r') as f:
#     rttm = [line.strip().split() for line in f]
#
# model = whisper.load_model("medium")
#
# # For each event in the RTTM file
# for event in rttm:
#     # Extract start time and duration
#     start_time = float(event[3])
#     duration = float(event[4])
#     # Convert start time and duration from seconds to milliseconds
#     start_time *= 1000
#     duration *= 1000
#     # Slice the audio
#     segment = audio_py[start_time - 500:start_time + duration + 1000]
#     # You might need to convert the PyDub audio segment to the format expected by Whisper
#     audio = whisper.load_audio(segment)
#
#     result = model.transcribe(audio, language="nl", fp16=False, verbose=False)
#     print(result['text'], start_time/1000, duration/1000)



# load audio and pad/trim it to fit 30 seconds

# make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)
#
# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")
#
# # decode the audio
# options = whisper.DecodingOptions(fp16 = False)
# result = whisper.decode(model, mel, options)
#
# # print the recognized text
# print(result)