import cv2
import numpy as np
import pyaudio
import wave
import librosa
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import threading
import time

# Load the trained face emotion model
# model_best = load_model('face_model.h5')  # Ensure your model is in the same folder
model_best = load_model('efficient_face_model.h5')  # Ensure your model is in the same folder


# Emotion classes for face
face_class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Audio recording settings
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
audio_filename = "output.wav"

# Shared variable for audio emotion
audio_emotion_label = "Listening..."

def record_audio(filename, duration=5):
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    print('Recording audio...')
    for _ in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def analyze_audio(filename):
    global audio_emotion_label
    try:
        y, sr = librosa.load(filename)

        # Extract pitch and energy
        pitch = librosa.yin(y, fmin=50, fmax=300)
        energy = np.sum(y ** 2) / len(y)

        avg_pitch = np.mean(pitch)
        avg_energy = energy

        # Basic classification
        if avg_pitch > 180 or avg_energy > 0.01:
            audio_emotion_label = "Excited / Stressed"
        elif avg_pitch < 120:
            audio_emotion_label = "Calm / Confident"
        else:
            audio_emotion_label = "Neutral"
    except Exception as e:
        audio_emotion_label = f"Audio Error"

def audio_loop():
    while True:
        record_audio(audio_filename, duration=5)
        analyze_audio(audio_filename)
        time.sleep(1)

# Start audio thread
audio_thread = threading.Thread(target=audio_loop, daemon=True)
audio_thread.start()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.vstack([face_image])

        predictions = model_best.predict(face_image, verbose=0)
        face_emotion_label = face_class_names[np.argmax(predictions)]

        # Draw rectangle and label
        cv2.putText(frame, f'Face: {face_emotion_label}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show audio emotion
    cv2.putText(frame, f'Voice: {audio_emotion_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection (Face + Voice)', frame)

    # Break loop if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
