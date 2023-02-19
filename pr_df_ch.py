import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained neural network for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained deep learning model for detecting deepfakes
model = tf.keras.models.load_model('deepfake_detection_model.h5')

# Function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to detect faces in an image
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

# Function to classify an image as real or fake
def classify_image(img):
    img = preprocess_image(img)
    pred = model.predict(img)[0][0]
    return pred

# Function to process a video and detect deepfakes
def detect_deepfakes(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            pred = classify_image(face_img)
            if pred > 0.5:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Fake', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Real', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Example usage: Detect deepfakes in a video file
detect_deepfakes('test_video.mp4')