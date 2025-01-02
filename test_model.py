import dlib
import cv2
import argparse
from tensorflow.keras.models import load_model
import numpy as np
import os

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Path to the input directory containing test images")
args = vars(ap.parse_args())

# Load Dlib's pre-trained frontal face detector for detecting faces
detector = dlib.get_frontal_face_detector()

# Load the pre-trained model (HDF5 format) built in build_model.py
model = load_model("emotion_model.h5")

# Define the 3 emotion categories corresponding to the model's output
emotion_labels = ["Angry", "Happy", "Sad"]

# Load all jpg images from the specified test directory
image_paths = [os.path.join(args["dir"], fname) for fname in os.listdir(args["dir"]) if fname.endswith(".jpg")]

# Loop through each image in the directory
for image_path in image_paths:
    print(f"Processing image: {image_path}")

    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using Dlib's face detector, downscale the image by a factor of 2
    faces = detector(gray, 2)

    # If no faces are detected, skip the image
    if len(faces) == 0:
        print("No faces detected in the image.")
        continue
    else:
        print(f"Detected faces: {len(faces)}")

    face_images = []  # Store cropped face images
    for rect in faces:
        # Extract face bounding box coordinates
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

        # Ensure the cropped region is within the image boundaries
        if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
            print(f"Invalid face region: {x}, {y}, {w}, {h}")
        else:
            # Crop the face region
            face = gray[y:y+h, x:x+w]
            # If the cropped region has zero size, skip it
            if face.size == 0:
                print("Empty face region detected.")
            else:
                # Resize the face to match the size used during training (48x48 pixels)
                face_resized = cv2.resize(face, (48, 48))
                face_images.append(face_resized)

    emotions = []  # Store the predicted emotions for each face

    # Predict emotions for each cropped face
    for face in face_images:
        # Normalize pixel values
        face_normalized = face / 255.0
        face_reshaped = np.expand_dims(face_normalized, axis=(0, -1))

        # Use the pre-trained model to predict emotions
        predictions = model.predict(face_reshaped)
        # Get the emotion label with the highest probability
        emotion = emotion_labels[np.argmax(predictions)]
        emotions.append(emotion)
        print(f"Detected emotion: {emotion}")

    # Draw face bounding boxes and predicted emotion labels on the original image
    for rect, emotion in zip(faces, emotions):
        # Extract bounding box coordinates
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        # Draw bounding box around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add the predicted emotion label above the bounding box
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the result image
    cv2.imshow("Emotion Detection", image)
    cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
