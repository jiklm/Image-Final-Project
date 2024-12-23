# Image-Final-Project
# Emotion Recognition using CNN and Dlib

This project is a complete pipeline for emotion recognition using convolutional neural networks (CNNs) and Dlib's face detection capabilities. It includes training a model on a labeled dataset and using the trained model to detect emotions in images with human faces.

---

## **Features**
- Uses a **CNN** for emotion classification.
- Supports three emotion classes: `Angry`, `Happy`, and `Sad`.
- Utilizes **Dlib** for robust face detection.
- Processes images in bulk from a folder and displays detected emotions.

---

## **Project Structure**

```
project-directory/
├── face/
│   ├── train/
│   │   ├── Angry/
│   │   ├── Happy/
│   │   ├── Sad/
├── train.py
├── predict.py
├── emotion_model.h5
└── README.md
```

---

## **Setup Instructions**

### **Prerequisites**
Ensure you have the following installed:
- Python 3.6+
- TensorFlow/Keras
- OpenCV
- Dlib

### **Dataset Preparation**
1. Using [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) as dataset

---

## **How to Train the Model**

Run the training script, path to dataset is :
```bash
python build_model.py -d <path_to_training_data> -o <output_file_name>
```
Replace `<path_to_training_data>` with the directory containing your training images and `<output_file_name>` with `.h5` file name(default name is `emotion_model.h5`).

---

## **How to Test the Model**

Run the testing script:
```bash
python test_model.py -d <path_to_test_images>
```
Replace `<path_to_test_images>` with the directory containing your test images.

### **Key Steps in `predict.py`**
1. Loads the pre-trained model (`emotion_model.h5`).
2. Detects faces in the images using **Dlib's frontal face detector**.
3. Crops, resizes, and preprocesses the detected faces.
4. Predicts the emotion for each face and overlays the result on the image.
5. Displays the processed images with bounding boxes and predicted labels.

---

## **Customization**
1. **Add More Emotions**:
    - Add new labeled folders in the dataset (`face/train/<EmotionName>`).
    - Update the `emotion_labels` list in the training and testing scripts.
2. **Adjust Model Parameters**:
    - Modify the CNN architecture or training hyperparameters in `build_model.py`.

---

## **License**
This project is open-source and available under the MIT License.
