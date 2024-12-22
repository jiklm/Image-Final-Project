import dlib
import cv2
import argparse
from tensorflow.keras.models import load_model
import numpy as np
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Path to the input directory containing test images") #測試資料目錄的路徑
args = vars(ap.parse_args())

# 載入 Dlib 的預訓練人臉檢測器，用於檢測正面人臉
detector = dlib.get_frontal_face_detector()

# 載入build_model.py建立的預訓練模型（HDF5 格式）
model = load_model("emotion_model.h5")

# 定義3種情緒，與模型輸出對應
emotion_labels = ["Angry", "Happy", "Sad"]

# 載入測試資料目錄內的jpg圖片
image_paths = [os.path.join(args["dir"], fname) for fname in os.listdir(args["dir"]) if fname.endswith(".jpg")]

# 遍歷每張圖片
for image_path in image_paths:
    print(f"Processing image: {image_path}")

    # 使用 OpenCV 讀取圖片
    image = cv2.imread(image_path)
    # 將圖片轉為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 Dlib 偵測人臉，2代表圖片會被縮小2倍
    faces = detector(gray, 2)

    # 如果未檢測到人臉，跳過該圖片
    if len(faces) == 0:
        print("No faces detected in the image.")
        continue
    else:
        print(f"Detected faces: {len(faces)}")

    face_images = []  # 儲存裁剪後的人臉圖片
    for rect in faces:
        # 提取人臉的邊界框座標
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

        # 確保裁剪區在圖片範圍內
        if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
            print(f"Invalid face region: {x}, {y}, {w}, {h}")
        else:
            # 裁剪人臉區
            face = gray[y:y+h, x:x+w]
            # 如果裁剪區大小為 0，跳過該區域
            if face.size == 0:
                print("Empty face region detected.")
            else:
                # 將人臉調整為與FER-2013的訓練模型資料大小相同（48x48 像素）
                face_resized = cv2.resize(face, (48, 48))
                face_images.append(face_resized)

    emotions = []  # 儲存每個人臉的預測情緒

    # 對每張裁剪出的人臉進行情緒預測
    for face in face_images:
        # 將像素值標準化
        face_normalized = face / 255.0
        face_reshaped = np.expand_dims(face_normalized, axis=(0, -1))

        # 使用預訓練模型進行情緒預測
        predictions = model.predict(face_reshaped)
        # 取得情緒標籤
        emotion = emotion_labels[np.argmax(predictions)]
        emotions.append(emotion)
        print(f"Detected emotion: {emotion}")

    # 在原圖上繪製人臉邊界框和預測的情緒標籤
    for rect, emotion in zip(faces, emotions):
        # 提取邊界框座標
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        # 繪製邊界框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 在邊界框上方加上預測的情緒
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 顯示結果圖片
    cv2.imshow("Emotion Detection", image)
    cv2.waitKey(0)

# 關閉所有顯示窗
cv2.destroyAllWindows()
