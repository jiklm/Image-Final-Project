from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

# 解析命令行參數
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=True, help="Path to the training data directory")
ap.add_argument("-o", "--output", default="emotion_model.h5", help="Path to save the trained model (default: emotion_model.h5)")
args = vars(ap.parse_args())

# 構建模型(CNN)
model = Sequential([
    # 第一層捲積層：使用32個3x3的濾波器，激活函數為ReLU，輸入圖片大小為48x48，單通道
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    # 最大池化層：將特徵圖降維，池化區域大小為2x2
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout 層：丟棄25%的神經元以減少過擬合
    Dropout(0.25),
    # 第二層捲積層：使用64個3x3的濾波器
    Conv2D(64, (3, 3), activation='relu'),
    # 最大池化層
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout 層
    Dropout(0.25),
    # 將多維數據展平成一維向量，為全連接層準備輸入
    Flatten(),
    # 全連接層：包含128個神經元，激活函數為ReLU
    Dense(128, activation='relu'),
    # Dropout 層
    Dropout(0.5),
    # 輸出層：3個神經元（3種情緒），使用softmax激活函數輸出概率分布
    Dense(3, activation='softmax')  # 3種情緒
])

# 編譯模型
# 使用 Adam 優化器，學習率設為 0.001
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練資料增強
# 使用ImageDataGenerator進行資料預處理和增強
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 將像素值標準化到[0, 1]
    rotation_range=10,  # 隨機旋轉角度範圍為[-10, 10]度
    width_shift_range=0.1,  # 隨機水平平移範圍為圖片寬度的10%
    height_shift_range=0.1  # 隨機垂直平移範圍為圖片高度的10%
)

# 從目錄載入訓練資料
train_generator = train_datagen.flow_from_directory(
    args["data_dir"],  # 訓練資料目錄
    target_size=(48, 48),  # 將圖片調整為48x48大小
    batch_size=64,  # 每批處理64張圖片
    color_mode="grayscale",  # 灰階
    class_mode="categorical"  # 設定標籤為多分類模式，返回one-hot形式
)

# 訓練模型
# 訓練模型50個epoch
model.fit(train_generator, epochs=50, steps_per_epoch=100)

# 儲存模型
# 將訓練完成的模型儲存
model.save(args["output"])
