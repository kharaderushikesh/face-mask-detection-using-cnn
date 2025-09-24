😷 Face Mask Detection using CNN

This project demonstrates Face Mask Detection using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.
The model is trained on the Face Mask Dataset
 from Kaggle.

📌 Features

Downloads dataset directly from Kaggle.

Preprocessing with ImageDataGenerator (rescaling + validation split).

Simple CNN model with Conv2D, MaxPooling, Dropout.

Training & Validation accuracy/loss visualization.

Prediction on custom images.

🚀 Getting Started
1. Open in Google Colab

Click the button to run this project in Colab:

2. Install Dependencies
!pip install -q kaggle tensorflow matplotlib opencv-python

3. Setup Kaggle API

Go to Kaggle Account → API
.

Create and download kaggle.json.

Upload kaggle.json to your Colab runtime.

Run:

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

4. Download Dataset
!kaggle datasets download -d omkargurav/face-mask-dataset
!unzip -q face-mask-dataset.zip


Dataset structure:

data/
 ├── with_mask/
 └── without_mask/

🧠 Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


Optimizer: Adam

Loss: Binary Crossentropy

Metrics: Accuracy

🖼 CNN Architecture Flow
Input Image (128x128x3)
        │
        ▼
┌────────────────────┐
│ Conv2D (32 filters)│
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ MaxPooling2D       │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ Conv2D (64 filters)│
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ MaxPooling2D       │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ Flatten            │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ Dense (128, ReLU)  │
│ Dropout (0.5)      │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ Dense (64, ReLU)   │
│ Dropout (0.5)      │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ Dense (1, Sigmoid) │ → Output: [Mask / No Mask]
└────────────────────┘

📊 Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

Accuracy & Loss Curves

Training vs. Validation Accuracy

Training vs. Validation Loss

✅ Evaluation
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy = {acc}")

🔮 Prediction

Test the model on a custom image:

img = cv2.imread("test.jpg")
resized_img = cv2.resize(img, (128,128)) / 255.0
reshaped_img = np.reshape(resized_img, (1,128,128,3))

pred = model.predict(reshaped_img)

if pred[0][0] > 0.5:
    print("😷 The person is wearing a mask.")
else:
    print("❌ The person is NOT wearing a mask.")

📌 Results

Achieves high accuracy (~95%+) on validation data.

Works on custom images uploaded to Colab.

📜 License

This project is for educational purposes. Dataset license: Unknown on Kaggle.
