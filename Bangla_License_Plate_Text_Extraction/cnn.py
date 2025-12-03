import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================
# 1️⃣ Configuration
# =============================
DATASET_DIR =r"C:\LEVEL-3 SEM-1\LEVEL-3 SEM-1\Numerical Method Sessional-312\New folder\train\train"
  # <-- তোমার dataset path দাও
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 20

# =============================
# 2️⃣ Load dataset
# =============================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"  # only digits, so grayscale is enough
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

class_names = train_ds.class_names
print("✅ Classes:", class_names)

# =============================
# 3️⃣ Optimize loading (prefetch)
# =============================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =============================
# 4️⃣ Custom CNN Model
# =============================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.summary()

# =============================
# 5️⃣ Compile model
# =============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =============================
# 6️⃣ Train the model
# =============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =============================
# 7️⃣ Evaluate
# =============================
test_loss, test_acc = model.evaluate(val_ds)
print(f"\n🎯 Validation Accuracy: {test_acc:.4f}")

# =============================
# 8️⃣ Save model
# =============================
model.save("bangla_digit_cnn_model.h5")
print("✅ Model saved successfully!")

# =============================
# 9️⃣ Plot Accuracy & Loss
# =============================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')
plt.show()

# =============================
# 🔟 Prediction Example
# =============================
def predict_digit(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale"
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # batch dimension
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    pred_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    print(f"🔢 Predicted Digit: {pred_class} ({confidence:.2f}% confidence)")
    return pred_class

# Example:
# predict_digit(r"C:\path\to\test_img.png")

