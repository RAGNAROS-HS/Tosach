import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and prepare data (Custom Dataset)
data_dir = r'C:\Users\Hugo\Downloads\ocr\kaggle_data\training_data'
batch_size = 64
img_height = 28
img_width = 28

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size
)

class_names = train_ds.class_names
print(f"Classes found: {len(class_names)}")

# Define model
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPool2D(2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPool2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names))
])

# Compile with optimizer, loss, metrics
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Train
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save model
model.save('ocr_model.keras')
print("Model saved to 'ocr_model.keras'")