import tensorflow as tf
import numpy as np

data_dir = r'C:\Users\Hugo\Downloads\ocr\kaggle_data\training_data'
ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(28, 28),
    batch_size=32,
    label_mode='int'
)

print(f"Total files: {len(ds.class_names)} classes detected.")
print("Dataset structure:", ds.element_spec)

# Peek first batch
for images, labels in ds.take(1):
    print(f"Images: {images.shape} (dtype: {images.dtype}, range [{tf.reduce_min(images):.0f}, {tf.reduce_max(images):.0f}])")
    print(f"Labels: {labels.shape}, range {tf.reduce_min(labels).numpy()}-{tf.reduce_max(labels).numpy()}")

print("Class names:", ds.class_names)
print("Num classes:", len(ds.class_names))
print("Expected model output: Dense({})".format(len(ds.class_names)))

# Optional: Full label coverage check
label_set = set()
for _, labels in ds.take(10):  # Check first ~320 samples
    label_set.update(labels.numpy())
print(f"Labels in first batches: {sorted(label_set)} (coverage: {len(label_set)}/{len(ds.class_names)})")
