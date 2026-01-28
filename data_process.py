import os
import shutil
from pathlib import Path
import tensorflow as tf

flat_dir = r'C:\Users\Hugo\Downloads\synth\images' 
target_base = r'C:\Users\Hugo\Downloads\ocr\kaggle_data\training_data' 

classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]

# Process images
moved_count = 0
skipped = 0
for filename in os.listdir(flat_dir):
    if filename.lower().endswith('.png'):
        # Extract class from filename, e.g., image_A_747.png or image_a_1234.png -> 'A'
        base = Path(filename).stem  # Remove extension
        if base.startswith('image_') and len(base) > 6:
            cls_char = base.split('_', 2)[1].upper()  # Split max 2 times: image_|X|NNNN -> X.upper()
            if cls_char in classes:
                src = os.path.join(flat_dir, filename)
                dst = os.path.join(target_base, cls_char, filename)
                shutil.copy(src, dst)  # Change to shutil.move(src, dst) to move instead of copy
                moved_count += 1
                print(f"Copied {filename} -> {cls_char}/")
            else:
                print(f"Skipping {filename}: invalid class '{cls_char}'")
                skipped += 1
        else:
            print(f"Skipping {filename}: unexpected name format (expected image_X_NNNN.png)")
            skipped += 1

print(f"\nProcessed: {moved_count} copied, {skipped} skipped.")
print(f"Target: {target_base}")
print("Ready for TensorFlow: tf.keras.utils.image_dataset_from_directory(target_base)")



# Your dataset (post-script)
ds = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\Hugo\Downloads\ocr\kaggle_data\training_data',
    image_size=(28, 28),  # Matches EMNIST; adjust if needed
    batch_size=32,
    label_mode='int'
)

# 1. Print structure (shows batch dims, e.g., (None, 28, 28, 3), (None,))
print("Dataset structure:")
print(ds.element_spec)  # Or: tf.data.experimental.structure(ds)

# 2. Peek first batch shapes
for batch_images, batch_labels in ds.take(1):
    print(f"Images batch shape: {batch_images.shape} (batch_size, height, width, channels)")
    print(f"Labels batch shape: {batch_labels.shape} (batch_size,)")
    print(f"Image dtype: {batch_images.dtype}, range: [{tf.reduce_min(batch_images):.1f}, {tf.reduce_max(batch_images):.1f}]")
    print(f"Num classes (inferred): {batch_labels.numpy().max() + 1}")

# 3. Dataset info
print(f"Class names: {ds.class_names}")  # Alphabetical folder order
print(f"Expected classes: {len(ds.class_names)}")  # Should be 36 (0-9, A-Z)