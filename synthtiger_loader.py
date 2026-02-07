"""
TensorFlow/Keras data loader for SynthTiger dataset.
Designed for CRNN training with CTC loss.
"""
import tensorflow as tf
import os

# Character set - standard alphanumeric for OCR
# Adjust this to match your training needs
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_INDEX = len(CHARS)  # CTC blank label

# Image dimensions (CRNN standard)
IMG_HEIGHT = 32
IMG_WIDTH = 100  # Will be padded/resized proportionally


def char_to_index(char):
    """Convert character to index. Unknown chars map to blank."""
    if char in CHARS:
        return CHARS.index(char)
    return BLANK_INDEX


def index_to_char(idx):
    """Convert index back to character."""
    if 0 <= idx < len(CHARS):
        return CHARS[idx]
    return ''


def encode_label(label, max_label_len):
    """Encode string label to integer array for CTC."""
    encoded = [char_to_index(c) for c in label]
    # Truncate if too long
    encoded = encoded[:max_label_len]
    # Pad with -1 (will be masked in CTC)
    pad_len = max_label_len - len(encoded)
    encoded = encoded + [-1] * pad_len
    return encoded


def decode_label(encoded):
    """Decode integer array back to string."""
    return ''.join(index_to_char(i) for i in encoded if i >= 0)


def load_and_preprocess_image(img_path):
    """Load image, convert to grayscale, resize to fixed height with proportional width."""
    # Read and decode
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 1])
    
    # Resize to fixed height, proportional width
    orig_shape = tf.cast(tf.shape(img), tf.float32)
    scale = IMG_HEIGHT / orig_shape[0]
    new_width = tf.cast(orig_shape[1] * scale, tf.int32)
    new_width = tf.maximum(new_width, 1)
    
    img = tf.image.resize(img, [IMG_HEIGHT, new_width])
    
    # Pad or crop to fixed width
    img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
    
    # Normalize to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    
    return img


def create_synthtiger_dataset(
    root_dir,
    gt_file='gt.txt',
    max_label_len=25,
    batch_size=32,
    shuffle=True,
    subset_size=None
):
    """
    Create a tf.data.Dataset from SynthTiger directory.
    
    Args:
        root_dir: Path to extracted SynthTiger data (contains gt.txt and images/)
        gt_file: Ground truth filename
        max_label_len: Maximum label length for padding
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        subset_size: If set, only use first N samples (for testing)
    
    Returns:
        tf.data.Dataset yielding (image, label, input_length, label_length)
    """
    gt_path = os.path.join(root_dir, gt_file)
    
    # Read ground truth file
    img_paths = []
    labels = []
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if subset_size and i >= subset_size:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_rel_path, label = parts[0], parts[1]
                img_paths.append(os.path.join(root_dir, img_rel_path))
                labels.append(label)
    
    print(f"Loaded {len(img_paths)} samples from {gt_path}")
    
    # Encode all labels
    encoded_labels = [encode_label(lbl, max_label_len) for lbl in labels]
    label_lengths = [min(len(lbl), max_label_len) for lbl in labels]
    
    # Calculate input length (after CNN, width is reduced)
    # Based on CRNN paper architecture:
    # - Pool1 (2,2): width / 2
    # - Pool2 (2,2): width / 4
    # - Pool3 (2,1): width unchanged (only height halved)
    # - Pool4 (2,1): width unchanged (only height halved)
    # - Conv7 (2x2, valid): width - 1
    # Result: (width // 4) - 1 = 24 for width=100
    input_length = (IMG_WIDTH // 4) - 1
    
    # Create dataset
    img_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(encoded_labels)
    input_len_ds = tf.data.Dataset.from_tensor_slices([[input_length]] * len(img_paths))
    label_len_ds = tf.data.Dataset.from_tensor_slices([[l] for l in label_lengths])
    
    # Combine all inputs
    dataset = tf.data.Dataset.zip((img_ds, label_ds, input_len_ds, label_len_ds))
    
    # Load and preprocess images
    def process_sample(img_path, label, input_len, label_len):
        img = load_and_preprocess_image(img_path)
        return (img, label, input_len, label_len)
    
    dataset = dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(img_paths)


def format_for_ctc_model(dataset):
    """
    Reformat dataset for the CTC training model.
    The model expects: [image, label, input_length, label_length]
    """
    def reformat(img, label, input_len, label_len):
        # Model inputs: dict with named inputs
        inputs = {
            'image_input': img,
            'label': label,
            'input_length': input_len,
            'label_length': label_len
        }
        # Dummy output (CTC loss computed inside model)
        dummy_output = tf.zeros((tf.shape(img)[0],))
        return inputs, dummy_output
    
    return dataset.map(reformat, num_parallel_calls=tf.data.AUTOTUNE)


# Quick test
if __name__ == "__main__":
    # Test with small subset
    DATA_DIR = r"E:\synthtiger_data"
    
    print(f"Character set: {CHARS}")
    print(f"Num classes (incl. blank): {BLANK_INDEX + 1}")
    
    # Check if data exists
    if os.path.exists(os.path.join(DATA_DIR, 'gt.txt')):
        ds, num_samples = create_synthtiger_dataset(
            DATA_DIR,
            max_label_len=25,
            batch_size=4,
            subset_size=10  # Just test with 10 samples
        )
        
        ds = format_for_ctc_model(ds)
        
        # Test one batch
        for inputs, outputs in ds.take(1):
            print(f"\nBatch shapes:")
            print(f"  Image: {inputs['image_input'].shape}")
            print(f"  Label: {inputs['label'].shape}")
            print(f"  Input length: {inputs['input_length'].shape}")
            print(f"  Label length: {inputs['label_length'].shape}")
    else:
        print(f"Data not yet extracted. Run extract_synthtiger.py first.")
        print(f"Looking for: {os.path.join(DATA_DIR, 'gt.txt')}")
