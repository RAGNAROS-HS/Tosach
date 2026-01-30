import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Config
DATA_DIR = r'C:\Users\Hugo\Downloads\ocr\kaggle_data\training_data'
MODEL_PATH = 'ocr_model.keras'
IMG_SIZE = (28, 28)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run model_training.py first.")
        return

    print("Loading dataset to fetch class names and validation samples...")
    # We load the validation set to see how it performs on 'unseen' data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=32
    )
    
    class_names = val_ds.class_names
    print(f"Classes: {class_names}")

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Running predictions on a batch of images...")
    plt.figure(figsize=(12, 10))
    
    # Take 1 batch from the dataset
    for images, labels in val_ds.take(1):
        # Predict
        predictions = model.predict(images)
        predicted_ids = np.argmax(predictions, axis=1)
        
        # Show first 15 images in the batch
        for i in range(min(15, len(images))):
            ax = plt.subplot(3, 5, i + 1)
            
            # Convert tensor to numpy and reshape for display (removing channel dim)
            img = images[i].numpy().astype("uint8").squeeze()
            
            true_label = class_names[labels[i]]
            pred_label = class_names[predicted_ids[i]]
            confidence = 100 * np.max(tf.nn.softmax(predictions[i]))
            
            plt.imshow(img, cmap='gray')
            
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f"True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)", color=color, fontsize=9)
            plt.axis("off")
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
