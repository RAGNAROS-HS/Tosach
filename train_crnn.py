import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def build_crnn(img_width, img_height, num_classes):
    """
    Builds the CRNN architecture as described in the study.
    
    Args:
        img_width: Width of the input image (scaled to be proportional, min 100)
        img_height: Height of the input image (fixed at 32)
        num_classes: Number of characters in alphabet + 1 (blank) for CTC
    """
    
    # Input: Gray-scale image (H x W x 1)
    # Note: Keras uses (H, W, C) format by default
    inputs = layers.Input(shape=(img_height, img_width, 1), name="image_input")
    
    # -------------------------------------------------------------------------
    # Convolutional Layers (VGG-style)
    # -------------------------------------------------------------------------
    
    # Layer 1: Conv 64, 3x3, s1, p1 (same)
    # "m" in table stands for number of maps (filters)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    
    # Layer 2: Conv 128, 3x3, s1, p1 (same)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
    
    # Layer 3: Conv 256, 3x3, s1, p1 (same)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3')(x)
    
    # Layer 4: Conv 256, 3x3, s1, p1 (same)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4')(x)
    
    # MaxPool 3: Window: 1x2 (HxW?), Stride: 2. 
    # Paper: "1x2 sized rectangular pooling windows... yields feature maps with larger width"
    # This implies we pool Height(2) but keep Width(1). 
    # Keras pool_size=(row, col). So (2, 1).
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool3')(x)
    
    # Layer 5: Conv 512, 3x3, s1, p1 (same) + Batch Normalization
    x = layers.Conv2D(512, (3, 3), padding='same', use_bias=False, name='conv5')(x)
    x = layers.BatchNormalization(name='batchnorm5')(x)
    x = layers.Activation('relu', name='relu5')(x)
    
    # Layer 6: Conv 512, 3x3, s1, p1 (same) + Batch Normalization
    x = layers.Conv2D(512, (3, 3), padding='same', use_bias=False, name='conv6')(x)
    x = layers.BatchNormalization(name='batchnorm6')(x)
    x = layers.Activation('relu', name='relu6')(x)
    
    # MaxPool 4: Window: 1x2, s2
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool4')(x)
    
    # Layer 7: Conv 512, 2x2, s1, p0 (valid)
    # "Map-to-Sequence" usually happens after this.
    # With input height 32:
    # pool1 -> 16
    # pool2 -> 8
    # pool3 (2, 1) -> 4
    # pool4 (2, 1) -> 2
    # Conv7 (2x2 valid) on height 2 -> height 1.
    x = layers.Conv2D(512, (2, 2), padding='valid', activation='relu', name='conv7')(x)
    
    # -------------------------------------------------------------------------
    # Map-to-Sequence
    # -------------------------------------------------------------------------
    # Current shape: (Batch, H=1, W', C=512)
    # We need to squeeze H dimension.
    # Note: Keras shape excludes batch. So (1, W', 512).
    # We want (W', 512) for RNN input (TimeSteps, Features).
    
    # Reshape to (Batch, TimeSteps, Features) = (Batch, W', 512)
    # Calculate target width. If None is used for width in input, we can use Reshape dynamic.
    # However, Reshape(-1, 512) works if we know the last dim.
    # Or Squeeze logic.
    x = layers.Reshape((-1, 512), name='map_to_sequence')(x)
    
    # -------------------------------------------------------------------------
    # Recurrent Layers (Bidirectional LSTMs)
    # -------------------------------------------------------------------------
    
    # Bidirectional-LSTM 1: #hidden units: 256
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name='bilstm1')(x)
    
    # Bidirectional-LSTM 2: #hidden units: 256
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name='bilstm2')(x)
    
    # -------------------------------------------------------------------------
    # Transcription Layer
    # -------------------------------------------------------------------------
    
    # Output layer: Project to number of classes
    # The output is a probability distribution for each frame (TimeStep)
    outputs = layers.Dense(num_classes, activation='softmax', name='prediction')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CRNN")
    return model

if __name__ == "__main__":
    # Example usage
    # Height is fixed at 32 according to paper.
    # Width can be variable (None).
    # num_classes = (Alphabet size) + 1 (blank). Example: 37
    
    model = build_crnn(img_width=None, img_height=32, num_classes=37)
    model.summary()
    
    print("\nModel created successfully using TensorFlow/Keras.")
