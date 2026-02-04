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

    inputs = layers.Input(shape=(img_height, img_width, 1), name="image_input")
    
    x = layers.Conv2D(64, (3, 3), padding= "same", strides= 1, activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    
    x = layers.Conv2D(128, (3, 3), padding= "same", strides= 1, activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)

    x = layers.Conv2D(256, (3, 3), padding= "same", strides= 1, activation='relu', name='conv3')(x)
    x = layers.Conv2D(256, (3, 3), padding= "same", strides= 1, activation='relu', name='conv4')(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='pool3')(x)

    x = layers.Conv2D(512, (3, 3), padding= "same", strides= 1, activation="relu")(x)         
    x = layers.BatchNormalization()(x)                                           
    x = layers.Conv2D(512, (3, 3), padding= "same", strides= 1, activation="relu")(x)         
    x = layers.BatchNormalization()(x)                                           
    x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)                
    
    x = layers.Conv2D(512, (2, 2), padding= 0, strides= 1, use_bias=False, name='conv6')(x)


 
    def map_to_sequence(t):
        # t: (B, H=1, W, C) -> (B, W, C)
        t = tf.squeeze(t, axis=1)
        return t
    x = layers.Lambda(map_to_sequence, name="map_to_sequence")(x)

    x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True),
            merge_mode="concat",
            name="bilstm_1",
        )(x)

    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True),
        merge_mode="concat",
        name="bilstm_2",
    )(x)
    
    y_pred = layers.Dense(num_classes, activation="linear", name="logits")(x)
    y_softmax = layers.Activation("softmax", name="y_pred")(y_pred)

    # Model for inference (CTC decoding done outside the model)
    model = models.Model(inputs=input_img, outputs=y_softmax, name="crnn_ctc")
    return model



if __name__ == "__main__":
    # Example usage
    # Height is fixed at 32 according to paper.
    # Width can be variable (None).
    # num_classes = (Alphabet size) + 1 (blank). Example: 37
    
    model = build_crnn(img_width=100, img_height=32, num_classes=37)
    model.summary()
    
    print("\nModel created successfully using TensorFlow/Keras.")
