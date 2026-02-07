import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers

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
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool3')(x)

    x = layers.Conv2D(512, (3, 3), padding= "same", strides= 1, activation="relu")(x)         
    x = layers.BatchNormalization()(x)                                           
    x = layers.Conv2D(512, (3, 3), padding= "same", strides= 1, activation="relu")(x)         
    x = layers.BatchNormalization()(x)                                           
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool4')(x)                
    
    x = layers.Conv2D(512, (2, 2), padding="valid", strides=1, use_bias=False, name='conv6')(x)


 
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
    
    logits = layers.Dense(num_classes, activation="linear", name="logits")(x)

    model = models.Model(inputs=inputs, outputs=logits, name="crnn_body")
    return model


def ctc_loss_layer(args):
    """
    args: [y_pred, labels, input_length, label_length]
    y_pred: (B, T, C) logits or softmax probs
    labels: (B, max_label_len)
    input_length: (B, 1)
    label_length: (B, 1)
    """
    y_pred, labels, input_length, label_length = args
    # CTC expects shape (B, T, C) with time_major=False [web:6][web:9]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_ctc_train_model(img_width, img_height, num_classes, max_label_len):
    """
    Wraps CRNN with inputs for labels and lengths and outputs CTC loss.
    Use this model for training.
    """

    crnn_body = build_crnn(img_width, img_height, num_classes)

    # Inputs
    image_input = crnn_body.input  # (B, H, W, 1)
    logits = crnn_body.output      # (B, T, C)

    label_input = layers.Input(shape=(max_label_len,),
                               dtype="int32",
                               name="label")
    input_len_input = layers.Input(shape=(1,),
                                   dtype="int32",
                                   name="input_length")
    label_len_input = layers.Input(shape=(1,),
                                   dtype="int32",
                                   name="label_length")

    loss_out = layers.Lambda(
        ctc_loss_layer,
        name="ctc_loss"
    )([logits, label_input, input_len_input, label_len_input])

    train_model = models.Model(
        inputs=[image_input, label_input, input_len_input, label_len_input],
        outputs=loss_out,
        name="crnn_ctc_train",
    )

    # y_true is ignored; loss_out already contains the per-sample loss
    train_model.compile(
        optimizer= optimizers.Adadelta(learning_rate=1.0, rho=0.9, epsilon=1e-6),
        loss=lambda y_true, y_pred: y_pred,
    )
    return train_model, crnn_body


def ctc_decode_greedy(y_pred, blank_index):
    """
    Greedy CTC decoding (lexicon-free):
    y_pred: (B, T, C) softmax probs
    returns: list of lists of label indices (no blanks, collapsed)
    """
    path = np.argmax(y_pred, axis=-1)  # (B, T)
    decoded = []
    for seq in path:
        prev = None
        out = []
        for p in seq:
            if p != blank_index and p != prev:
                out.append(int(p))
            prev = p
        decoded.append(out)
    return decoded


def build_inference_model(crnn_body):
    """
    Build an inference model that outputs softmax probabilities.
    Decoding (CTC argmax/collapse or beam search) is done outside. [file:1][web:4]
    """
    inputs = crnn_body.input
    logits = crnn_body.output
    y_softmax = layers.Activation("softmax", name="softmax")(logits)
    infer_model = models.Model(inputs=inputs, outputs=y_softmax, name="crnn_ctc_infer")
    return infer_model


if __name__ == "__main__":
    import os
    from synthtiger_loader import (
        create_synthtiger_dataset,
        format_for_ctc_model,
        BLANK_INDEX,
        CHARS,
        IMG_HEIGHT,
        IMG_WIDTH
    )
    
    # Configuration
    DATA_DIR = r"E:\synthtiger_data"
    MAX_LABEL_LEN = 93
    BATCH_SIZE = 32
    EPOCHS = 10
    SUBSET_SIZE = 10000  # Set to e.g. 10000 for quick testing, None for full dataset
    
    # Model parameters
    img_height = IMG_HEIGHT  # 32
    img_width = IMG_WIDTH    # 100
    num_classes = BLANK_INDEX + 1  # alphabet + blank
    
    print(f"Character set: {CHARS}")
    print(f"Num classes: {num_classes}")
    print(f"Image size: {img_height}x{img_width}")
    
    # Check if data exists
    gt_path = os.path.join(DATA_DIR, 'gt.txt')
    if not os.path.exists(gt_path):
        print(f"\nERROR: Dataset not found at {DATA_DIR}")
        print("Run extract_synthtiger.py first to extract the dataset.")
        exit(1)
    
    # Create dataset
    print(f"\nLoading dataset from {DATA_DIR}...")
    train_ds, num_samples = create_synthtiger_dataset(
        DATA_DIR,
        max_label_len=MAX_LABEL_LEN,
        batch_size=BATCH_SIZE,
        shuffle=True,
        subset_size=SUBSET_SIZE
    )
    train_ds = format_for_ctc_model(train_ds)
    
    print(f"Dataset ready: {num_samples} samples")
    steps_per_epoch = num_samples // BATCH_SIZE
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Build model
    print("\nBuilding model...")
    train_model, crnn_body = build_ctc_train_model(
        img_width=img_width,
        img_height=img_height,
        num_classes=num_classes,
        max_label_len=MAX_LABEL_LEN,
    )
    train_model.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'crnn_model_best.keras',
            monitor='loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    
    # Train
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = train_model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )
    
    # Save final model
    crnn_body.save('crnn_body_final.keras')
    print("\nTraining complete! Models saved:")
    print("  - crnn_model_best.keras (best checkpoint)")
    print("  - crnn_body_final.keras (final CRNN body for inference)")
    
    # Build and save inference model
    infer_model = build_inference_model(crnn_body)
    infer_model.save('crnn_inference.keras')
    print("  - crnn_inference.keras (inference model with softmax)")
