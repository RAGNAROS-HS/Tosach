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
    import time
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
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
    
    print("=" * 60)
    print("CRNN Training Configuration")
    print("=" * 60)
    print(f"Character set size: {len(CHARS)} characters")
    print(f"Num classes (with blank): {num_classes}")
    print(f"Image size: {img_height}x{img_width}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {EPOCHS}")
    print(f"Subset size: {SUBSET_SIZE if SUBSET_SIZE else 'Full dataset'}")
    
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
    
    # Count model parameters
    total_params = train_model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in train_model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {non_trainable_params:,}")
    
    train_model.summary()
    
    # Custom callback for timing and LR tracking
    class TrainingStatsCallback(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_times = []
            self.epoch_start_time = None
            self.lr_history = []
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            current_lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
            self.lr_history.append(current_lr)
    
    stats_callback = TrainingStatsCallback()
    
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
        stats_callback,
    ]
    
    # Train
    print("\n" + "=" * 60)
    print(f"Starting training for {EPOCHS} epochs...")
    print("=" * 60)
    
    training_start_time = time.time()
    start_datetime = datetime.now()
    
    history = train_model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )
    
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # Save final model
    crnn_body.save('crnn_body_final.keras')
    
    # Build and save inference model
    infer_model = build_inference_model(crnn_body)
    infer_model.save('crnn_inference.keras')
    
    # ==================== TRAINING STATISTICS ====================
    print("\n" + "=" * 60)
    print("TRAINING STATISTICS")
    print("=" * 60)
    
    # Time statistics
    epochs_completed = len(history.history['loss'])
    avg_epoch_time = np.mean(stats_callback.epoch_times) if stats_callback.epoch_times else 0
    avg_step_time = avg_epoch_time / steps_per_epoch if steps_per_epoch > 0 else 0
    
    print("\n📊 TIME STATISTICS")
    print("-" * 40)
    print(f"  Training started:  {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Training finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"  Epochs completed: {epochs_completed}/{EPOCHS}")
    print(f"  Avg time/epoch: {avg_epoch_time:.2f}s")
    print(f"  Avg time/step: {avg_step_time*1000:.2f}ms")
    print(f"  Total steps: {epochs_completed * steps_per_epoch:,}")
    
    # Loss statistics
    losses = history.history['loss']
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)
    min_loss_epoch = losses.index(min_loss) + 1
    loss_improvement = ((initial_loss - final_loss) / initial_loss) * 100
    
    print("\n📉 LOSS STATISTICS")
    print("-" * 40)
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Best loss:    {min_loss:.4f} (epoch {min_loss_epoch})")
    print(f"  Worst loss:   {max_loss:.4f}")
    print(f"  Improvement:  {loss_improvement:.2f}%")
    print(f"  Loss per epoch: {[f'{l:.4f}' for l in losses]}")
    
    # Learning rate statistics
    if stats_callback.lr_history:
        print("\n📈 LEARNING RATE HISTORY")
        print("-" * 40)
        print(f"  Initial LR: {stats_callback.lr_history[0]:.6f}")
        print(f"  Final LR:   {stats_callback.lr_history[-1]:.6f}")
        lr_reductions = sum(1 for i in range(1, len(stats_callback.lr_history)) 
                           if stats_callback.lr_history[i] < stats_callback.lr_history[i-1])
        print(f"  LR reductions: {lr_reductions}")
    
    # Dataset statistics
    print("\n📁 DATASET STATISTICS")
    print("-" * 40)
    print(f"  Training samples: {num_samples:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total batches processed: {epochs_completed * steps_per_epoch:,}")
    print(f"  Samples processed: {epochs_completed * num_samples:,}")
    
    # Model files saved
    print("\n💾 SAVED MODELS")
    print("-" * 40)
    print("  - crnn_model_best.keras (best checkpoint)")
    print("  - crnn_body_final.keras (final CRNN body)")
    print("  - crnn_inference.keras (inference model with softmax)")
    
    # ==================== VISUALIZATION ====================
    print("\n📊 Generating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CRNN Training Statistics', fontsize=14, fontweight='bold')
    
    # Plot 1: Loss over epochs
    ax1 = axes[0, 0]
    ax1.plot(range(1, epochs_completed + 1), losses, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=min_loss, color='g', linestyle='--', alpha=0.7, label=f'Best: {min_loss:.4f}')
    ax1.fill_between(range(1, epochs_completed + 1), losses, alpha=0.3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate over epochs
    ax2 = axes[0, 1]
    if stats_callback.lr_history:
        ax2.plot(range(1, epochs_completed + 1), stats_callback.lr_history, 'r-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No LR data', ha='center', va='center')
    
    # Plot 3: Time per epoch
    ax3 = axes[1, 0]
    if stats_callback.epoch_times:
        ax3.bar(range(1, epochs_completed + 1), stats_callback.epoch_times, color='orange', alpha=0.7)
        ax3.axhline(y=avg_epoch_time, color='r', linestyle='--', label=f'Avg: {avg_epoch_time:.1f}s')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Time per Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss improvement bar chart
    ax4 = axes[1, 1]
    epoch_improvements = [0] + [losses[i-1] - losses[i] for i in range(1, len(losses))]
    colors = ['green' if x > 0 else 'red' for x in epoch_improvements]
    ax4.bar(range(1, epochs_completed + 1), epoch_improvements, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Reduction')
    ax4.set_title('Loss Improvement per Epoch')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_filename}")
    
    plt.show()
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best loss achieved: {min_loss:.4f} at epoch {min_loss_epoch}")
    print(f"Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"Plot saved to: {plot_filename}")
