# Tosach — CRNN for Scene Text Recognition

Implementation of the CRNN (Convolutional Recurrent Neural Network) architecture from [Shi et al., 2015](https://arxiv.org/abs/1507.05717) for end-to-end scene text recognition using TensorFlow/Keras.

> **Status: Work in Progress** — Architecture is implemented, dataset is prepared, training has not been run yet.

## Overview

The model recognizes text in natural scene images by combining:
- **CNN** (VGG-style) for visual feature extraction
- **Bidirectional LSTM** for sequence modeling
- **CTC loss** for sequence-to-label transcription without character-level annotations

## Project Structure

| File                    | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `train_crnn.py`         | CRNN model architecture + training pipeline             |
| `synthtiger_loader.py`  | Data loader for SynthTiger dataset (tf.data)            |
| `extract_synthtiger.py` | Memory-efficient extractor for the large SynthTiger zip |
| `study_content.txt`     | Extracted text from the reference paper                 |
| `study.pdf`             | Original paper (Shi et al., 2015)                       |
| `legacy/`               | Earlier prototype (deprecated)                          |

## Dataset

**SynthTiger v1.0** — synthetic text images with ground truth labels. The dataset is ~38GB and must be extracted before training.

```bash
pip install stream-unzip
python extract_synthtiger.py
```

## Architecture (from paper)

```
Input (W × 32 grayscale)
  → 7 Conv layers (64 → 512 maps) + BatchNorm + MaxPool
  → Map-to-Sequence
  → 2× Bidirectional LSTM (256 hidden units)
  → CTC Transcription
  → Predicted text
```

## Legacy

The `legacy/` folder contains an earlier prototype that used a simpler per-character classification approach:

- **Simple CNN** (Conv2D → MaxPool → Dense) trained on 28×28 grayscale character images from a Kaggle OCR dataset
- **Per-character pipeline**: segment individual characters → classify each with the CNN → concatenate results
- Abandoned in favor of the CRNN approach, which handles full words end-to-end without character segmentation

| File                | Purpose                                             |
| ------------------- | --------------------------------------------------- |
| `model_training.py` | CNN classifier training (62 classes: A-Z, a-z, 0-9) |
| `data_process.py`   | Sorts character images into class folders           |
| `run_inference.py`  | Runs predictions on validation samples              |
| `idea.md`           | Original project plan                               |

