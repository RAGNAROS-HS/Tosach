```
# Screen Text Snipper Project

A Windows Snipping Tool clone that captures a selected screen area via global hotkey, performs OCR to extract text, and copies it to clipboard. Option to build custom OCR (segmentation + per-char recognition) instead of Tesseract.

## Project Goals
- Global hotkey (Ctrl+Shift+S) → drag-select rectangle → OCR text → clipboard
- Cross-platform: Windows/macOS/Linux
- Custom OCR option: character segmentation + PyTorch CNN classifier
- Leverage your PyTorch/Python expertise

## Core Components

```
[Global Hotkey] → [Screen Region Selection] → [Image Preprocessing] → [OCR Pipeline] → [Clipboard]
```

### 1. Global Hotkey & Region Selection
| Tech                    | Purpose                             | Platform Notes                    |
| ----------------------- | ----------------------------------- | --------------------------------- |
| `keyboard` lib          | Register Ctrl+Shift+S               | Windows/Linux ✓; macOS → `pynput` |
| `pyautogui` + `tkinter` | Transparent overlay for drag-select | Cross-platform                    |
| `mss`                   | Faster screenshot                   | All platforms                     |

**Flow**: Hotkey → fullscreen transparent window → mouse drag → capture bbox → crop image

### 2. OCR Pipeline Options

#### Option A: Tesseract (Baseline - 1 day)
```
pip install pytesseract pillow pyperclip
# Preprocess: grayscale → pytesseract.image_to_string() → clipboard
```

#### Option B: Custom OCR (Project - 2-4 weeks)
**Two sub-modules**:

**a) Character Segmentation**
```
Image → Binarize (Otsu) → Connected Components (cv2.findContours())
→ Filter by aspect ratio → Sort left-to-right → [char1_img, char2_img, ...]
```

**b) Per-Character Recognition**
```
PyTorch CNN (ResNet18/LeNet) → EMNIST-trained → 62 classes (A-Z,a-z,0-9,punct)
Each char_img → model.forward() → argmax → char
''.join(chars) → clipboard
```

## Tech Stack

| Category  | Libraries                                                    | Notes              |
| --------- | ------------------------------------------------------------ | ------------------ |
| Hotkeys   | `keyboard`, `pynput`                                         | Global hooks       |
| Capture   | `pyautogui`, `mss`, `Pillow`                                 | Screenshot + crop  |
| GUI       | `tkinter`                                                    | Selection overlay  |
| OCR       | `pytesseract` **OR** `torch`, `torchvision`, `opencv-python` | Baseline vs custom |
| Clipboard | `pyperclip`                                                  | Cross-platform     |
| Packaging | `PyInstaller`                                                | .exe / .app        |

## Installation Matrix

| OS          | Tesseract                                           | Python packages                                                                       |
| ----------- | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Windows** | GitHub installer → `C:\Program Files\Tesseract-OCR` | `pip install keyboard pyautogui pillow pytesseract pyperclip mss opencv-python torch` |
## Data for Custom OCR Training
```
Augmentations: noise, blur, skew, font variations
Classes: 62 (A-Z, a-z, 0-9, common punctuation)
```

## Development Timeline
```
Week 1: Hotkey + region select + Tesseract baseline ✅
Week 2: Character segmentation (OpenCV contours)
Week 3: PyTorch char classifier + training
Week 4: Integration, polish, PyInstaller packaging
```

## Key Files Structure
```
sniptext/
├── main.py              # Hotkey + app loop
├── ocr/
│   ├── tesseract_ocr.py # Baseline
│   ├── segmenter.py     # OpenCV contours
│   └── char_model.py    # PyTorch CNN
├── gui/
│   └── selector.py      # Tkinter drag rect
├── train_char_model.py  # Training script
└── requirements.txt
```

## Success Metrics
- [ ] Hotkey triggers selector instantly
- [ ] OCR accuracy >90% on clean UI text
- [ ] Packaged as standalone .exe


biglam/early_printed_books_font_detection
linxy/LaTeX_OCR\
rootsautomation/pubmed-ocr
echo840/OCRBench


this is best probably: https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset
or this for handwriting:https://www.kaggle.com/datasets/crawford/emnist
