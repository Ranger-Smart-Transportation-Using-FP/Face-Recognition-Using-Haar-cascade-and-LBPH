# Face-Recognition-Using-Haar-cascade-and-LBPH

# Facial Recognition CLI using OpenCV LBPH

This is a simple command-line application for face-based user registration and login using OpenCV’s Local Binary Patterns Histograms (LBPH) recognizer.

## Features

* **Register** new users by capturing face samples from a webcam
* **Train** an LBPH model on all registered faces
* **Login** by recognizing a live face and returning the user’s name
* **Pure Python** with a single dependency: `opencv-contrib-python` and `numpy`

## Repository Structure

```
├── app.py             # Main application script
├── dataset/           # Captured face images (auto-generated)
│   └── <Name>_#.jpg   # Cropped grayscale face samples
├── labels.pickle      # Persisted mapping: name → numeric ID
├── trainer.yml        # Trained LBPH model data
└── README.md          # This documentation
```

## Prerequisites

* Python 3.7+
* Webcam or USB camera

## Installation

1. Clone the repository (if applicable) or copy `app.py` to your project folder.
2. Install dependencies:

   ```bash
   pip install opencv-contrib-python numpy
   ```

## Usage

Run the script and choose between **register** and **login** modes:

```bash
python app.py
```

### 1. Register a New User

1. At the prompt, enter `register`.
2. Enter the user’s name (e.g. `Alice`).
3. A webcam window opens; a green box shows where to position your face.
4. Press **c** (or **C**) to capture each sample; by default it collects 20 samples.
5. Press **q** at any time to stop early.
6. After capturing, the LBPH model retrains automatically and saves `trainer.yml` and `labels.pickle`.

### 2. Login (Recognize a Face)

1. At the prompt, enter `login`.
2. A webcam window opens with the same box overlay.
3. Position your face; the script computes LBPH features and predicts a label.
4. If the match confidence is above the threshold, your name appears in green; otherwise, `Unknown` in red.
5. Press **q** or wait until a known face is detected to exit.

## How It Works

1. **Face Detection**: Uses OpenCV’s Haar cascade to locate faces in camera frames.
2. **Feature Extraction (LBPH)**: Computes histograms of local binary patterns on each face crop.
3. **Training**: Stores histograms and their labels in `trainer.yml`.
4. **Prediction**: Compares a live histogram against stored examples to find the closest match.

## Configuration

* **Sample Count**: Change the number of registration captures by editing `SAMPLE_COUNT` in `app.py`.
* **Threshold**: Adjust `threshold` in `recognize_user()` to fine-tune recognition sensitivity.

## Dependencies

* [OpenCV](https://pypi.org/project/opencv-contrib-python/)
* [NumPy](https://pypi.org/project/numpy/)

## License

MIT License. Feel free to use, modify, or distribute.

## Acknowledgments

* Based on OpenCV’s LBPH face recognizer example
* Inspired by simple CLI demos for computer vision tasks
