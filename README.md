# 🎙️ Audio Watermarking for Deepfake Detection

A deep-learning pipeline that embeds imperceptible watermarks into audio files across multiple benchmark datasets, enabling reliable detection of AI-generated (deepfake) speech.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Notes](#notes)

---

## Overview

This project implements an **audio watermarking system** designed to help combat audio deepfakes. It processes large-scale speech datasets — both real and synthesized — and embeds hidden watermarks into the audio signals. These watermarks can later be detected to verify whether an audio clip originated from a known, trusted source.

The pipeline is designed to run on **Kaggle** (or any GPU-enabled environment) and supports CUDA acceleration via PyTorch.

---

## ✨ Features

- ✅ Supports multiple benchmark anti-spoofing datasets
- ✅ GPU-accelerated processing with PyTorch & torchaudio
- ✅ Imperceptible watermark embedding (inaudible to humans)
- ✅ Handles both real and synthetic (deepfake) audio
- ✅ Outputs organized per-dataset for clean downstream analysis
- ✅ Visualization support via Matplotlib

---

## 🗂️ Project Structure

```
audio-watermarking-deepfake/
│
├── README.md                        # This file
├── audio_watermarking.py            # Main pipeline script
│
├── outputs/                         # All watermarked audio outputs (auto-created)
│   ├── asvspoof2021_df/             # Watermarked ASVspoof 2021 DF files
│   ├── asvspoof2019_la/             # Watermarked ASVspoof 2019 LA files
│   ├── for_original/                # Watermarked Fake-or-Real files
│   └── audio_deepfake/              # Watermarked AudioDeepfake files
│
└── (Kaggle Input Datasets)          # Read-only Kaggle inputs (not committed)
    ├── avsspoof-2021/
    │   ├── ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/
    │   ├── ASVspoof2021_DF_eval_part01/ASVspoof2021_DF_eval/flac/
    │   └── ASVspoof2021_DF_eval_part02/ASVspoof2021_DF_eval/flac/
    │
    ├── asvpoof-2019-dataset/
    │   └── LA/LA/
    │       ├── ASVspoof2019_LA_train/flac/
    │       ├── ASVspoof2019_LA_dev/flac/
    │       └── ASVspoof2019_LA_eval/flac/
    │
    ├── the-fake-or-real-dataset/
    │   └── for-original/for-original/
    │       ├── training/real/
    │       ├── training/fake/
    │       ├── testing/real/
    │       ├── testing/fake/
    │       ├── validation/real/
    │       └── validation/fake/
    │
    └── audio-deepfake-detection-dataset/
        ├── real_samples/
        ├── FlashSpeech/
        ├── NaturalSpeech3/
        ├── OpenAI/
        ├── PromptTTS2/
        ├── VALLE/
        ├── VoiceBox/
        ├── seedtts_files/
        └── xTTS/
```

---

## 📦 Datasets

The pipeline is configured for **4 major audio deepfake benchmarks**:

| Dataset | Description | Split Coverage |
|---|---|---|
| **ASVspoof 2021 DF** | Deepfake evaluation set (3 parts) | Eval only |
| **ASVspoof 2019 LA** | Logical access spoofing (train/dev/eval) | Train, Dev, Eval |
| **The Fake-or-Real (FoR)** | Real vs. synthesized speech | Train, Test, Validation |
| **AudioDeepfake** | Multi-system TTS deepfakes (8 TTS systems + real) | All systems |

### AudioDeepfake TTS Systems Included:
`FlashSpeech` · `NaturalSpeech3` · `OpenAI` · `PromptTTS2` · `VALLE` · `VoiceBox` · `SeedTTS` · `xTTS`

---

## 🛠️ Requirements

```txt
torch>=2.0.0
torchaudio>=2.0.0
numpy
pandas
matplotlib
soundfile
```

> **Python version:** 3.9+
> **CUDA:** Optional but strongly recommended for large datasets

---

## ⚙️ Installation

### 1. Clone the repository

```bash
https://github.com/Mijan2001/Audio-Watermarking-deepface-detection-python-code-.git
cd audio-watermarking-deepfake
```

### 2. Install dependencies

```bash
pip install torch torchaudio numpy pandas matplotlib soundfile
```

### 3. (Kaggle) Add datasets

On Kaggle, add the following datasets to your notebook via **Add Data**:

- `mohammedabdeldayem/avsspoof-2021`
- `awsaf49/asvpoof-2019-dataset`
- `mohammedabdeldayem/the-fake-or-real-dataset`
- `adarshsingh0903/audio-deepfake-detection-dataset`

---

## 🚀 Usage

### Run the full pipeline

```bash
python audio_watermarking.py
```

On startup, the script will:

1. Detect whether a **CUDA GPU** is available and print the active device
2. Register all 4 datasets and verify that each input directory exists
3. Auto-create all output directories under `/kaggle/working/outputs/`
4. Process and watermark audio files dataset by dataset

### Expected console output

```
Device : cuda
Datasets registered: 4

  [ASVspoof2021-DF]  →  /kaggle/working/outputs/asvspoof2021_df
    /kaggle/input/.../flac  ->  True
    ...

  [ASVspoof2019-LA]  →  /kaggle/working/outputs/asvspoof2019_la
    ...

  [The Fake-or-Real]  →  /kaggle/working/outputs/for_original
    ...

  [AudioDeepfake]  →  /kaggle/working/outputs/audio_deepfake
    ...
```

---

## 📁 Output Structure

After processing, watermarked audio files are saved under `/kaggle/working/outputs/`:

```
outputs/
├── asvspoof2021_df/
│   └── *.flac          # Watermarked eval files
│
├── asvspoof2019_la/
│   └── *.flac          # Watermarked train/dev/eval files
│
├── for_original/
│   └── *.wav           # Watermarked real & fake files
│
└── audio_deepfake/
    └── *.wav           # Watermarked real & TTS-generated files
```

---

## 🔧 Configuration

All dataset paths and output directories are defined in the `DATASETS` list at the top of the script. To add a new dataset:

```python
DATASETS = [
    # ... existing datasets ...
    {
        "name":       "MyNewDataset",          # Human-readable label
        "input_dirs": [
            "/path/to/audio/split1",           # One or more input folders
            "/path/to/audio/split2",
        ],
        "output_dir": "/kaggle/working/outputs/my_new_dataset",  # Where to save output
    },
]
```

The device is auto-selected:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

To force CPU:

```python
DEVICE = torch.device("cpu")
```

---

## 🔬 How It Works

```
Raw Audio (.flac / .wav)
        │
        ▼
  Load with torchaudio
        │
        ▼
  Preprocess (resample, normalize)
        │
        ▼
  Watermark Embedding
  (frequency-domain perturbation via FFT / spread-spectrum)
        │
        ▼
  Quality Check (SNR, PESQ)
        │
        ▼
  Save Watermarked Audio → output_dir/
        │
        ▼
  Detection Phase:
  Extract watermark → verify origin
```

### Key Libraries

| Library | Role |
|---|---|
| `torch` / `torchaudio` | Audio I/O, tensor ops, GPU acceleration |
| `numpy` | Signal processing math |
| `soundfile` | Read/write audio (WAV, FLAC) |
| `matplotlib` | Spectrogram & waveform visualization |
| `pandas` | Metadata / results logging |

---

## 📝 Notes

- Input directories are **read-only** on Kaggle; the script never modifies source files.
- If an input directory does not exist (`-> False` in the output), the dataset will be skipped gracefully.
- All output directories are created automatically with `os.makedirs(..., exist_ok=True)` — no manual setup needed.
- The `DATASETS` block appears twice in the source; only the second definition is active at runtime (Python last-assignment semantics). This is safe to deduplicate.

---

## 📄 License

This project is intended for academic and research use. Please respect the individual licenses of each dataset used.

---

## 🙏 Acknowledgements

- [ASVspoof Challenge](https://www.asvspoof.org/) — for benchmark datasets
- [The Fake-or-Real Dataset](https://bil.eecs.yorku.ca/datasets/) — York University BIL Lab
- Kaggle dataset contributors: `mohammedabdeldayem`, `awsaf49`, `adarshsingh0903`
