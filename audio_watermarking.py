# =============================================================================
# Audio Watermarking for Deepfake Detection
# =============================================================================



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import math
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as TAF
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf


# =============================================================================
# §0  DATASET CONFIGURATIONS
# =============================================================================

DATASETS = [
    {
        "name":       "ASVspoof2021-DF",
        "input_dirs": [
            "/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac",
            "/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021/ASVspoof2021_DF_eval_part01/ASVspoof2021_DF_eval/flac",
            "/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021/ASVspoof2021_DF_eval_part02/ASVspoof2021_DF_eval/flac",
        ],
        "output_dir": "/kaggle/working/outputs/asvspoof2021_df",
    },
    {
        "name":       "ASVspoof2019-LA",
        "input_dirs": [
            "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac",
            "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev/flac",
            "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_eval/flac",
        ],
        "output_dir": "/kaggle/working/outputs/asvspoof2019_la",
    },
    {
        "name":       "The Fake-or-Real",
        "input_dirs": [
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/training/real",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/training/fake",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/testing/real",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/testing/fake",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/validation/real",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/validation/fake",
        ],
        "output_dir": "/kaggle/working/outputs/for_original",
    },
    {
        "name":       "AudioDeepfake",
        "input_dirs": [
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/real_samples",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/FlashSpeech",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/NaturalSpeech3",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/OpenAI",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/PromptTTS2",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/VALLE",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/VoiceBox",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/seedtts_files",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/xTTS",
        ],
        "output_dir": "/kaggle/working/outputs/audio_deepfake",
    },
]

# Create all output directories upfront
for ds in DATASETS:
    os.makedirs(ds["output_dir"], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")
print(f"Datasets registered: {len(DATASETS)}")
for ds in DATASETS:
    print(f"\n  [{ds['name']}]  →  {ds['output_dir']}")
    for d in ds["input_dirs"]:
        print(f"    {d}  ->  {os.path.exists(d)}")



# =============================================================================
# §0  DATASET CONFIGURATIONS
# =============================================================================

DATASETS = [
    {
        "name":       "ASVspoof2021-DF",
        "input_dirs": [
            "/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac",
            "/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021/ASVspoof2021_DF_eval_part01/ASVspoof2021_DF_eval/flac",
            "/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021/ASVspoof2021_DF_eval_part02/ASVspoof2021_DF_eval/flac",
        ],
        "output_dir": "/kaggle/working/outputs/asvspoof2021_df",
    },
    {
        "name":       "ASVspoof2019-LA",
        "input_dirs": [
            "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac",
            "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev/flac",
            "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_eval/flac",
        ],
        "output_dir": "/kaggle/working/outputs/asvspoof2019_la",
    },
    {
        "name":       "The Fake-or-Real",
        "input_dirs": [
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/training/real",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/training/fake",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/testing/real",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/testing/fake",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/validation/real",
            "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-original/for-original/validation/fake",
        ],
        "output_dir": "/kaggle/working/outputs/for_original",
    },
    {
        "name":       "AudioDeepfake",
        "input_dirs": [
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/real_samples",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/FlashSpeech",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/NaturalSpeech3",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/OpenAI",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/PromptTTS2",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/VALLE",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/VoiceBox",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/seedtts_files",
            "/kaggle/input/datasets/adarshsingh0903/audio-deepfake-detection-dataset/xTTS",
        ],
        "output_dir": "/kaggle/working/outputs/audio_deepfake",
    },
]

# Create all output directories upfront
for ds in DATASETS:
    os.makedirs(ds["output_dir"], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")
print(f"Datasets registered: {len(DATASETS)}")
for ds in DATASETS:
    print(f"\n  [{ds['name']}]  →  {ds['output_dir']}")
    for d in ds["input_dirs"]:
        print(f"    {d}  ->  {os.path.exists(d)}")

