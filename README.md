

# PianoVAM-Code
_This Repo is currently Under Construction! (Sep 13, 2025)_

[**[PianoVAM Dataset (HuggingFace)]**](https://huggingface.co/PianoVAM/)
[**[PianoVAM Description (GitHub Page)]**](https://yonghyunk1m.github.io/PianoVAM/)

This repository provides the implementation for the ISMIR 2025 paper, "[PianoVAM: A Multimodal Piano Performance Dataset](https://arxiv.org/abs/2509.08800)," and its associated LBD paper, "[Two Web Toolkits for Multimodal Piano Performance Dataset Acquisition and Fingering Annotation](https://arxiv.org/abs/2509.15222)."

It includes the two primary web toolkits used to collect and process the PianoVAM dataset, as well as scripts to reproduce the benchmark experiments described in the paper.

## 🎹 About the PianoVAM Dataset

**PianoVAM** is a comprehensive piano performance dataset that includes synchronized video, audio, MIDI, hand landmarks, fingering labels, and rich metadata. The dataset consists of 106 solo piano recordings from 10 amateur performers, totaling approximately 21 hours of content. The data was collected using a Disklavier piano under realistic practice conditions.

## 🛠️ Toolkits

This repository offers two Graphical User Interface (GUI) toolkits that support the entire pipeline, from multimodal dataset acquisition to annotation.

### 1. PiaRec: GUI for Data Acquisition

**PiaRec** is a system designed to automate the synchronized acquisition of piano performance data, including audio, video, MIDI, and associated metadata.

  * **Key Features**:
      * A web dashboard built with Python and Streamlit.
      * QR code-based control system for initiating recording, stopping, and user identification.
      * Automated control of external software like Logic Pro and OBS Studio to eliminate manual synchronization errors.
      * Precise, automated alignment of data streams in post-processing by cross-correlating audio sources.

### 2. ASDF: GUI for Fingering Annotation

**ASDF (semi-Automated System for Detecting Fingering)** is a toolkit for the efficient annotation of piano fingering from captured video data.

  * **Key Features**:
      * Supports a hybrid workflow combining an automated fingering detection algorithm with human verification.
      * Allows users to calibrate the keyboard area within the video and extract hand skeleton data using MediaPipe Hands.
      * The algorithm automatically suggests likely fingering candidates for each note.
      * An interactive interface highlights notes requiring manual review, allowing users to visually verify and easily assign or correct fingering labels while watching the video.

## 📂 Repository Structure

```
PianoVAM-Code/
├── FingeringDetection/   # Code for the ASDF fingering annotation toolkit
├── PreProcessing/       # Data pre-processing (dataset download, audio-MIDI alignment, etc.)
│   ├── Dataset-Download/
│   ├── Audio-MIDI-Alignment/
│   └── Audio-Loudness-Normalization/
├── Transcription/       # Code to reproduce the piano transcription benchmarks from the paper
├── setup.sh             # Environment setup
├── download.sh          # Dataset download launcher
└── README.md
```

## 🚀 Getting Started

### Environment Setup

**Option 1: setup.sh (recommended)** — Automatically creates conda environment `pianoVAM` and installs dependencies

```bash
./setup.sh
conda activate pianoVAM
```

**Option 2: Manual installation** — All required libraries are listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

> **Note:** PreProcessing/Audio-MIDI-Alignment requires `fluidsynth` to be installed on your system.  
> Ubuntu/Debian: `sudo apt-get install fluidsynth fluid-soundfont-gm`

### Usage

#### Data Acquisition with PiaRec

1.  Launch the PiaRec Streamlit application.
2.  Register user information and generate QR codes in the 'Registration' tab. (Only the first time)
3.  Enter performance metadata (e.g., composer, piece title) in the 'Record' tab.
4.  Scan the generated QR codes with the camera to start and stop the recording.

#### Fingering Annotation with ASDF

1.  Launch the ASDF Streamlit application.
2.  Load a performance video and its corresponding MIDI file.
3.  Calibrate the keyboard area in the 'Keyboard Detection' tab and extract hand data in the 'Generate Mediapipe Data' tab.
4.  Generate automated fingering candidates from the 'Pre-labeling' tab.
5.  Use the interactive interface in the 'Labeling' tab to review and correct the suggested fingerings.

#### Download PianoVAM Dataset

**Interactive mode** (run without arguments):

```bash
./download.sh
```

You will be prompted to select modalities (audio, video, midi, handskeleton, tsv), splits (train, validation, test), and output directory.

**CLI mode** (non-interactive / scripting):

```bash
./download.sh -m audio,midi      # Audio and MIDI only
./download.sh -s train           # Train split only
./download.sh -m video -s train,test
./download.sh --list             # List files without downloading
./download.sh -o my_data         # Custom output directory
```

Options: `-m` modalities, `-s` splits, `-o` output dir, `--list` dry run.

Requires the `pianoVAM` conda environment (see Environment Setup).

## 📜 License

The PianoVAM dataset is distributed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/). The code in this repository is subject to the same license.

## ✍️ Citation

If you find this dataset or code useful in your research, please cite the following papers:

**PianoVAM Dataset**:

```bibtex
@inproceedings{kim2025pianovam,
  title={PianoVAM: A Multimodal Piano Performance Dataset},
  author={Kim, Yonghyun and Park, Junhyung and Bae, Joonhyung and Kim, Kirak and Kwon, Taegyun and Lerch, Alexander and Nam, Juhan},
  booktitle={Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2025},
  address={Daejeon, South Korea}
}
```

**Toolkits (PiaRec & ASDF)**:

```bibtex
@misc{park2025toolkits,
  title={Two Web Toolkits for Multimodal Piano Performance Dataset Acquisition and Fingering Annotation},
  author={Park, Junhyung and Kim, Yonghyun and Bae, Joonhyung and Kim, Kirak and Kwon, Taegyun and Lerch, Alexander and Nam, Juhan},
  howpublished={Late-Breaking Demo of the 26th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2025}
}
```

## 🙏 Acknowledgments

We sincerely appreciate the members of the [KAIST Music and Audio Computing Lab (KAIST MAC Lab)](https://mac.kaist.ac.kr/) and the [PIAST (KAIST Piano club)](https://www.youtube.com/c/PIASTKAIST) who participated as performers in the dataset acquisition. This research was supported by the National Research Foundation of Korea (NRF) funded by the Korea Government (MSIT).
