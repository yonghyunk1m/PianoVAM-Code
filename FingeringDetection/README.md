# How to use our semi-Automatic System for Detecting Fingering (ASDF) for PianoVAM

* 0. Recommended Prerequisites: Python 3.9.21 with
```
miditok 3.0.5.post1
shapely 2.0.7
geopandas 1.0.1
mido 1.3.3
pretty_midi-0.2.10
streamlit 1.37.1
streamlit_image_coordinates 0.2.0
numpy 1.24.1
```
with your recorded MIDI files in FingeringDetection/midiconvert folder and video files in FingeringdDetection/videocapture folder.
We recommend each video within no longer than 30 minutes if your MediaPipe environment does not provide GPU support.

## Data Folder Structure Setup

Before using the system, you need to prepare the following folder structure:

```
FingeringDetection/
戍式式 midiconvert/          # Place your MIDI files here
弛   戌式式 your_recording.mid
戍式式 videocapture/         # Place your video files here
弛   戌式式 your_recording.mp4
戍式式 ASDF.py              # Main application file
戍式式 main.py
戍式式 midicomparison.py
戍式式 hand_landmarker.task
戌式式 other files...
```

**Important Notes:**
- MIDI and video files should have the same base name (e.g., `song1.mid` and `song1.mp4`)
- Video files must be in MP4 format
- MIDI files must be in .mid format
- The system will automatically create additional subfolders and data files during processing

* 1. Download the whole FingeringDetection folder.
* 2. ~~Change the directory part...~~ **No longer needed!** File paths are now automatically detected based on script location.
* 3. Run:
```
streamlit run ./FingeringDetection/ASDF.py
```
* 4. If your MIDI recording environment uses Logic's Smart Tempo, you should unify the tempo information by using 'Delete Smart Tempo' tab.
* 5. For each video, you must specify your keyboard location information by using 'Keyboard detection' tab and then 'Keyboard distortion' tab.
* 6. Before generating fingering information, we have to preprocess the data with MediaPipe, by using 'Generate MediaPipe data'.
* 7. Now, we can generate fingering information by using 'Pre-finger labeling' tab.
* 8. Finally, for the unlabeled notes, we can manually label the fingering of those notes in the 'Label' tab.
* 9. If you want to label every fingering numbers of the video by yourself, use 'Groundtruth annotation' tab.
