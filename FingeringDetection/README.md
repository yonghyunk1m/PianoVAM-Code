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
* 1. Download the whole FingeringDetection folder.
* 2. Change the directory part of line 35 of Fingeringdetection/main.py and line 20 of FingeringDetection/midicomparison.py to your home directory:
```
filepath = os.path.join(os.path.expanduser('~'),your-directory,'FingeringDetection','videocapture') #Your home directory
```
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
