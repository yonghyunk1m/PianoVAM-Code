\section{Benchmark Results}\label{sec:transcription}

We evaluate our dataset for benchmarking in two piano transcription settings: audio-only and audio-visual. In the audio-visual setting, we specifically examine the visual modality's contribution to enhancing acoustic robustness. 

\subsection{Data Split}
To facilitate reproducibility of results, we provide information on data splits designed to meet the following criteria: 
\begin{inparaenum}[(i)]
    \item no composition appears in more than one split, and 
    \item the dataset is divided approximately into 80/10/10 percent for the training, validation, and test set, respectively, based on total duration. 
\end{inparaenum}
The resulting train/validation/test splits contain 73, 19, and 14 files, respectively. While these splits are intended to support reproducibility and comparability, we acknowledge that different experimental objectives might require different splits.

\subsection{Audio-Only Piano Transcription}
As MAESTRO is widely regarded as a standard dataset in piano transcription research, we deemed it a suitable reference point for evaluating our dataset. Accordingly, we performed a comparative analysis using the Onsets and Frames model \cite{ISMIR18Hawthorne}, following its original specifications. The model was trained on each dataset as well as on a Combined version. We utilized the model weights corresponding to the checkpoint with the lowest validation loss for inference. 

All results in Table \ref{tab:performance_comparison} are presented as F1 Scores (\%) and calculated over the entire duration of the respective test splits. The terms \textit{Note}, \textit{w/ Offset}, and \textit{w/ Vel.} refer to note evaluation with onset, with onset \& offset, and with onset \& velocity, respectively (cf. \cite{TASLP21Kong}). All four evaluation metrics, including \textit{Frame}, were computed using the \textit{mir\_eval} package. By established conventions in transcription research, offset timings were adjusted to the pedal-release time if the sustain pedal remained engaged. 

\begin{table}[!t]
\centering
\small
\label{tab:performance_comparison}
\begin{tabular*}{\columnwidth}{l@{\extracolsep{\fill}}cccc}
%\begin{tabular}{l c c c c}
\toprule
\textbf{Train Dataset} & \textbf{Note} & \textbf{w/ Offset} & \textbf{w/ Vel.} & \textbf{Frame} \\
\midrule
    MAESTROv3 & 93.4 & 62.3 & 90.3 & 78.2 \\
    PianoVAM & \underline{\textbf{95.8}} & 60.4 & \underline{\underline{\textbf{93.9}}} & 80.0 \\
    Combined & \underline{95.2} & \underline{\underline{\textbf{73.5}}} & \underline{93.0} & \underline{\underline{\textbf{86.9}}} \\
\bottomrule
\end{tabular*}
\caption{F1 scores on the PianoVAM test split. Bold: highest; Underline: significantly higher than the lowest; Double-line: significantly higher over both others. ($p < .0167$)}
\end{table}

A Friedman test revealed statistically significant differences across training conditions for all F1 score metrics: \textit{Note} ($\chi^2$(2) = 22.39, $p$ < .001), \textit{w/ Offset} ($\chi^2$(2) = 18.58, $p$ < .001), \textit{w/ Vel.} ($\chi^2$(2) = 23.80, $p$ < .001), and \textit{Frame} ($\chi^2$(2) = 18.43, $p$ < .001). Post-hoc Wilcoxon signed-rank tests with Bonferroni correction ($\alpha = .0167$) showed that both the PianoVAM- and Combined-trained models significantly outperformed the MAESTROv3-trained model in \textit{Note} F1 ($p = .00147$), \textit{w/ Velocity} F1 ($p = .00147$), and \textit{Frame} F1 ($p = .00012$ and $p = .00024$, respectively). For the \textit{w/ Offset} metric, the Combined-trained model also significantly outperformed both MAESTROv3 ($p = .00037$) and PianoVAM ($p = .00012$), while no significant difference was observed between MAESTROv3 and PianoVAM ($p = .50675$). Although PianoVAM slightly outperformed the Combined model in both \textit{Note} ($p = .05974$) and \textit{w/ Velocity} ($p = .00370$) F1 scores, only the latter remained statistically significant.

\subsection{Audio-Visual Piano Transcription}

Various approaches have been explored for piano transcription when both audio and video are available. For instance, \cite{CJE15Wan, DAFx21Wang} proposed a method that enhances the output of an audio-only AMT system by incorporating visual information, while \cite{ICASSPW23Li, TASLP24Li} utilized both modalities jointly to improve onset prediction. 

For this study, we focus on examining how visual information can be used to improve transcription performance under suboptimal recording conditions. Specifically, we implement a simple post-processing pipeline that refines MIDI outputs from an audio-only AMT model by using top-view video, estimated piano keyboard corner coordinates, and hand skeletons detected with MediaPipe Hands \cite{arXiv20Zhang}. This process enables the elimination of physically implausible notes by referencing visual evidence, thereby improving onset precision. The full implementation and additional details are available on GitHub\footref{github-link}, and a brief overview follows.

The pipeline begins by extracting onset events from the predicted MIDI file. For each onset, the nearest video frame is retrieved, and hand landmarks are predicted by \cite{arXiv20Zhang}. Each video frame's timestamp is defined as the midpoint of the time interval it covers. If no hand is detected, the corresponding onset is skipped. When both hands are detected, a perspective transformation is applied using the keyboard corner metadata to produce a normalized rectangular image ($H:W=125:1024$), which maintains the standard height-to-width ratio ($1:8.147$) of an 88-key piano. The same transformation is applied to the predicted hand landmarks. Assuming that the 52 white keys are evenly spaced, the algorithm estimates which white key region each fingertip corresponds to, based on its transformed x-coordinate. To account for possible errors in hand landmark detection, multiple candidate keys are considered for each fingertip, with a tunable threshold determining the candidate range ($\pm2$ white keys in our experiment). The final set of valid pitch candidates is obtained by intersecting all fingertip candidate sets. For each onset, if the pitch predicted by the audio-only AMT model falls within this candidate set, the note is retained; otherwise, it is discarded. This process is repeated for all onsets in the transcription.

\begin{table}[!t]
\centering
\small
\begin{tabular*}{\columnwidth}{ll@{\extracolsep{\fill}}ccc}
\toprule
\textbf{Input} & \textbf{Method} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
\midrule
\multirow{3}{*}{Noisy} 
    & Vanilla     & 96.0 & 43.7 & 57.2 \\
    & + NoiseAug  & 96.1 & \textbf{\underline{82.8}} & \underline{88.7} \\
    & + Video     & \textbf{\underline{97.2}} & 82.7 & \textbf{\underline{89.2}} \\
\midrule
\multirow{2}{*}{Reverberant} 
    & Vanilla     & 66.8 & \textbf{68.2} & 64.4 \\
    & + Video     & \textbf{\underline{68.1}} & 67.8 & \textbf{64.8} \\
\bottomrule
\end{tabular*}
\caption{Onset prediction performance under different acoustic conditions. Bold: highest in each column; Underline: significantly higher over the preceding method (paired $t$-test, $p < 0.05$).}
\label{tab:onset_performance_combined}
\end{table}

Table~\ref{tab:onset_performance_combined} summarizes onset prediction performance under two challenging acoustic conditions: SNR=0\si{dB} Gaussian noise, and reverberation.

To evaluate the model's robustness under reverberant acoustic conditions, we applied convolutional reverb using a real-world impulse response (IR) recorded in St.~George's Church\footnote{\href{https://webfiles.york.ac.uk/OPENAIR/IRs/st-georges-episcopal-church/st-georges-episcopal-church.zip}{https://webfiles.york.ac.uk/OPENAIR/IRs/st-georges-episcopal-church/st-georges-episcopal-church.zip}; st\_georges\_far.wav}. The IR was originally sampled at 96\si{kHz} and downsampled to 16\si{kHz} to match the audio input. All audio samples were convolved with the mono version of this IR using FFT-based convolution. To compensate for the inherent delay in the IR (with its peak located at sample index 653), we removed the first 653 samples from each convolved output to ensure proper temporal alignment. The resulting signals were then peak-normalized to maintain consistent amplitude and avoid distortion.

Under noisy conditions, the baseline model (\textit{Vanilla}), trained on clean audio only, exhibited substantial degradation. Introducing noise during training (\textit{+ NoiseAug}) significantly improved performance, particularly in recall ($t = -9.975$, $p < 0.0001$) and F1 score ($t = -7.623$, $p < 0.0001$), while precision remained unchanged ($t = -0.142$, $p = 0.8891$). We selected CNR=1 and SNR dB value is randomly sampled from [0, 24] (c.f. \cite{ISMIR24Kim}). Adding visual information (\textit{+ Video}) further increased precision ($t = -3.354$, $p = 0.0052$) and F1 ($t = -3.006$, $p = 0.0101$), though the difference in recall compared to the noise-augmented model was not statistically significant ($t = 1.166$, $p = 0.2647$).

In reverberant conditions, the baseline model exhibited a notable decline in performance, highlighting its vulnerability to temporal smearing. Incorporating visual cues led to a significant improvement in precision (\(t = -4.567\), \(p = 0.0005\)). While recall slightly decreased, the change was not statistically significant (\(t = 0.902\), \(p = 0.3836\)). The F1 score increased, approaching significance (\(t = -2.152\), \(p = 0.0508\)). Qualitative inspection revealed that reverberant tails were sometimes misclassified as new onsets, while the visual modality helped reduce such errors.
