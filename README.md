# GarmentsVision

**Pose-estimation pipeline that measures garment-line workers' hand-motion cadence from overhead video — using YOLO11-pose, ByteTrack, and Fourier analysis.**

GarmentsVision turns raw factory-floor footage into a quantitative measure of how fast an operator repeats a manual work cycle (e.g. a sewing operation). It crops a high-resolution region of interest around a workstation, runs pose estimation to recover wrist trajectories, tracks the operator across frames, and applies frequency analysis to discover the dominant cycle rate.

---

## How it works

```
4K video → ROI crop → YOLO11-pose → ByteTrack → wrist trajectory → FFT cadence
```

1. **ROI crop** — instead of down-scaling the full 3840×2160 frame (which makes distant workers too small to detect), a configurable region around the workstation is cropped so pose runs at high effective resolution.
2. **Pose estimation** — `YOLO11n-pose` recovers 17 COCO keypoints per person; the wrists (indices 9 = left, 10 = right) are the signal of interest.
3. **Tracking** — `ByteTrack` (`model.track(persist=True)`) maintains stable IDs across frames so a single operator's trajectory can be isolated.
4. **Signal conditioning** — the wrist series is re-indexed onto the full frame grid with linear interpolation (no edge-pad artifacts), then detrended to remove slow positional drift.
5. **Cadence discovery** — an FFT peak-pick (Hann-windowed) finds the dominant cyclic frequency within a plausible band, reporting the cycle period and cycles-per-minute. The frequency is *discovered from the data*, not assumed in advance.

---

## Repository structure

```
GarmentsVision/
├── r&d (using fourier transformation).ipynb   # end-to-end pipeline & analysis
├── output/
│   └── hand_points.txt                         # sample extracted wrist trajectories
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The source video and rendered annotation video are intentionally **not** tracked
> (large files / proprietary footage — see `.gitignore`). The extracted `output/hand_points.txt`
> is included so the frequency analysis can be reproduced without re-running detection.

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the CUDA build of PyTorch that matches your toolkit
(see https://pytorch.org/get-started/locally/). The pipeline auto-selects `cuda:0`
when a GPU is available and falls back to CPU otherwise.

### 2. Run the notebook

Open `r&d (using fourier transformation).ipynb` and **Run All**. Configuration lives in a
single block at the top of the extraction cell:

```python
VIDEO_FILE = "path/to/your/video.mp4"
X_LINES, Y_LINES = [1000, 2500], [1300, 2100]   # ROI crop lines
CROP_IDX = 4                                     # which grid cell to analyse
CONF, IOU = 0.5, 0.7                             # detection thresholds
```

The extraction stage writes `output/hand_points.txt` and an annotated
`output/output_pose_tracking.mp4`; the downstream cells perform the cadence analysis.

> **Windows / Anaconda note:** if you hit `OMP: Error #15` (duplicate OpenMP runtime),
> set `KMP_DUPLICATE_LIB_OK=TRUE` in the environment before launching the kernel.

---

## Example result

On the reference clip (61 s, ~3,680 frames), the dominant operator is tracked across
**99.9 %** of frames, and the analysis recovers:

| Metric | Value |
| --- | --- |
| Dominant hand-cycle frequency | **0.114 Hz** |
| Cycle period | **8.8 s** |
| Cycles per minute | **~6.8** |

This independently corroborates a manual stopwatch count of ~6 cycles in 61 s (≈0.098 Hz).

---

## Limitations & roadmap

- The cadence signal currently uses wrist position in image coordinates; a wrist-relative-to-shoulder
  signal would be fully invariant to whole-body movement.
- A small number of brief track fragments remain under heavy occlusion.
- Multi-station / multi-operator batch processing and a headless CLI are natural next steps.

---

## License

Released under the MIT License. The example footage referenced by the pipeline is not
included and remains the property of its owner.
