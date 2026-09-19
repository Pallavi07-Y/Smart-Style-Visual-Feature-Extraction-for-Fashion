# Model checkpoints

Download the official MediaPipe Face Landmarker task model and save it here as:

```text
models/face_landmarker.task
```

Download the official MediaPipe Pose Landmarker task model and save it here as:

```text
models/pose_landmarker_lite.task
```

For the wardrobe and clothing modules, add the reviewed project checkpoints:

```text
models/sam2_hiera_small.pt
models/yolo11n.pt
```

The YOLO file must be trained or fine-tuned for clothing classes; the standard
COCO checkpoint is not sufficient for reliable garment categories. SAM 2 also
needs its matching model configuration from the installed SAM 2 release. FashionCLIP
weights are loaded by the selected OpenCLIP adapter and should be cached locally
before a demo.

The analyzer expects a local checkpoint path so model downloads are explicit and do not happen during a user's first upload. Keep large checkpoint files out of Git and document their source and version before sharing the project.
