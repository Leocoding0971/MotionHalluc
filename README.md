# MotionHalluc: Diagnosing Kinematic Hallucinations in Fine-Grained Motion Reasoning

MotionHalluc is a benchmark and evaluation suite for diagnosing **kinematic hallucinations** in fine-grained motion reasoning tasks in video-based multimodal models.

It provides:
- Three task settings: **AH / TH / DH**
- Evaluation pipelines for accuracy and task-specific metrics
- Video preprocessing utilities for Fit3D-based experiments
- Motion representations derived from pretrained human motion estimation models

---

# 📦 1. Dataset Preparation

## 📥 Step 1: Download from Hugging Face

We release our benchmark and annotations on Hugging Face:

👉 **MotionHalluc Benchmark + Annotations**: ([HF link here](https://huggingface.co/datasets/motionhalluc/MotionHalluc))

Please download:
- `motion_4dHumans/Bench/` (benchmark inputs)
- `anno/` (all QA annotations)

## 📥 Step 2: Download Fit3D dataset (required)

We also use the Fit3D dataset as the source of raw videos:

👉 Fit3D dataset: ([Fit3D official link here](https://fit3d.imar.ro/))

⚠️ **Important Notice:**
We do not own or redistribute the Fit3D dataset.
Users must download it from the original source.


## 📁 Step 3: Place files

After downloading, organize your project as:
dataset/
├── Bench/
├── Fit3D/
anno/
...
---

## ⚙️ Motion Representation Note

The HF benchmark already includes motion representations extracted using a pretrained **4D human motion estimation model**.

These representations are widely used in our experiments.

> Due to licensing restrictions of the Fit3D dataset, we do not redistribute any derived motion data.
>
> We provide data indices and detailed descriptions of the processing procedure.
>
> The full preprocessing pipeline will be released upon acceptance to facilitate reproducibility.

---

# 🎬 2. Video Preprocessing (Fit3D Clip Generation)

We provide an automatic script to extract clips from raw Fit3D videos:

```bash
python produce_clip_from_fit3d/produce_clip_from_fit3d.py
```
This script will:

Automatically locate corresponding Fit3D videos
Crop clips according to benchmark annotations

# 🧪 3. Evaluation

We provide evaluation scripts for three tasks:

- AH (Action Hallucination)
- TH (Temporal Hallucination)
- DH (Direction/Displacement Hallucination)

export API_KEY="YOUR_API_KEY"
