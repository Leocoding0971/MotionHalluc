# MotionHalluc: Diagnosing Kinematic Hallucinations in Fine-Grained Motion Reasoning

MotionHalluc is a benchmark and evaluation suite for diagnosing **kinematic hallucinations** in fine-grained motion reasoning tasks in video-based multimodal models.

It provides:
- Three task settings: **AH / TH / DH**
- Evaluation pipelines for accuracy and task-specific metrics
- Video preprocessing utilities for Fit3D-based experiments
- Motion representations derived from pretrained human motion estimation models

---

# 📦 1. Dataset Preparation

📥 Step 1: Download from Hugging Face

We release our benchmark and annotations on Hugging Face:

👉 **MotionHalluc Benchmark + Annotations**: ([HF link here](https://huggingface.co/datasets/motionhalluc/MotionHalluc))

Please download:
- `motion_4dHumans/Bench/` (benchmark inputs)
- `anno/` (all QA annotations)

📥 Step 2: Download Fit3D dataset (required)

We also use the Fit3D dataset as the source of raw videos:

👉 Fit3D dataset: ([Fit3D official link here](https://fit3d.imar.ro/))

⚠️ **Important Notice:**
We do not own or redistribute the Fit3D dataset.
Users must download it from the original source.


📁 Step 3: Place files:
After downloading, organize your project as:
```bash
dataset/
├── Bench/
├── Fit3D/
anno/
```
---

⚙️ Motion Representation Note

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
python produce_clip_from_fit3d.py
```
This script will:

Automatically locate corresponding Fit3D videos
Crop clips according to benchmark annotations

# 🧪 3. Evaluation

We provide evaluation scripts for three tasks:

- AH (Attributional Hallucination)
- TH (Temporal Hallucination)
- DH (Directional Hallucination)

🔑 Set API Key
```bash
export API_KEY="YOUR_API_KEY"
```

🚀 Run evaluation
AH Task
```bash
python AH.py \
  --samples-json AH1.json \
  --video-root . \
  --api-url api_url \
  --model-name YOUR_MODEL \
  --output-json AH1_results.json
```

DH Task
```bash
python DH.py \
  --samples-json DH.json \
  --video-root . \
  --api-url api_url \
  --model-name YOUR_MODEL \
  --output-json DH_results.json
```

TH Task
```bash
python TH.py \
  --samples-json TH.json \
  --video-root . \
  --api-url api_url \
  --model-name YOUR_MODEL \
  --output-json TH_results.json
```

📊 Final Scoring
After obtaining predictions:
```bash
python score_all.py \
  --dh-gt BinaryQA.json \
  --th-gt Clip_QA.json \
  --ah1-gt MCQ_1.json \
  --ah2-gt MCQ_2.json \
  --dh-pred dh_pred.json \
  --th-pred th_pred.json \
  --ah1-pred ah1_pred.json \
  --ah2-pred ah2_pred.json \
  --out scores.json
```


# 📜 License

This repository is released for non-commercial research purposes only.

Users must comply with the license of the Fit3D dataset and any third-party resources used.



