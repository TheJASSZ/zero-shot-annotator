# Zero-Shot Video Annotator

A [FiftyOne](https://docs.voxel51.com/) plugin that automatically labels video clips using [Twelve Labs](https://twelvelabs.io/) **Pegasus** (video-to-text) and **Marengo** (text embedding matching). No training data needed — just provide a taxonomy and get instant labels.

## Demo

![Zero Shot Demo](Zero%20Shot%20Demo.gif)

## The Problem

Labeling video is expensive and slow. A single annotator reviewing workplace safety footage can take days to label hundreds of clips. In scarce-data domains — manufacturing safety, medical procedures, wildlife monitoring — you often can't afford to label enough data to train a classifier.

**This plugin eliminates that bottleneck.** Give it unlabeled video clips and a list of categories. It labels everything automatically using zero-shot video understanding. No training examples. No fine-tuning. No manual annotation.

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Video Clip  │────▶│  Pegasus (v1.2)  │────▶│  "A worker walks  │     │              │
│  (unlabeled) │     │  video → text    │     │   along the aisle │     │  Cosine      │
└─────────────┘     └──────────────────┘     │   near machinery" │────▶│  Similarity   │──▶ Best Label
                                              └───────────────────┘     │  Matching     │   + Confidence
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐     │              │
│  Taxonomy    │────▶│  Marengo (3.0)   │────▶│  512-d embedding  │────▶│              │
│  (user text) │     │  text → embed    │     │  per label        │     └──────────────┘
└─────────────┘     └──────────────────┘     └───────────────────┘
```

1. **Pegasus** watches each video clip and writes a natural language description of what's happening
2. **Marengo** embeds both the description and your taxonomy labels into 512-dimensional vectors
3. **Cosine similarity** finds the closest taxonomy label — that becomes the predicted label with a confidence score
4. Low-confidence predictions are flagged for human review

## Why This Approach

Instead of training a classifier (which needs labeled data), we use Twelve Labs to **describe** what's happening (Pegasus) and **match** those descriptions to category labels (Marengo). This means:

- **Zero training examples** needed — works on any domain from day one
- **Any taxonomy** — just type comma-separated labels, change them anytime
- **Transparent predictions** — you can read *why* a label was assigned (the Pegasus description)
- **Human-in-the-loop** — low-confidence predictions are flagged for review, not silently accepted

## Install

### Requirements

- Python 3.10+
- [FiftyOne](https://docs.voxel51.com/getting_started/install.html) >= 0.25
- [Twelve Labs API key](https://api.twelvelabs.io/) (free tier works)

### Setup

```bash
# Install dependencies
pip install fiftyone twelvelabs numpy

# Set your API key
export TWELVELABS_API_KEY="your-key-here"

# Install the plugin
fiftyone plugins download https://github.com/TheJASSZ/zero-shot-annotator
```

## Quick Start

### Option 1: FiftyOne App (GUI)

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load a video dataset
dataset = load_from_hub(
    "Voxel51/Safe_and_Unsafe_Behaviours",
    max_samples=10,
    name="safety_demo",
    overwrite=True,
)

# Launch the App
session = fo.launch_app(dataset)
```

Then in the FiftyOne App:
1. Press the **backtick key** (`` ` ``) to open the operator menu
2. Run **"Zero-Shot: Describe Videos"** — Pegasus generates a description for each clip
3. Run **"Zero-Shot: Annotate from Taxonomy"** — type your categories, get instant labels
4. Run **"Zero-Shot: Review Low Confidence"** — filter to uncertain predictions for manual review

### Option 2: Python Script

```python
import os
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Assuming the plugin is on your Python path
from zero_shot_annotator import get_client, describe_video, embed_text, embed_taxonomy, find_best_match

# Load dataset
dataset = load_from_hub(
    "Voxel51/Safe_and_Unsafe_Behaviours",
    max_samples=10,
    name="safety_demo",
    overwrite=True,
)

client = get_client()

# Define your taxonomy — change these to anything
TAXONOMY = [
    "safe walking in designated area",
    "unsafe walkway violation",
    "safe forklift operation",
    "unsafe forklift overload",
    "authorized equipment use",
    "unauthorized intervention near machinery",
    "proper safety gear usage",
    "missing safety equipment",
]

PROMPT = "Describe in one detailed sentence what is happening in this video, focusing on any notable human actions or safety-relevant behavior."

# Step 1: Describe videos with Pegasus
for sample in dataset:
    desc = describe_video(client, sample.filepath, PROMPT)
    sample["pegasus_description"] = desc
    sample.save()
    print(f"Described: {desc[:80]}...")

# Step 2: Embed taxonomy labels with Marengo
taxonomy_vecs = embed_taxonomy(client, TAXONOMY)

# Step 3: Match descriptions to labels
for sample in dataset.exists("pegasus_description"):
    desc = sample["pegasus_description"]
    if desc.startswith("Error:"):
        continue
    desc_vec = embed_text(client, desc)
    label, confidence = find_best_match(desc_vec, taxonomy_vecs)
    sample["zero_shot_label"] = fo.Classification(label=label, confidence=confidence)
    sample.save()

# View results
session = fo.launch_app(dataset)
```

### Option 3: Streamlit Demo UI

```bash
cd zero-shot-annotator
streamlit run demo.py
```

This launches an interactive dashboard showing video clips, Pegasus descriptions, predicted labels, confidence scores, and similarity breakdowns — all from cached data (no API calls needed).

## Operators

| Operator | What You See | What It Does |
|---|---|---|
| **Zero-Shot: Describe Videos** | "Describe Videos" in operator menu | Runs Pegasus on each video → stores `pegasus_description` field. Skips already-described samples. |
| **Zero-Shot: Annotate from Taxonomy** | "Annotate from Taxonomy" in operator menu | User types comma-separated taxonomy → Marengo embeds descriptions + labels → cosine similarity assigns best match → writes `fo.Classification` with confidence score |
| **Zero-Shot: Review Low Confidence** | "Review Low Confidence" in operator menu | Filters to samples below a confidence threshold so you can manually review uncertain predictions |

### Operator Parameters

**Describe Videos:**
- `prompt` — What to ask Pegasus (default: safety-focused prompt)
- `max_samples` — Limit for prototyping (default: 10, set 0 for all)

**Annotate from Taxonomy:**
- `taxonomy` — Comma-separated category labels
- `confidence_threshold` — Below this, samples get tagged `needs_review` (default: 0.5)

**Review Low Confidence:**
- `threshold` — Confidence cutoff (default: 0.5)

## Plugin Structure

```
zero-shot-annotator/
├── fiftyone.yml           # Plugin manifest
├── __init__.py            # All operators + helper functions
├── demo.py                # Streamlit demo UI
├── tests.py               # Phase-by-phase test suite
├── cache_descriptions.py  # Utility to pre-cache description embeddings
├── README.md              # This file
└── .cache/                # Local API response cache (auto-generated)
    ├── descriptions.json  # Pegasus video descriptions
    └── embeddings.json    # Marengo text embeddings
```

## API Caching

All Twelve Labs API responses are automatically cached locally in `.cache/`. This means:
- **Re-running the plugin** on the same videos costs zero API calls
- **Demo presentations** work offline from cached data
- **Rate limits** are preserved — you never waste a call on repeated input

## Example: Workplace Safety Dataset

Tested on [Safe & Unsafe Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours) — 691 clips from a Turkish manufacturing facility with 8 behavior classes (4 safe, 4 unsafe): walkway violations, unauthorized interventions, forklift overloads, etc.

The dataset has ground-truth labels — perfect for validation. We pretend they don't exist, auto-label with the plugin, then compare accuracy. Zero training examples used.

**Example taxonomy:**
```
safe walking in designated area, unsafe walkway violation, safe forklift operation,
unsafe forklift overload, authorized equipment use, unauthorized intervention near machinery,
proper safety gear usage, missing safety equipment
```

## Use It On Your Own Data

This plugin works on **any video dataset** — just change the taxonomy:

| Domain | Example Taxonomy |
|---|---|
| **Workplace Safety** | safe walking, walkway violation, forklift overload, missing safety gear |
| **Wildlife Monitoring** | bird feeding, predator approach, nest building, territorial display |
| **Sports Analysis** | goal attempt, foul, corner kick, penalty |
| **Retail Analytics** | customer browsing, checkout queue, shelf restocking, theft attempt |
| **Medical** | proper hand washing, equipment sterilization, patient interaction |

## Testing

Run individual phases or all tests:

```bash
export TWELVELABS_API_KEY="your-key"
cd zero-shot-annotator

python3 tests.py phase1   # API connectivity (~10s)
python3 tests.py phase2   # Pegasus descriptions (~2 min)
python3 tests.py phase3   # Cosine similarity + matching (~15s)
python3 tests.py phase4   # Plugin manifest validation (instant)
python3 tests.py phase5   # Full end-to-end pipeline (~10 min)
python3 tests.py all      # Everything
```

## Tech Stack

- **[FiftyOne](https://docs.voxel51.com/)** — Plugin framework, dataset management, App UI
- **[Twelve Labs Marengo 3.0](https://docs.twelvelabs.io/)** — 512-d text embeddings for taxonomy matching
- **[Twelve Labs Pegasus 1.2](https://docs.twelvelabs.io/)** — Video-to-text description generation
- **[Streamlit](https://streamlit.io/)** — Demo UI dashboard
- **NumPy** — Cosine similarity computation

## Built at

Video Understanding AI Hackathon @ Northeastern — April 3, 2026
