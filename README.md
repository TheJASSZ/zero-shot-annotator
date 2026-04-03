# Zero-Shot Video Annotator

A FiftyOne plugin that automatically labels video clips using Twelve Labs
Pegasus (video-to-text) and Marengo (text embedding matching). No training
data needed — just provide a taxonomy.

## How It Works

1. **Pegasus** watches each video clip and writes a natural language description
2. **Marengo** embeds both the descriptions and your taxonomy labels into 512-d vectors
3. **Cosine similarity** matches each description to the closest taxonomy label
4. Results are stored as `fo.Classification` with confidence scores

## Install

```bash
pip install fiftyone twelvelabs
export TWELVELABS_API_KEY="your-key"
fiftyone plugins download https://github.com/TheJASSZ/zero-shot-annotator
```

## Quick Start

1. Load any video dataset in FiftyOne
2. Open operator menu (backtick key)
3. Run "Zero-Shot: Describe Videos"
4. Run "Zero-Shot: Annotate from Taxonomy"
   - Enter: `safe walking, walkway violation, unsafe forklift, ...`
5. Run "Zero-Shot: Review Low Confidence" to check uncertain predictions

## Operators

| Operator | Description |
|---|---|
| **Zero-Shot: Describe Videos** | Runs Pegasus on each video, stores natural language description |
| **Zero-Shot: Annotate from Taxonomy** | Embeds descriptions + taxonomy labels, assigns best match via cosine similarity |
| **Zero-Shot: Review Low Confidence** | Filters to samples below a confidence threshold for manual review |

## Example Taxonomy (Workplace Safety)

```
safe walking in designated area, unsafe walkway violation, safe forklift operation,
unsafe forklift overload, authorized equipment use, unauthorized intervention near machinery,
proper safety gear usage, missing safety equipment
```

## Dataset

Tested on [Safe & Unsafe Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours) — 691 clips from a Turkish manufacturing facility with 8 behavior classes.

## Built at

Video Understanding AI Hackathon @ Northeastern — April 3, 2026
