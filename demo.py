"""
Zero-Shot Video Annotator — Live Demo UI
Run: streamlit run demo.py
"""

import json
import os
import hashlib

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PLUGIN_DIR, ".cache")
EMBED_CACHE = os.path.join(CACHE_DIR, "embeddings.json")
DESC_CACHE = os.path.join(CACHE_DIR, "descriptions.json")
VIDEO_DIR = os.path.expanduser(
    "~/fiftyone/huggingface/hub/Voxel51/Safe_and_Unsafe_Behaviours/data"
)

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

# Category colors
LABEL_COLORS = {
    "safe walking in designated area": "#2ecc71",
    "unsafe walkway violation": "#e74c3c",
    "safe forklift operation": "#2ecc71",
    "unsafe forklift overload": "#e74c3c",
    "authorized equipment use": "#2ecc71",
    "unauthorized intervention near machinery": "#e74c3c",
    "proper safety gear usage": "#2ecc71",
    "missing safety equipment": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def file_hash(filepath: str) -> str:
    stat = os.stat(filepath)
    key = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_caches():
    descriptions = {}
    embeddings = {}
    if os.path.exists(DESC_CACHE):
        with open(DESC_CACHE) as f:
            descriptions = json.load(f)
    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE) as f:
            embeddings = json.load(f)
    return descriptions, embeddings


def get_videos():
    """Get video files and match to cached descriptions."""
    if not os.path.exists(VIDEO_DIR):
        return []
    descriptions, _ = load_caches()
    videos = []
    for fname in sorted(os.listdir(VIDEO_DIR)):
        if not fname.endswith(".mp4"):
            continue
        fpath = os.path.join(VIDEO_DIR, fname)
        fhash = file_hash(fpath)
        desc = descriptions.get(fhash, None)
        videos.append({"name": fname, "path": fpath, "hash": fhash, "description": desc})
    return videos


def annotate_video(description, taxonomy_labels, embeddings):
    """Match description to best taxonomy label using cached embeddings."""
    if not description or description.startswith("Error:"):
        return None, 0.0

    desc_key = description[:500]
    desc_vec = embeddings.get(desc_key)
    if desc_vec is None:
        return None, 0.0

    best_label = None
    best_score = -1.0
    for label in taxonomy_labels:
        label_vec = embeddings.get(label)
        if label_vec is None:
            continue
        score = cosine_sim(desc_vec, label_vec)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label, best_score


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zero-Shot Video Annotator",
    page_icon="🎯",
    layout="wide",
)

# --- Header ---
st.title("🎯 Zero-Shot Video Annotator")
st.markdown(
    "Automatically label video clips using **Twelve Labs Pegasus** (video → text) "
    "and **Marengo** (text → embedding similarity). **No training data needed.**"
)

# --- Sidebar ---
st.sidebar.header("How It Works")
st.sidebar.markdown("""
1. **Pegasus** watches each video and writes a description
2. **Marengo** embeds descriptions + taxonomy labels into 512-d vectors
3. **Cosine similarity** matches each description to the closest label
4. Results include confidence scores for review
""")

st.sidebar.header("Taxonomy")
st.sidebar.markdown("The categories we're matching against:")
for label in TAXONOMY:
    color = LABEL_COLORS.get(label, "#999")
    safe_or_unsafe = "✅" if "safe" in label.lower() and "unsafe" not in label.lower() else "⚠️"
    st.sidebar.markdown(f"{safe_or_unsafe} `{label}`")

st.sidebar.divider()
st.sidebar.markdown("**Built at** Video Understanding AI Hackathon @ Northeastern — April 3, 2026")

# --- Load data ---
videos = get_videos()
descriptions, embeddings = load_caches()

if not videos:
    st.error(f"No videos found in `{VIDEO_DIR}`. Load the dataset first.")
    st.stop()

# --- Pipeline Overview ---
st.header("Pipeline Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Videos", len(videos))
col2.metric("Described", sum(1 for v in videos if v["description"]))
col3.metric("Taxonomy Labels", len(TAXONOMY))
col4.metric("Cached Embeddings", len(embeddings))

st.divider()

# --- Step 1: Describe ---
st.header("Step 1: Pegasus Video Descriptions")
st.markdown("Pegasus watches each clip and writes a natural language description of what's happening.")

described_videos = [v for v in videos if v["description"] and not v["description"].startswith("Error:")]

if not described_videos:
    st.warning("No descriptions cached yet. Run `python3 tests.py phase5` first.")
    st.stop()

# --- Step 2: Annotate ---
st.header("Step 2: Zero-Shot Annotation Results")
st.markdown("Each description is embedded and matched against the taxonomy via cosine similarity.")

# Confidence threshold slider
threshold = st.slider("Confidence threshold for review", 0.0, 1.0, 0.5, 0.05)

# Results
results = []
for v in described_videos:
    label, conf = annotate_video(v["description"], TAXONOMY, embeddings)
    results.append({**v, "label": label, "confidence": conf})

# Sort by confidence
results.sort(key=lambda x: x["confidence"], reverse=True)

# --- Video Cards ---
for i, r in enumerate(results):
    conf = r["confidence"]
    label = r["label"] or "Unknown"
    color = LABEL_COLORS.get(label, "#999")
    needs_review = conf < threshold

    with st.container():
        st.divider()
        left, right = st.columns([1, 2])

        with left:
            st.video(r["path"])
            st.caption(r["name"])

        with right:
            # Label badge
            review_badge = " 🔍 **NEEDS REVIEW**" if needs_review else ""
            st.markdown(
                f"### <span style='color:{color}'>{label}</span>{review_badge}",
                unsafe_allow_html=True,
            )

            # Confidence bar
            st.progress(conf, text=f"Confidence: {conf:.1%}")

            # Description
            with st.expander("📝 Pegasus Description", expanded=False):
                st.write(r["description"])

            # Similarity breakdown
            if r["description"] and not r["description"].startswith("Error:"):
                desc_vec = embeddings.get(r["description"][:500])
                if desc_vec:
                    with st.expander("📊 Similarity Scores (all labels)", expanded=False):
                        scores = {}
                        for tax_label in TAXONOMY:
                            tax_vec = embeddings.get(tax_label)
                            if tax_vec:
                                scores[tax_label] = cosine_sim(desc_vec, tax_vec)
                        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        for sl, sc in sorted_scores:
                            bar_color = LABEL_COLORS.get(sl, "#999")
                            is_best = " ← **best match**" if sl == label else ""
                            st.markdown(f"`{sc:.3f}` — {sl}{is_best}")

# --- Summary ---
st.divider()
st.header("Summary")

total = len(results)
high_conf = sum(1 for r in results if r["confidence"] >= threshold)
low_conf = total - high_conf

safe_count = sum(1 for r in results if r["label"] and ("safe" in r["label"].lower() and "unsafe" not in r["label"].lower()))
unsafe_count = sum(1 for r in results if r["label"] and "unsafe" in r["label"].lower())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Annotated", total)
col2.metric("High Confidence", high_conf, help=f"Above {threshold:.0%}")
col3.metric("Needs Review", low_conf, delta=f"-{low_conf}" if low_conf else "0", delta_color="inverse")
col4.metric("Safe / Unsafe", f"{safe_count} / {unsafe_count}")

st.markdown("---")
st.markdown(
    "*Zero-shot accuracy achieved with **zero training examples**. "
    "Just a taxonomy + Twelve Labs Pegasus & Marengo.*"
)
