"""
Zero-Shot Video Annotator — FiftyOne Plugin

Automatically labels video clips using Twelve Labs Pegasus (video-to-text)
and Marengo (text embedding matching). No training data needed.
"""

import hashlib
import json
import os
import uuid

import numpy as np
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone import ViewField as F
from twelvelabs import TwelveLabs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MARENGO_MODEL = "marengo3.0"
PEGASUS_MODEL = "pegasus1.2"
PEGASUS_OPTIONS = ["visual", "audio"]
DEFAULT_PROMPT = (
    "Describe in one detailed sentence what is happening in this video, "
    "focusing on any notable human actions or safety-relevant behavior."
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Cache directory — sits next to this file
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_EMBED_CACHE = os.path.join(_CACHE_DIR, "embeddings.json")
_DESC_CACHE = os.path.join(_CACHE_DIR, "descriptions.json")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _file_hash(filepath: str) -> str:
    """Fast hash of file path + size + mtime for cache key."""
    stat = os.stat(filepath)
    key = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Helper functions (Phase 1–3)
# ---------------------------------------------------------------------------
def get_client() -> TwelveLabs:
    """Return an authenticated Twelve Labs client."""
    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        raise ValueError("TWELVELABS_API_KEY not set")
    return TwelveLabs(api_key=api_key)


def embed_text(client: TwelveLabs, text: str) -> list[float]:
    """Embed a text string with Marengo -> 512-d vector. Results are cached."""
    text = text[:500]

    # Check cache
    cache = _load_cache(_EMBED_CACHE)
    cache_key = text
    if cache_key in cache:
        return cache[cache_key]

    # Call API
    res = client.embed.create(model_name=MARENGO_MODEL, text=text)
    if res.text_embedding is None or not res.text_embedding.segments:
        err = getattr(res.text_embedding, "error_message", None) if res.text_embedding else None
        raise ValueError(f"Marengo returned no embedding for text: {text[:80]}... (error: {err})")
    seg = res.text_embedding.segments[0]
    if seg.float_ is None:
        raise ValueError(f"Marengo segment has no float_ for text: {text[:80]}...")

    vec = seg.float_

    # Save to cache
    cache[cache_key] = vec
    _save_cache(_EMBED_CACHE, cache)

    return vec


def embed_taxonomy(client: TwelveLabs, labels: list[str]) -> dict[str, list[float]]:
    """Embed every label in a taxonomy -> {label: 512-d vector}."""
    return {label: embed_text(client, label) for label in labels}


def describe_video(client: TwelveLabs, filepath: str, prompt: str) -> str:
    """
    Index video with Pegasus, generate description, cleanup, return text.
    Each call takes ~30-60 seconds due to indexing. Results are cached.
    """
    # Check cache
    cache = _load_cache(_DESC_CACHE)
    cache_key = _file_hash(filepath)
    if cache_key in cache:
        return cache[cache_key]

    # Call API
    index = client.indexes.create(
        index_name=f"zs-{uuid.uuid4().hex[:8]}",
        models=[{"model_name": PEGASUS_MODEL, "model_options": PEGASUS_OPTIONS}],
    )
    try:
        with open(filepath, "rb") as f:
            task = client.tasks.create(index_id=index.id, video_file=f)
        client.tasks.wait_for_done(task.id)
        result = client.analyze(video_id=task.video_id, prompt=prompt)
        desc = result.data
    except Exception as e:
        desc = f"Error: {e}"
    finally:
        client.indexes.delete(index.id)

    # Save to cache
    cache[cache_key] = desc
    _save_cache(_DESC_CACHE, cache)

    return desc


def cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def find_best_match(desc_vec: list, taxonomy_vecs: dict) -> tuple[str, float]:
    """
    Compare description embedding against all taxonomy label embeddings.
    Returns (best_label, confidence_score).
    """
    best_label = None
    best_score = -1.0
    for label, label_vec in taxonomy_vecs.items():
        score = cosine_sim(desc_vec, label_vec)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label, best_score


# ---------------------------------------------------------------------------
# Operator: DescribeVideos (Phase 4B)
# ---------------------------------------------------------------------------
class DescribeVideos(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="describe_videos",
            label="Zero-Shot: Describe Videos",
            description=(
                "Run Twelve Labs Pegasus on each video to generate a "
                "natural language description."
            ),
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = foo.types.Object()
        inputs.str(
            "prompt",
            label="Pegasus Prompt",
            description="What to ask Pegasus about each video",
            default=DEFAULT_PROMPT,
        )
        inputs.int(
            "max_samples",
            label="Max Samples",
            description="Limit number of samples to process (0 = all)",
            default=10,
        )
        return foo.types.Property(inputs)

    async def execute(self, ctx):
        prompt = ctx.params.get("prompt", DEFAULT_PROMPT)
        max_samples = ctx.params.get("max_samples", 10)

        client = get_client()
        dataset = ctx.dataset
        samples = list(dataset)
        if max_samples > 0:
            samples = samples[:max_samples]
        total = len(samples)

        for i, sample in enumerate(samples):
            # Skip already described samples
            if sample.get_field("pegasus_description"):
                ctx.set_progress(progress=(i + 1) / total, label=f"Skipped {i+1}/{total} (already described)")
                continue

            ctx.set_progress(progress=i / total, label=f"Describing {i+1}/{total}...")
            desc = describe_video(client, sample.filepath, prompt)
            sample["pegasus_description"] = desc
            sample.save()

        ctx.set_progress(progress=1.0, label=f"Done — {total} videos described")
        ctx.trigger("reload_dataset")


# ---------------------------------------------------------------------------
# Operator: AnnotateZeroShot (Phase 4C)
# ---------------------------------------------------------------------------
class AnnotateZeroShot(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="annotate_zero_shot",
            label="Zero-Shot: Annotate from Taxonomy",
            description=(
                "Assign labels from a user-defined taxonomy to each video "
                "using Marengo embedding similarity."
            ),
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = foo.types.Object()
        inputs.str(
            "taxonomy",
            label="Taxonomy (comma-separated)",
            description=(
                "Comma-separated list of category labels, e.g. "
                '"safe walking, walkway violation, unsafe forklift"'
            ),
            required=True,
        )
        inputs.float(
            "confidence_threshold",
            label="Confidence Threshold",
            description="Samples below this confidence get tagged 'needs_review'",
            default=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        return foo.types.Property(inputs)

    async def execute(self, ctx):
        raw_taxonomy = ctx.params.get("taxonomy", "")
        threshold = ctx.params.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)

        labels = [l.strip() for l in raw_taxonomy.split(",") if l.strip()]
        if not labels:
            ctx.trigger("show_output", params={"outputs": foo.types.Object()})
            return

        client = get_client()

        # Embed taxonomy labels (fast)
        ctx.set_progress(progress=0.0, label="Embedding taxonomy labels...")
        taxonomy_vecs = embed_taxonomy(client, labels)

        # Get samples that have descriptions
        described = ctx.dataset.exists("pegasus_description")
        samples = list(described)
        total = len(samples)

        for i, sample in enumerate(samples):
            ctx.set_progress(progress=(i + 1) / total, label=f"Annotating {i+1}/{total}...")
            desc = sample["pegasus_description"]
            if desc.startswith("Error:"):
                continue

            desc_vec = embed_text(client, desc)
            best_label, confidence = find_best_match(desc_vec, taxonomy_vecs)

            sample["zero_shot_label"] = fo.Classification(
                label=best_label,
                confidence=confidence,
            )

            if confidence < threshold:
                sample.tags.append("needs_review")

            sample.save()

        ctx.set_progress(progress=1.0, label=f"Done — {total} samples annotated")
        ctx.trigger("reload_dataset")


# ---------------------------------------------------------------------------
# Operator: ReviewLowConfidence (Phase 4D)
# ---------------------------------------------------------------------------
class ReviewLowConfidence(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="review_low_confidence",
            label="Zero-Shot: Review Low Confidence",
            description=(
                "Filter the dataset to show only samples below a "
                "confidence threshold for manual review."
            ),
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = foo.types.Object()
        inputs.float(
            "threshold",
            label="Confidence Threshold",
            description="Show samples with confidence below this value",
            default=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        return foo.types.Property(inputs)

    async def execute(self, ctx):
        threshold = ctx.params.get("threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        low_conf_view = ctx.dataset.filter_labels(
            "zero_shot_label", F("confidence") < threshold
        )
        ctx.trigger(
            "set_view",
            params={"view": low_conf_view._serialize()},
        )


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
def register(plugin):
    plugin.register(DescribeVideos)
    plugin.register(AnnotateZeroShot)
    plugin.register(ReviewLowConfidence)
