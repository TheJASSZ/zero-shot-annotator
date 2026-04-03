"""
Zero-Shot Video Annotator — Phase-by-phase tests.

Run individual phases:
    python tests.py phase1
    python tests.py phase2
    python tests.py phase3
    python tests.py phase4
    python tests.py phase5

Or run all:
    python tests.py all
"""

import os
import sys
import math

import numpy as np

# Import helpers from the plugin
from __init__ import (
    get_client,
    embed_text,
    embed_taxonomy,
    describe_video,
    cosine_sim,
    find_best_match,
)

PASSED = 0
FAILED = 0


def assert_true(condition, msg=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {msg}")
    else:
        FAILED += 1
        print(f"  FAIL: {msg}")


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ===== PHASE 1: API Connectivity =====
def phase1():
    print("\n=== Phase 1: API Connectivity ===")
    client = get_client()

    # Test 1.1 — Marengo text embedding returns 512-d vector
    print("\nTest 1.1 — Marengo text embedding")
    res = client.embed.create(
        model_name="marengo3.0",
        text="a person walking on a factory floor",
    )
    vec = res.text_embedding.segments[0].float_
    assert_true(isinstance(vec, list), "embedding is a list")
    assert_true(len(vec) == 512, f"embedding dim = {len(vec)} (expected 512)")
    assert_true(all(isinstance(v, float) for v in vec), "all values are floats")

    # Test 1.2 — Marengo works for a batch of taxonomy labels
    print("\nTest 1.2 — Marengo taxonomy batch")
    labels = [
        "safe walking",
        "unsafe forklift operation",
        "walkway violation",
        "unauthorized intervention",
    ]
    embeddings = {}
    for label in labels:
        res = client.embed.create(
            model_name="marengo3.0",
            text=label,
        )
        embeddings[label] = res.text_embedding.segments[0].float_
    assert_true(len(embeddings) == 4, "4 labels embedded")
    assert_true(all(len(v) == 512 for v in embeddings.values()), "all 512-d")

    # Test helpers
    print("\nTest 1.3 — embed_text helper")
    vec2 = embed_text(client, "test phrase")
    assert_true(len(vec2) == 512, "embed_text returns 512-d")

    print("\nTest 1.4 — embed_taxonomy helper")
    tax = embed_taxonomy(client, ["label_a", "label_b"])
    assert_true(len(tax) == 2, "embed_taxonomy returns 2 entries")


# ===== PHASE 2: Pegasus Video Descriptions =====
def phase2():
    print("\n=== Phase 2: Pegasus Video Descriptions ===")
    import fiftyone as fo
    from fiftyone.utils.huggingface import load_from_hub

    dataset = load_from_hub(
        "Voxel51/Safe_and_Unsafe_Behaviours",
        max_samples=2,
        name="zero_shot_phase2_test",
        overwrite=True,
    )
    client = get_client()

    # Test 2.1 — Pegasus describes one video
    print("\nTest 2.1 — Pegasus describe single video")
    sample = dataset.first()
    desc = describe_video(client, sample.filepath, "Describe in one detailed sentence what is happening in this video.")
    print(f"  Pegasus says: {desc}")
    assert_true(isinstance(desc, str), "description is a string")
    assert_true(len(desc) > 10, f"description is non-trivial (len={len(desc)})")
    assert_true(not desc.startswith("Error:"), "no error")

    # Test 2.2 — Description stored on FiftyOne sample
    print("\nTest 2.2 — Description stored on sample")
    sample["pegasus_description"] = "A worker operates a forklift near a loading dock"
    sample.save()
    retrieved = dataset.first()
    assert_true(
        retrieved["pegasus_description"] == "A worker operates a forklift near a loading dock",
        "description persists on sample",
    )

    fo.delete_dataset("zero_shot_phase2_test")


# ===== PHASE 3: Cosine Similarity + Label Assignment =====
def phase3():
    print("\n=== Phase 3: Cosine Similarity + Label Assignment ===")
    client = get_client()

    # Test 3.1 — Cosine similarity math
    print("\nTest 3.1 — Cosine similarity math")
    a = [1.0, 0.0, 0.0]
    assert_true(approx(cosine_sim(a, a), 1.0), "identical vectors -> 1.0")
    b = [-1.0, 0.0, 0.0]
    assert_true(approx(cosine_sim(a, b), -1.0), "opposite vectors -> -1.0")
    c = [0.0, 1.0, 0.0]
    assert_true(approx(cosine_sim(a, c), 0.0), "orthogonal vectors -> 0.0")

    # Test 3.2 — Semantic similarity ordering
    print("\nTest 3.2 — Semantic similarity ordering")
    v_forklift = embed_text(client, "a person driving a forklift dangerously")
    v_unsafe = embed_text(client, "unsafe forklift operation")
    v_beach = embed_text(client, "beautiful sunset at the beach")
    sim_related = cosine_sim(v_forklift, v_unsafe)
    sim_unrelated = cosine_sim(v_forklift, v_beach)
    print(f"  related sim = {sim_related:.4f}, unrelated sim = {sim_unrelated:.4f}")
    assert_true(sim_related > sim_unrelated, "related text is closer than unrelated")

    # Test 3.3 — Best match from taxonomy
    print("\nTest 3.3 — Best label assignment")
    description = "a worker crosses the forklift lane without looking"
    desc_vec = embed_text(client, description)
    taxonomy = ["safe walking", "walkway violation", "forklift overload", "safe equipment use"]
    taxonomy_vecs = embed_taxonomy(client, taxonomy)
    best_label, confidence = find_best_match(desc_vec, taxonomy_vecs)
    print(f"  best_label = {best_label}, confidence = {confidence:.4f}")
    assert_true(best_label == "walkway violation", f"expected 'walkway violation', got '{best_label}'")
    assert_true(0.0 < confidence <= 1.0, "confidence in valid range")

    # Test 3.4 — Classification stored
    print("\nTest 3.4 — fo.Classification stored")
    import fiftyone as fo
    from fiftyone.utils.huggingface import load_from_hub

    dataset = load_from_hub(
        "Voxel51/Safe_and_Unsafe_Behaviours",
        max_samples=1,
        name="zero_shot_phase3_test",
        overwrite=True,
    )
    sample = dataset.first()
    sample["zero_shot_label"] = fo.Classification(label="walkway violation", confidence=0.87)
    sample.save()
    retrieved = dataset.first()
    assert_true(retrieved["zero_shot_label"].label == "walkway violation", "label persists")
    assert_true(approx(retrieved["zero_shot_label"].confidence, 0.87), "confidence persists")
    fo.delete_dataset("zero_shot_phase3_test")


# ===== PHASE 4: Plugin Manifest =====
def phase4():
    print("\n=== Phase 4: Plugin Manifest ===")
    import yaml

    # Test 4.1 — fiftyone.yml is valid
    print("\nTest 4.1 — fiftyone.yml valid")
    with open("fiftyone.yml") as f:
        config = yaml.safe_load(f)
    assert_true(config["name"] == "zero-shot-annotator", "plugin name correct")
    assert_true("describe_videos" in config["operators"], "describe_videos listed")
    assert_true("annotate_zero_shot" in config["operators"], "annotate_zero_shot listed")
    assert_true("review_low_confidence" in config["operators"], "review_low_confidence listed")

    # Test 4.3 — Taxonomy parsing
    print("\nTest 4.3 — Taxonomy parsing")
    raw = "safe walking, walkway violation, unsafe forklift, unauthorized intervention"
    labels = [l.strip() for l in raw.split(",")]
    assert_true(len(labels) == 4, "4 labels parsed")
    assert_true(labels[0] == "safe walking", "first label correct")
    assert_true(labels[3] == "unauthorized intervention", "last label correct")


# ===== PHASE 5: End-to-End Integration =====
def phase5():
    print("\n=== Phase 5: End-to-End Integration ===")
    import fiftyone as fo
    from fiftyone.utils.huggingface import load_from_hub

    dataset = load_from_hub(
        "Voxel51/Safe_and_Unsafe_Behaviours",
        max_samples=10,
        name="e2e_test",
        overwrite=True,
    )
    client = get_client()

    PROMPT = (
        "Describe in one detailed sentence what is happening in this video, "
        "focusing on any notable human actions or safety-relevant behavior."
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

    # Step 1: Describe all videos
    for i, sample in enumerate(dataset):
        print(f"  Describing {i+1}/{len(dataset)}...")
        desc = describe_video(client, sample.filepath, PROMPT)
        sample["pegasus_description"] = desc
        sample.save()

    described = dataset.exists("pegasus_description")
    assert_true(len(described) >= 8, f"at least 8 described ({len(described)} actual)")

    # Step 2: Embed taxonomy
    taxonomy_vecs = embed_taxonomy(client, TAXONOMY)
    assert_true(len(taxonomy_vecs) == len(TAXONOMY), "all taxonomy labels embedded")

    # Step 3: Annotate
    for sample in described:
        desc = sample["pegasus_description"]
        print(f"  Description: {desc[:80]}...")
        if desc.startswith("Error:"):
            print(f"    SKIPPING (error description)")
            continue
        try:
            desc_vec = embed_text(client, desc)
            label, conf = find_best_match(desc_vec, taxonomy_vecs)
            sample["zero_shot_label"] = fo.Classification(label=label, confidence=conf)
            sample.save()
            print(f"    -> {label} ({conf:.3f})")
        except Exception as e:
            print(f"    SKIPPING (embed error: {e})")

    annotated = dataset.exists("zero_shot_label")
    assert_true(len(annotated) > 0, f"{len(annotated)} samples annotated")

    # Step 4: Validate against ground truth
    correct = 0
    total = 0
    for sample in annotated:
        predicted = sample["zero_shot_label"].label
        ground_truth = sample.get_field("label") or sample.get_field("ground_truth")
        if ground_truth:
            total += 1
            gt_str = str(ground_truth).lower() if not hasattr(ground_truth, "label") else ground_truth.label.lower()
            if any(word in predicted.lower() for word in gt_str.split()):
                correct += 1

    if total > 0:
        accuracy = correct / total
        print(f"\n  Zero-shot accuracy: {correct}/{total} = {accuracy:.0%}")

    # Step 5: Low confidence
    from fiftyone import ViewField as F
    if len(annotated) > 0:
        low_conf = dataset.filter_labels("zero_shot_label", F("confidence") < 0.5)
        print(f"  {len(low_conf)} samples below 0.5 confidence")
    else:
        print("  Skipping low-confidence filter (no annotations)")

    fo.delete_dataset("e2e_test")


# ===== Main =====
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"

    phases = {
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "phase4": phase4,
        "phase5": phase5,
    }

    if phase == "all":
        for name, fn in phases.items():
            fn()
    elif phase in phases:
        phases[phase]()
    else:
        print(f"Unknown phase: {phase}")
        print("Usage: python tests.py [phase1|phase2|phase3|phase4|phase5|all]")
        sys.exit(1)

    print(f"\n{'='*40}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    if FAILED > 0:
        sys.exit(1)
