#!/usr/bin/env python3
"""
Smoke test for Experiment 2 representation mechanics.

This script verifies that all 6 Exp2 configurations work correctly:
- representation ∈ {discrete, soft, continuous}
- temporal ∈ {time_tokens, time2vec}

Tests:
1. Model wrapper instantiation with each config
2. Forward pass with synthetic data including numeric_values and relative_times
3. Shape and dtype correctness of outputs
4. Relative time computation is correct

Usage:
    python scripts/smoke_test_exp2.py

    # With verbose output
    python scripts/smoke_test_exp2.py --verbose

    # Test specific config
    python scripts/smoke_test_exp2.py --representation soft --temporal time2vec
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add fms-ehrs to path
FMS_EHRS_PATH = Path(__file__).parent.parent.parent / "fms-ehrs"
sys.path.insert(0, str(FMS_EHRS_PATH))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from fms_ehrs.framework.model_wrapper import (
    RepresentationModelWrapper,
    create_representation_model,
)
from fms_ehrs.framework.vocabulary import Vocabulary


def compute_relative_times_hours(times_list: list) -> list[float]:
    """Convert a list of timestamps to relative hours since first timestamp.

    Local implementation to avoid importing the full dataset module.
    """
    if not times_list or all(t is None for t in times_list):
        return [0.0] * len(times_list) if times_list else []

    # Find first non-null timestamp as admission time
    t0 = None
    for ts in times_list:
        if ts is not None:
            t0 = ts
            break

    if t0 is None:
        return [0.0] * len(times_list)

    result = []
    for ts in times_list:
        if ts is None:
            result.append(0.0)
        elif isinstance(ts, datetime):
            delta = ts - t0
            result.append(delta.total_seconds() / 3600)
        else:
            result.append(0.0)
    return result


def create_mock_vocabulary(num_bins: int = 20) -> Vocabulary:
    """Create a mock vocabulary with quantile tokens and sample codes."""
    vocab = Vocabulary()

    # Add special tokens
    vocab("TL_START")
    vocab("TL_END")
    vocab("PAD")
    vocab("TRUNC")
    vocab(None)
    vocab("nan")

    # Add quantile tokens (Q0, Q1, ..., Qn-1)
    for i in range(num_bins):
        vocab(f"Q{i}")

    # Add time spacing tokens
    time_tokens = [
        "T_5m-15m", "T_15m-1h", "T_1h-2h", "T_2h-6h", "T_6h-12h",
        "T_12h-1d", "T_1d-3d", "T_3d-1w", "T_1w-2w", "T_2w-1mt",
    ]
    for t in time_tokens:
        vocab(t)

    # Add sample laboratory codes with auxiliary data (quantile breaks)
    lab_codes = ["LAB_glucose", "LAB_creatinine", "LAB_hemoglobin", "LAB_potassium"]
    for code in lab_codes:
        vocab(code)
        # Create realistic-ish breaks for each code
        if code == "LAB_glucose":
            breaks = list(range(60, 250, 10))  # 60-240 mg/dL
        elif code == "LAB_creatinine":
            breaks = [0.5 + i * 0.2 for i in range(num_bins - 1)]  # 0.5-4.3 mg/dL
        elif code == "LAB_hemoglobin":
            breaks = [8 + i * 0.5 for i in range(num_bins - 1)]  # 8-17.5 g/dL
        else:  # potassium
            breaks = [2.5 + i * 0.2 for i in range(num_bins - 1)]  # 2.5-6.3 mEq/L
        vocab.set_aux(code, breaks[:num_bins - 1])

    vocab.is_training = False
    return vocab


def create_mock_base_model(vocab_size: int, hidden_size: int = 64) -> AutoModelForCausalLM:
    """Create a tiny LLaMA-style model for testing."""
    from transformers import LlamaConfig

    # Create config directly without loading from HuggingFace
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=1024,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    return AutoModelForCausalLM.from_config(config)


def create_synthetic_batch(
    vocab: Vocabulary,
    batch_size: int = 2,
    seq_len: int = 32,
) -> dict:
    """Create synthetic batch with input_ids, numeric_values, and relative_times."""
    # Create input_ids with some pattern:
    # [TL_START, demographics..., (code, quantile) pairs..., TL_END, PAD...]
    input_ids = torch.full((batch_size, seq_len), vocab("PAD"), dtype=torch.long)

    # Start token
    input_ids[:, 0] = vocab("TL_START")

    # Add some code/quantile pairs
    lab_codes = ["LAB_glucose", "LAB_creatinine", "LAB_hemoglobin"]
    pos = 1
    for code in lab_codes:
        if pos + 2 < seq_len - 1:
            # Time spacing token
            input_ids[:, pos] = vocab("T_1h-2h")
            pos += 1
            # Code token
            input_ids[:, pos] = vocab(code)
            pos += 1
            # Quantile token
            input_ids[:, pos] = vocab("Q10")  # Middle bin
            pos += 1

    # End token
    input_ids[:, pos] = vocab("TL_END")

    # Create numeric_values (NaN for non-numeric positions)
    numeric_values = torch.full((batch_size, seq_len), float("nan"))

    # Fill in values at quantile token positions
    # Find Q10 positions and add sample values
    q10_id = vocab("Q10")
    for b in range(batch_size):
        for s in range(seq_len):
            if input_ids[b, s] == q10_id:
                # Random value in middle of typical range
                numeric_values[b, s] = 100.0 + b * 10  # Different per batch

    # Create relative_times (hours since admission)
    # Simulate 24-hour timeline
    relative_times = torch.zeros((batch_size, seq_len))
    for s in range(seq_len):
        if input_ids[0, s] != vocab("PAD"):
            relative_times[:, s] = s * 0.5  # 30 min per token

    # Attention mask
    attention_mask = (input_ids != vocab("PAD")).long()

    return {
        "input_ids": input_ids,
        "numeric_values": numeric_values,
        "relative_times": relative_times,
        "attention_mask": attention_mask,
    }


def test_relative_time_computation():
    """Test the relative time computation function."""
    print("Testing relative time computation...")

    # Test case 1: Normal timestamps
    t0 = datetime(2025, 1, 1, 0, 0, 0)
    times = [
        t0,
        t0 + timedelta(hours=1),
        t0 + timedelta(hours=2.5),
        None,  # Padding
        t0 + timedelta(hours=24),
    ]
    result = compute_relative_times_hours(times)
    expected = [0.0, 1.0, 2.5, 0.0, 24.0]

    for i, (r, e) in enumerate(zip(result, expected)):
        assert abs(r - e) < 0.001, f"Position {i}: expected {e}, got {r}"

    # Test case 2: All None
    result = compute_relative_times_hours([None, None, None])
    assert result == [0.0, 0.0, 0.0], f"All None case failed: {result}"

    # Test case 3: Empty list
    result = compute_relative_times_hours([])
    assert result == [], f"Empty list case failed: {result}"

    print("  ✓ Relative time computation tests passed")


def test_config(
    representation: str,
    temporal: str,
    vocab: Vocabulary,
    base_model: AutoModelForCausalLM,
    batch: dict,
    verbose: bool = False,
) -> bool:
    """Test a single representation/temporal configuration."""
    config_name = f"{representation}+{temporal}"
    print(f"Testing {config_name}...")

    try:
        # Create wrapped model
        model = create_representation_model(
            base_model=base_model,
            vocab=vocab,
            representation=representation,
            temporal=temporal,
            num_bins=20,
            time2vec_dim=32,
        )

        # Prepare inputs based on config
        inputs = {"input_ids": batch["input_ids"]}

        if representation in ("soft", "xval"):
            inputs["numeric_values"] = batch["numeric_values"]

        if temporal == "time2vec":
            inputs["relative_times"] = batch["relative_times"]

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Verify output shape
        batch_size, seq_len = batch["input_ids"].shape
        expected_shape = (batch_size, seq_len, vocab.__len__())

        if outputs.logits.shape != expected_shape:
            print(f"  ✗ Shape mismatch: expected {expected_shape}, got {outputs.logits.shape}")
            return False

        # Verify output dtype
        if outputs.logits.dtype != torch.float32:
            print(f"  ✗ Dtype mismatch: expected float32, got {outputs.logits.dtype}")
            return False

        # Verify no NaN in outputs
        if torch.isnan(outputs.logits).any():
            print(f"  ✗ NaN values in output logits")
            return False

        if verbose:
            print(f"  Output shape: {outputs.logits.shape}")
            print(f"  Output dtype: {outputs.logits.dtype}")
            print(f"  Output range: [{outputs.logits.min():.4f}, {outputs.logits.max():.4f}]")

        print(f"  ✓ {config_name} passed")
        return True

    except Exception as e:
        print(f"  ✗ {config_name} failed with error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Exp2 configs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--representation",
        choices=["discrete", "soft", "xval"],
        default=None,
        help="Test specific representation (default: all)",
    )
    parser.add_argument(
        "--temporal",
        choices=["time_tokens", "time2vec"],
        default=None,
        help="Test specific temporal encoding (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 2 Smoke Test")
    print("=" * 60)
    print()

    # Test relative time computation first
    test_relative_time_computation()
    print()

    # Create mock vocabulary and model
    print("Setting up test fixtures...")
    vocab = create_mock_vocabulary(num_bins=20)
    base_model = create_mock_base_model(vocab_size=len(vocab))
    batch = create_synthetic_batch(vocab)
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Batch shape: {batch['input_ids'].shape}")
    print()

    # Define all configs
    if args.representation and args.temporal:
        configs = [(args.representation, args.temporal)]
    elif args.representation:
        configs = [(args.representation, t) for t in ["time_tokens", "time2vec"]]
    elif args.temporal:
        configs = [(r, args.temporal) for r in ["discrete", "soft", "xval"]]
    else:
        configs = [
            ("discrete", "time_tokens"),
            ("discrete", "time2vec"),
            ("soft", "time_tokens"),
            ("soft", "time2vec"),
            ("xval", "time_tokens"),
            ("xval", "time2vec"),
        ]

    # Run tests
    print("Running configuration tests...")
    results = []
    for representation, temporal in configs:
        # Create fresh base model for each test (avoids state contamination)
        base_model = create_mock_base_model(vocab_size=len(vocab))
        passed = test_config(
            representation=representation,
            temporal=temporal,
            vocab=vocab,
            base_model=base_model,
            batch=batch,
            verbose=args.verbose,
        )
        results.append((f"{representation}+{temporal}", passed))

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    print()
    print(f"Passed: {n_passed}/{n_total}")

    if n_passed == n_total:
        print("\n✓ All smoke tests passed!")
        return 0
    else:
        print(f"\n✗ {n_total - n_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
