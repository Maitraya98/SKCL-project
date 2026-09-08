"""
test_cifar_lt.py

Synthetic (no-network, no-real-CIFAR-needed) sanity tests for the
long-tailed subsampling logic in src/datasets/cifar_lt.py. A lightweight
FakeIMBALANCECIFAR stands in for the real torchvision-backed classes so
these run instantly, anywhere.

NOTE ON RECONSTRUCTION: reconstructed to match the described test suite
(7 tests: exponential endpoints, monotonic decrease, total-count
consistency, CIFAR-10-LT beta=100, Many/Medium/Few split, step imbalance,
seed reproducibility) -- re-run against the real module before trusting
report numbers.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "datasets"))
from cifar_lt import _LongTailedCIFARMixin, classify_many_medium_few  # noqa: E402


class FakeIMBALANCECIFAR(_LongTailedCIFARMixin):
    """Bypasses torchvision entirely -- fills in fake .data/.targets directly."""

    def __init__(self, num_classes, per_class, imb_type="exp", imb_factor=0.01, rand_number=0):
        self.cls_num = num_classes
        self.data = np.zeros((num_classes * per_class, 1), dtype=np.uint8)
        self.targets = [c for c in range(num_classes) for _ in range(per_class)]
        np.random.seed(rand_number)
        img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.img_num_per_cls = img_num_per_cls
        self.num_per_cls_dict = {}
        self._gen_imbalanced_data(img_num_per_cls)


def test_exp_imbalance_endpoints():
    ds = FakeIMBALANCECIFAR(num_classes=100, per_class=500, imb_type="exp", imb_factor=0.01)
    counts = ds.get_cls_num_list()
    assert counts[0] == 500, f"largest class should keep all 500: {counts[0]}"
    assert 3 <= counts[-1] <= 7, f"smallest class should be ~500*0.01=5: {counts[-1]}"
    print(f"[OK] exp endpoints: max={counts[0]}, min={counts[-1]}")


def test_monotonically_decreasing():
    ds = FakeIMBALANCECIFAR(num_classes=100, per_class=500, imb_type="exp", imb_factor=0.01)
    counts = ds.get_cls_num_list()
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), "counts must be non-increasing"
    print("[OK] monotonically decreasing")


def test_total_sample_count_matches_data_shape():
    ds = FakeIMBALANCECIFAR(num_classes=10, per_class=5000, imb_type="exp", imb_factor=0.01)
    counts = ds.get_cls_num_list()
    assert sum(counts) == ds.data.shape[0] == len(ds.targets)
    print(f"[OK] total sample count consistent: {sum(counts)}")


def test_cifar10_beta100():
    ds = FakeIMBALANCECIFAR(num_classes=10, per_class=5000, imb_type="exp", imb_factor=0.01)
    counts = ds.get_cls_num_list()
    assert counts[0] == 5000
    assert 40 <= counts[-1] <= 60, f"expected ~50 for CIFAR-10-LT beta=100 tail: {counts[-1]}"
    print(f"[OK] CIFAR-10-LT beta=100: max={counts[0]}, min={counts[-1]}")


def test_many_medium_few_split():
    ds = FakeIMBALANCECIFAR(num_classes=100, per_class=500, imb_type="exp", imb_factor=0.01)
    counts = ds.get_cls_num_list()
    many, medium, few = classify_many_medium_few(counts)
    assert len(many) + len(medium) + len(few) == 100
    assert all(counts[i] > 100 for i in many)
    assert all(20 <= counts[i] <= 100 for i in medium)
    assert all(counts[i] < 20 for i in few)
    print(f"[OK] Many/Medium/Few split: {len(many)}/{len(medium)}/{len(few)}")


def test_step_imbalance():
    ds = FakeIMBALANCECIFAR(num_classes=10, per_class=1000, imb_type="step", imb_factor=0.1)
    counts = ds.get_cls_num_list()
    assert counts[:5] == [1000] * 5, f"head half should be full: {counts[:5]}"
    assert counts[5:] == [100] * 5, f"tail half should be reduced: {counts[5:]}"
    print(f"[OK] step imbalance: {counts}")


def test_seed_reproducibility():
    ds1 = FakeIMBALANCECIFAR(num_classes=10, per_class=500, imb_type="exp", imb_factor=0.01, rand_number=42)
    ds2 = FakeIMBALANCECIFAR(num_classes=10, per_class=500, imb_type="exp", imb_factor=0.01, rand_number=42)
    assert ds1.get_cls_num_list() == ds2.get_cls_num_list()
    print("[OK] seeded runs produce identical per-class counts")


if __name__ == "__main__":
    tests = [
        test_exp_imbalance_endpoints,
        test_monotonically_decreasing,
        test_total_sample_count_matches_data_shape,
        test_cifar10_beta100,
        test_many_medium_few_split,
        test_step_imbalance,
        test_seed_reproducibility,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)
