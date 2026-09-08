"""
cifar_lt.py

Long-tailed variants of CIFAR-10 and CIFAR-100, built as a shared
data-loading component for BCL, ConCutMix, and SKCL (Part 1 reproduction).

Follows the standard exponential long-tailed construction protocol used
across the long-tailed recognition literature (Cui et al. 2019, Cao et al.
2019, and subsequently BCL / ConCutMix / SKCL), so that resulting per-class
sample counts match what those papers report their numbers against.

Imbalance factor convention
----------------------------
The paper reports results at imbalance factor beta = 100. In code this is
expressed as `imb_factor = 1 / beta` (i.e. imb_factor=0.01 for beta=100),
since imb_factor is defined as N_min / N_max -- the ratio of the smallest
class's sample count to the largest class's sample count.

Usage
-----
    from cifar_lt import IMBALANCECIFAR100, classify_many_medium_few

    train_set = IMBALANCECIFAR100(
        root="/path/to/data", imb_type="exp", imb_factor=0.01,
        train=True, download=False, transform=train_transform,
    )
    cls_num_list = train_set.get_cls_num_list()
    many, medium, few = classify_many_medium_few(cls_num_list)

NOTE ON RECONSTRUCTION: this file was rebuilt from fragments quoted across
several past chat turns. `get_img_num_per_cls`'s exponential-decay formula
and `classify_many_medium_few`'s thresholds are reconstructed to match the
described/tested behavior (endpoints, monotonic decrease, >100/20-100/<20
Many/Medium/Few thresholds) rather than copied verbatim -- verify against
your cluster copy before relying on it for report numbers.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Normalization stats (standard values used across the CIFAR-LT literature)
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class _LongTailedCIFARMixin:
    """
    Shared long-tailed subsampling logic. Mixed into torchvision's
    CIFAR10/CIFAR100 classes below rather than duplicated, since CIFAR100
    already carries its own correct base_folder/url/checksums via
    torchvision.datasets.CIFAR100 -- inheriting IMBALANCECIFAR100 from
    IMBALANCECIFAR10 directly would silently pull in CIFAR-10's download
    metadata instead. Verified via MRO inspection during development.
    """

    cls_num: int  # set by subclasses (10 or 100)

    def __init__(
        self,
        *args,
        imb_type: str = "exp",
        imb_factor: float = 0.01,
        rand_number: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]
        np.random.seed(rand_number)
        img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.img_num_per_cls = img_num_per_cls
        self.num_per_cls_dict: Dict[int, int] = {}
        self._gen_imbalanced_data(img_num_per_cls)

    def get_img_num_per_cls(self, cls_num: int, imb_type: str, imb_factor: float) -> List[int]:
        """
        Compute the target number of training images per class.

        imb_type='exp': exponential decay from the largest class down to
        the smallest, with ratio `imb_factor` between the two -- this is
        the convention used to report beta=100 results in the paper.

        imb_type='step': first half of classes at full count, second half
        reduced to imb_factor * full count (used only for the step-imbalance
        sanity test, not for the paper's reported numbers).
        """
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num - cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls = [int(img_max)] * cls_num
        return img_num_per_cls

    def _gen_imbalanced_data(self, img_num_per_cls: List[int]) -> None:
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # Shuffle within-class indices once per class so the kept subset
        # isn't just "the first N images" (avoids any ordering bias in the
        # raw CIFAR files).
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selected_idx = idx[:the_img_num]
            new_data.append(self.data[selected_idx, ...])
            new_targets.extend([the_class] * len(selected_idx))
            self.num_per_cls_dict[int(the_class)] = len(selected_idx)
        self.data = np.vstack(new_data)
        self.targets = new_targets

    def get_cls_num_list(self) -> List[int]:
        return [self.num_per_cls_dict[i] for i in range(self.cls_num)]


class IMBALANCECIFAR10(_LongTailedCIFARMixin, torchvision.datasets.CIFAR10):
    cls_num = 10


class IMBALANCECIFAR100(_LongTailedCIFARMixin, torchvision.datasets.CIFAR100):
    cls_num = 100


def classify_many_medium_few(cls_num_list: Sequence[int]) -> Tuple[List[int], List[int], List[int]]:
    """
    Standard Many (>100) / Medium (20-100) / Few (<20) split used for
    Table 2, computed against the training-set per-class counts.

    NOTE: with only 10 classes at beta=100, CIFAR-10-LT's tail class
    still clears the 20-image "few" cutoff, so its Few bucket comes back
    empty -- this matches standard practice in the literature (only
    CIFAR-100-LT papers report a Many/Medium/Few breakdown).
    """
    many, medium, few = [], [], []
    for idx, n in enumerate(cls_num_list):
        if n > 100:
            many.append(idx)
        elif n >= 20:
            medium.append(idx)
        else:
            few.append(idx)
    return many, medium, few


def get_transforms(dataset: str):
    """Standard crop+flip augmentation for CIFAR-LT training, matching mean/std to the dataset."""
    if dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, test_transform
