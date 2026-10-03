import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


PROTOCOL_TRAIN_SAMPLES = 10_000
PROTOCOL_TEST_SAMPLES = 2_000
PROTOCOL_SEEDS = (42, 100, 2026, 3141, 404)
NORMALIZE = transforms.Normalize((0.5,), (0.5,))


def _repo_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _resolve_data_root(root: str | Path) -> Path:
	path = Path(root)
	if not path.is_absolute():
		path = _repo_root() / path
	return path

def _load_split_indices(split_name: str, root: str | Path = "data") -> list[int]:
	with (_resolve_data_root(root) / "splits" / f"{split_name}_indices.json").open(
		"r", encoding="utf-8"
	) as file:
		payload = json.load(file)
	if isinstance(payload, dict) and "indices" in payload:
		payload = payload["indices"]
	return [int(index) for index in payload]


class NormalizedDataset(Dataset):
	def __init__(self, base_dataset: Dataset) -> None:
		super().__init__()
		self.base_dataset = base_dataset

	def __len__(self) -> int:
		return len(self.base_dataset)

	def __getitem__(self, index: int):
		image, label = self.base_dataset[index]
		return NORMALIZE(image), label


def get_mnist_loaders(
	root: str = "data",
	batch_size: int = 128,
	max_train_samples: int | None = None,
	max_test_samples: int | None = None,
	num_workers: int = 0,
	seed: int = 42,
	use_fixed_indices: bool = True,
) -> tuple[DataLoader, DataLoader]:
	transform = transforms.ToTensor()
	train_dataset = datasets.MNIST(_resolve_data_root(root), train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST(_resolve_data_root(root), train=False, download=True, transform=transform)
	if use_fixed_indices:
		train_indices = _load_split_indices("train", root)[:max_train_samples or len(_load_split_indices("train", root))]
		test_indices = _load_split_indices("test", root)[:max_test_samples or len(_load_split_indices("test", root))]
		train_dataset = Subset(train_dataset, train_indices)
		test_dataset = Subset(test_dataset, test_indices)
	elif max_train_samples is not None:
		train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
	elif max_test_samples is not None:
		test_dataset = Subset(test_dataset, range(min(max_test_samples, len(test_dataset))))
	train_dataset = NormalizedDataset(train_dataset)
	test_dataset = NormalizedDataset(test_dataset)
	generator = torch.Generator()
	generator.manual_seed(seed)
	return (
		DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=num_workers,
			generator=generator,
		),
		DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
	)


def get_standard_test_loader(
	root: str = "data",
	batch_size: int = 128,
	max_test_samples: int | None = PROTOCOL_TEST_SAMPLES,
	num_workers: int = 0,
	use_fixed_indices: bool = True,
) -> DataLoader:
	transform = transforms.ToTensor()
	dataset = datasets.MNIST(_resolve_data_root(root), train=False, download=True, transform=transform)
	indices = _load_split_indices("test", root)
	if max_test_samples is not None:
		indices = indices[:max_test_samples]
	return DataLoader(NormalizedDataset(Subset(dataset, indices)), batch_size=batch_size, shuffle=False, num_workers=num_workers)


def split_context_target(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	context = images[:, :, :, :14].contiguous().view(images.size(0), -1)
	target = images[:, :, :, 14:].contiguous().view(images.size(0), -1)
	return context, target

def generate_splits(
    data_root: str = "data",
    train_pool: int = 60_000,
    test_pool: int = 10_000,
    train_samples: int = 10_000,
    test_samples: int = 2_000,
    seed: int = 42,
) -> None:
    root = Path(data_root)
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    train_indices = rng.sample(range(train_pool), train_samples)
    test_indices = rng.sample(range(test_pool), test_samples)

    for name, indices in [("train", train_indices), ("test", test_indices)]:
        path = splits_dir / f"{name}_indices.json"
        path.write_text(json.dumps({"indices": indices}), encoding="utf-8")
