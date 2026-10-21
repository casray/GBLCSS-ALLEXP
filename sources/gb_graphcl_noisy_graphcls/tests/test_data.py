import torch

from gb_graphcl.data import corrupt_symmetric_labels, stratified_split


class DummyData:
    def __init__(self, y):
        self.y = torch.tensor([y], dtype=torch.long)


class DummyDataset:
    def __init__(self, labels):
        self.items = [DummyData(y) for y in labels]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_stratified_split_is_disjoint_and_covers_all():
    labels = [0] * 50 + [1] * 50
    dataset = DummyDataset(labels)
    train_idx, val_idx, test_idx = stratified_split(dataset, seed=0)
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(range(len(dataset)))
    all_ratio = sum(labels) / len(labels)
    for split in (train_idx, val_idx, test_idx):
        split_ratio = sum(labels[i] for i in split) / len(split)
        assert abs(split_ratio - all_ratio) <= 0.05


def test_symmetric_noise_only_changes_train():
    labels = torch.tensor([0] * 50 + [1] * 50, dtype=torch.long)
    dataset = DummyDataset(labels.tolist())
    train_idx, val_idx, test_idx = stratified_split(dataset, seed=0)
    noisy = corrupt_symmetric_labels(labels, train_idx, 0.4, seed=0)
    assert torch.equal(noisy[val_idx], labels[val_idx])
    assert torch.equal(noisy[test_idx], labels[test_idx])
    changed = int((noisy[train_idx] != labels[train_idx]).sum().item())
    assert 0.3 * len(train_idx) <= changed <= 0.5 * len(train_idx)

