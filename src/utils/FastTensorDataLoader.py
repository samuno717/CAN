import torch


class FastTensorDataLoader:
    def __init__(self, dataset_tensor, batch_size: int, shuffle: bool = True):
        self.dataset = dataset_tensor
        self.dataset_len = dataset_tensor.size(0)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = (self.dataset_len + batch_size - 1) // batch_size

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(
                self.dataset_len, device=self.dataset.device
            )
        else:
            self.indices = torch.arange(
                self.dataset_len, device=self.dataset.device
            )

        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.dataset_len:
            raise StopIteration

        batch_indices = self.indices[self.idx : self.idx + self.batch_size]
        batch = self.dataset[batch_indices]
        self.idx += self.batch_size
        return batch

    def __len__(self):
        return self.num_batches