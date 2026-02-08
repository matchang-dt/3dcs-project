from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    """
    Dummy dataset for the MVSPlat model.
    """
    def __init__(self):
        """
        Initialize the DummyDataset.
        Args:
            dtype (torch.dtype): data type
        """
        super().__init__()
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return 0