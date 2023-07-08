import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import numpy as np

DATA_DIR = '/var/lib/alpha/omega/dyson/dyson/data'

DATA_FN = 'SPY_1T_2016_2017'

FILE_PATH = DATA_DIR + '/' + DATA_FN + '.parquet'

class PhotonDataset(Dataset):

  def __init__(self, file_path, dtype):

    self.file_path = file_path
    self.dtype = dtype

    self.full_bars = pq.read_table(self.file_path).to_pandas().astype(self.dtype )

    x = self.full_bars.iloc[:20].values
    y = self.full_bars.iloc[20:].values

    self.x_train = torch.tensor(x, dtype=torch.float32)
    self.y_train = torch.tensor(y, dtype=torch.float32)

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self, idx):
    return self.x_train[idx], self.y_train[idx]


dataset = PhotonDataset(FILE_PATH, np.float32)






