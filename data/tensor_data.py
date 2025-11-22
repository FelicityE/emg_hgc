## Data
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

## Data
class TFData:
  def __init__(self, x, y):
    self.x = x
    self.y = y
  
  @classmethod
  def from_pair(self, data):
    x = data[0]
    y = data[1]
    return self(x, y)

  @classmethod
  def load(self, path, FEM):
    x = np.load(f'{path}/{FEM}X.npy')
    y = np.load(f'{path}/{FEM}Y.npy')
    return self(x, y)

  def shuffle(self, seed=42):
    self.x, self.y = shuffle(self.x, self.y, random_state=seed)

  def reshape(self):
    # self.x = self.x.reshape(self.x.shape[0], self.x.shape[1], self.x.shape[2], 1)
    self.x = self.x.reshape(self.x.shape[0], self.x.shape[1], self.x.shape[2])
    encoder = OneHotEncoder(sparse_output=False)
    self.y = encoder.fit_transform(self.y.reshape(-1, 1))
    self.categories = encoder.categories_[0]

  def split(self, ratio, seed=42):
    self.x, x, self.y, y = train_test_split(self.x, self.y, test_size=ratio, random_state=seed)
    return [x, y]

  def sample(self, ratio, seed=42):
    _, x, _, y = train_test_split(self.x, self.y, test_size=ratio, random_state=seed)
    return [x, y]