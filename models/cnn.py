# Imports
## Data
import numpy as np
from emg_hgc.data import tensor_data

## Model
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score

## Model
class Model:
  def __init__(self):
    self.eStop = EarlyStopping(patience=5, restore_best_weights=True)

  def build_model(self, xShape, nClasses:int, model=None, modelpath=None, compile=True, alpha = 0.001):
    # get inputs
    self.shape = xShape
    self.nClasses = nClasses
    self.alpha = alpha

    # build
    if model is not None:
      self.model = model
    elif model is not None:
      self.load_model(modelpath)
    else:
      self.build(compile)

    # Init
    # train
    self.history = None
    self.epc = None
    
    # predict
    self.predicted = None
    self.observed = None

  def load_model(self, path):
    self.model = load_model(path)

  def set_model(self, model):
    self.model = model

  def build(self, compile=True):
    self.model = Sequential([
      Input(shape=self.shape),  # Explicit input layer
      Conv2D(32, (3, 3), activation='relu'),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D(pool_size=(2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(self.nClasses, activation='softmax')
    ])
    if compile:
      self.compile()
  
  def compile(self):
    if not hasattr(self, 'model'):
      raise ValueError("No model found. Call build or load model.")
    
    self.model.compile(
      optimizer=Adam(learning_rate=self.alpha),
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )
  
  def train(self, tSet:tensor_data.TFData, vSet:tensor_data.TFData):
    if not hasattr(self, 'model'):
      raise ValueError("No model found. Call build or load model.")
    
    self.history = self.model.fit(
      tSet.x, tSet.y,
      epochs=50,
      validation_data=(vSet.x, vSet.y),
      callbacks=[self.eStop],
      verbose=False
    )
    self.epc = np.argmin(self.history.history['val_loss'])
    
  def test(self, test:tensor_data.TFData):
    if not hasattr(self, 'model'):
      raise ValueError("No model found. Call build or load model.")
    
    self.predicted = self.model.predict(test.x)
    self.predicted = self.predicted.argmax(axis=1)
    self.observed = test.y.argmax(axis=1)
    self.accuracy = accuracy_score(self.observed, self.predicted)

  def save_model(self, path):
    if not hasattr(self, 'model'):
      raise ValueError("No model found. Call search.")
    
    self.model.save(f"{path}.h5")
    print(f"Model save to {path}.h5")