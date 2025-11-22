# Imports
## Model
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import accuracy_score

## Optimizer
from tensorflow.keras.layers import Conv1D, MaxPooling1D #type: ignore
from keras_tuner.tuners import BayesianOptimization
import json
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import os
from keras_tuner import HyperParameters

## Optimizer
class HPT:

  def __init__(self, shape, nClasses, path, name):
    self.shape = shape
    self.nClasses = nClasses

    self.build_tuner(path, name)

  def manual_build(self, p):
    hp = HyperParameters()
    hp.Fixed('conv1_filters', p[0])
    hp.Fixed('conv1_kernel', p[1])
    hp.Fixed('conv2_filters', p[2])
    hp.Fixed('conv2_kernel', p[3])
    hp.Fixed('dense_units', p[4])
    hp.Fixed('dropout', p[5])
    hp.Fixed('lr', p[6])

    self.tuner.hypermodel.build(hp)

  def build_model(self, hp):
    # Model
    self.model = Sequential()
    # Input
    self.model.add(Input(shape=self.shape))
    # Layer 1: Cnv
    self.model.add(Conv1D(
      filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
      kernel_size=hp.Choice('conv1_kernel', values=[3, 5, 7]),
      activation='relu'
    ))
    self.model.add(MaxPooling1D(pool_size=2))
    # Layer 2: Cnv
    self.model.add(Conv1D(
      filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=64),
      kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
      activation='relu'
    ))
    self.model.add(MaxPooling1D(pool_size=2))
    # Transform
    self.model.add(Flatten())
    # Layer 3: Dense
    self.model.add(Dense(
      units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
      activation='relu'
    ))
    # Transform
    self.model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.6, step=0.1)))
    # Layer 4: Output
    self.model.add(Dense(self.nClasses, activation='softmax'))
    
    # Compile
    self.model.compile(
      optimizer=Adam(
        learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')
      ),
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )

    return self.model

  def build_tuner(self, path, name):
    self.path = path
    self.name = name
    self.tuner = BayesianOptimization(
      self.build_model,
      objective='val_loss',
      max_trials=20,
      executions_per_trial=3,
      directory=self.path,
      project_name=self.name
    )
    
  def search(self, data, verbose=False):
    if not hasattr(self, 'tuner'):
      raise ValueError("No tuner found. Call build_tuner.")
    
    trial_dir = os.path.join(self.path, self.name, "best_trial_weights")
    os.makedirs(trial_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
      filepath=os.path.join(trial_dir, 'checkpoint.weights.h5'),
      save_weights_only=True,
      save_best_only=True,
      monitor='val_loss',
      mode='min',
      verbose=0
    )
    
    self.tuner.search(
      data.x, data.y,
      epochs = 50,
      validation_split=0.2,
      batch_size=32,
      callbacks=[EarlyStopping(patience=5), checkpoint_cb],
      verbose=0
    )
    self.model = self.tuner.get_best_models(num_models=1)[0]
    self.trial = self.tuner.oracle.get_best_trials(num_trials=1)[0]

    if verbose:
      print(f"Best Trial ID: {self.trial.trial_id}")
      print("Best HP:")
      for k, v in self.trial.hyperparameters.values.items():
        print(f"\t{k}: {v}")
      print('')
  
  def set_prime_model(self):
    self.model = self.tuner.get_best_models(num_models=1)[0]
    self.trial = self.tuner.oracle.get_best_trials(num_trials=1)[0]
    self.hp = self.trial.hyperparameters

  def set_prime_hp(self, verbose=False):
    # No transfer learning from tuned model
    self.hp = self.tuner.get_best_hyperparameters(num_trials=1)[0]
    self.model = self.tuner.hypermodel.build(self.hp)
    if verbose:
      self.print_prime_hp()

  def print_prime_hp(self):
    if not hasattr(self, 'tuner'):
      raise ValueError("No tuner found.")
    print("Best HP:")
    for k, v in self.hp.values.items():
      print(f"\t{k}: {v}")
    print('')

  def save_trial(self,path):
    if not hasattr(self, 'trial'):
      raise ValueError("No trial found. Call search.")
    
    record = {
      'TID': self.trial.trial_id,
      'HP': self.trial.hyperparameters.values.items()
    }

    with open(f'{path}.json','w') as f:
      json.dump([record], f, indent=2)

  def save_model(self, path):
    if not hasattr(self, 'model'):
      raise ValueError("No model found. Call search.")
    
    self.model.save(f"{path}.h5")
    print(f"Model save to {path}.h5")
