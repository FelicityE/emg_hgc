"""
emg_hgc – EMG Hand Gesture Classification Toolbox
Provides dataset handling, windowing, feature extraction,
and neural network models for EMG HGC workflows.
"""

__version__ = "1.0"

# Re-export main classes for convenience
from .data.emg_dataset import EMG
from .data.tensor_data import TFData
from .utils.logging import Logger

from .features import extractors

from .models.hpo import HPT
from .models.cnn import Model

__all__ = [
  "EMG",
  "TFData",
  "Logger",
  "extractors",
  "HPT",
  "Model",
]
