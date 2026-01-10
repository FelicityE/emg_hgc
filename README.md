
---

# 💫 emg_hgc

A Python module for **EMG Hand Gesture Classification (HGC)** with a pipeline for:

* EMG dataset loading.
* Preprocessing & segmentation.
* Feature extraction.
* Aggregation into model-ready matrices.
* Training & hyperparameter optimization for CNN-based gesture models.

Designed for **research**, **reproducibility**, and **dynamic customization** of EMG workflows.

---

## 🔧 Features

* **Unified EMG dataset class (`EMG`)**  
  Load, clean, preprocess, segment, and extract features — all from one object.

* **Flexible segmentation/windowing**

  * Split by label transitions
  * (TBD) Fixed-size windows
  * (TBD) Changepoint-based segmentation
  * (TBD) Rolling windows
  * Dynamic target segmentation for model-ready features

* **Feature extraction**

  * Supports single or multiple feature extraction methods.
  * Built-in FEMs.
  * Extendablility with user-defined FEM modules.

* **TensorFlow-ready datasets (`TFData`)**

  * Shuffling, reshaping, and splitting.
  * Category encoding for gesture classes.
  * Ready-to-train `(X, Y)` pairs.

* **Modeling utilities**

  * Hyperparameter tuner (`HPT`).
  * CNN model builder (`Model`).
  * Training, validation, and testing.
  * Logging & experiment tracking.

---

# 📦 Installation

```bash
git clone https://github.com/FelicityE/emg_hgc
cd emg_hgc
pip install -e .
```

Requires:

* Python 3.10
* NumPy, SciPy, scikit-learn
* TensorFlow 2.10 (Requires Python 3.10)
* Matplotlib, Seaborn (optional)
* tqdm (optional)
* h5py (loading .mat files)

---

# 🚀 Quick Start Example

Below is a complete example for loading EMG data, preprocessing, windowing, extracting features, preparing TensorFlow datasets, running hyperparameter tuning, training a CNN, and logging results.

---

## 1️⃣ Load & Prepare the Dataset

EMG Data used in this example is available from the [Ninapro Repository](https://ninapro.hevs.ch/index.html).

```python
from emg_hgc.data.emg_dataset import EMG

# Load from MATLAB .mat file
data = EMG.from_mat("data/S1_A1_E1.mat")

# Remove rest class (0)
redata = data.remove_stimuli(remove=[0])

# Normalize + abs()
redata.apply_process(rectify=False, normalize=True)

# Window the EMG signal (by stimulus label)
redata.get_windowedData(mode="by_label")

# Extract Features
from emg_hgc.features.extractors import tap
redata.get_afm(mode='dynamic', fems=[tap], nSegments=200, 
               report=True, index_last='avs')

# Convert to model-ready (X, Y)
x, y = redata.to_xy()

# Summarize dataset
redata.summary()
```

Example summary:

```
EMG Dataset Summary
--------------------
Samples: 61951
Channels/Features: 10
Unique stimuli: 12
Unique repetitions: 10
Windowed Data Available
Aggregated Data Shape: (12, 9, 200, 10)
X Available: (108, 200, 10)
Y Available: (108,)
Metadata:
  Source: data/S1_A1_E1.mat
  Attributes: ('emg', 'stimulus', 'repetition')
  Filter: {'removed': [0], 'balanced': False, 'random_state': None}
  Preprocess: {'rectified': False, 'normalized': True}
  Windowed: by_label
  Aggregation: {'segmentation': 'dynamic', 'features': ['tap']}
```

---

## 2️⃣ Build TensorFlow Datasets

```python
from emg_hgc.data.tensor_data import TFData

train = TFData.from_pair(redata.to_xy())
train.shuffle()
train.reshape()

nClasses = len(train.categories)
shape = (train.x.shape[1], train.x.shape[2])

# Create validation, test, and tuning splits
test = TFData.from_pair(train.split(0.2))
val = TFData.from_pair(train.split(0.2))
tune = TFData.from_pair(train.sample(0.2))
```

---

## 3️⃣ Hyperparameter Optimization

```python
from emg_hgc.models.hpo import HPT
from emg_hgc.models.cnn import Model
from emg_hgc.util.logging import Logger

log = Logger()

seg = redata.metadata['aggregation']['segmentation']
dID = 'Ds4S1'
fem = "_".join(redata.metadata['aggregation']['features'])

# Build tuner
hpt = HPT(shape, nClasses, ".tuner/", fem)

# Search best hyperparameters
hpt.search(tune, verbose=False)

# Build CNN model
mod = Model()
mod.set_model(hpt.model)
```


---

## 4️⃣ Train, Test & Log Results

```python
mod.train(train, val)
mod.test(test)

log.log(dID, 'ds', 'TAP', mod)
log.print()
```

Example output:

```
DID      seg    FEM      epc   acc      val_acc  tac     
-------------------------------------------------------
Ds4S1    ds     TAP      11    0.6765   0.5556   0.4545
```

---

# 📁 Directory Structure

```
emg_hgc/
│
├── data/
│   ├── emg_dataset.py      # EMG class
│   ├── preprocessing.py    
│   ├── tensor_data.py      # TensorFlow dataset wrapper
│   ├── windowing.py      # TensorFlow dataset wrapper
│
├── features/
│   ├── aggregation.py
│   ├── extractors.py       # FEMs (TAP, MAV, ZCR, etc.)
│   ├── heuristics.py
│
├── models/
│   ├── cnn.py              # CNN architectures
│   ├── hpo.py              # KerasTuner utilities
│
├── utils/
│   ├── array_ops.py
│   ├── labeling.py
│   ├── logger.py           # Experiment logging
│   ├── plotting.py
│
└── README.md               # ← You are here
```

---

# 🧪 Citing This Work

If you use this package in academic publications, please cite:

Paper: 
> TBD  

Repostitory:
```
@software{Escarzaga_Electromyographic_Hand_Gesture_2025,  
  author = {Escarzaga, Felicity},  
  month = nov,  
  title = {{Electromyographic Hand Gesture Classification (EMG-HGC)}},   
  url = {https://github.com/FelicityE/emg_hgc},  
  version = {1.0},  
  year = {2025}  
}
```
<!-- ---

# 🐛 Issues & Contributions

Pull requests and feature requests are welcome!
If you find a bug or want to request a feature, open an issue on GitHub. -->

---

# 📬 Contact

Developed by **Felicity E.**
For questions or collaborations, please reach out via GitHub issues.
