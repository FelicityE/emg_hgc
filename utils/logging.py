# Imports
from emg_hgc.models import cnn
## Log
import pickle

class Logger:
  def __init__(self):
    self.records = []

  def log(self, did, seg, fem, mod:cnn.Model):
    self.records.append({
      'DID': did,
      'seg': seg,
      'FEM': fem,
      'epc': mod.epc,
      'acc': mod.history.history['accuracy'],
      'vac': mod.history.history['val_accuracy'],
      'los': mod.history.history['loss'],
      'vlo': mod.history.history['val_loss'],
      'tac': mod.accuracy,
      'prd': mod.predicted,
      'obs': mod.observed
    })

  def save(self, filepath='../Results/og_results.pkl'):
    with open(filepath, 'wb') as f:
      pickle.dump(self.records, f)

  def load(self, filepath='../Results/og_results.pkl'):
    with open(filepath, 'rb') as f:
      self.records = pickle.load(f)

  def print(self, show_full=False):
    print(f"{'DID':<8} {'seg':<6} {'FEM':<8} {'epc':<5} {'acc':<8} {'val_acc':<8} {'tac':<8}")
    print("-" * 50)
    for rec in self.records:
      acc_final = rec['acc'][-1] if rec['acc'] else None
      vac_final = rec['vac'][-1] if rec['vac'] else None
      try: tac = rec['tac']
      except: tac = "N"
      print(f"{rec['DID']:<8} {rec['seg']:<6} {rec['FEM']:<8} {rec['epc']:<5} {acc_final:<8.4f} {vac_final:<8.4f} {tac:<8}")
      if show_full:
        print("  → All Acc:", rec['acc'])
        print("  → All Val Acc:", rec['vac'])
        print("  → Predictions:", rec['prd'])
        print("  → Observed:", rec['obs'])
        print()