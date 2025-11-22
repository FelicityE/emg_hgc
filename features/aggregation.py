import numpy as np
from emg_hgc.utils import array_ops

def apply_aggregation(data:list, fems:list=[], mode:str='dynamic',  **kwargs):
  """
  Dispatch to the appropriate aggregation method.

  Returns:
  - Numpy ndarray indexed [Stimulus][Repetition][Feature Method][Aggregated Voltage Sample][Channel]
  """
  if mode == 'dynamic':
    return dynamic_aggregation(data, fems, **kwargs)
  elif mode == 'static':
    return dynamic_aggregation(data, fems, **kwargs)
  else:
    raise ValueError(f"Unknown aggregation mode '{mode}'.")

def dynamic_aggregation(
  data:list, fems:list=[], nSegments:int=200, 
  replace=True, mu=0, std_dev=0.5, report=False,
  padMode='linear_ramp', indexLast='channels', **kwargs
):
  """
  Aggregates EMG data using a list of FEMs

  Parameters: 
    - data (list of lists): EMG data structured such that `Data[Stimulus][Repetition][Voltage Samples][Channels]`
    - fems (list of methods): Functions which given a matrix (N, m) apply a specific feature extraction and output an array of size m.
    - nSegments (int): Number of output aggregated segments for each matrix
    - Modify bad samples:
      - replace (bool): Replace nan gesture window samples with similar repetition
      - mu (float): Add noise to replaced sample with mean mu
      - std_dev (float): Add noise to replaced sample with standard deviation std_dev
      - report: Notice when a sample is modified
      - padMode: Options for padding short samples
    - indexLast: Modify which index is reported last e.g. 'channels', 'avs', or 'features'

  Returns: 
  - rtn (Numpy ndarray): n dimensional array of matrices where each matrix is `rtn[Stimulus][Repetition][Feature Method][Aggregated Voltage Sample][Channel]
  *Index

  """
  nStm = len(data)
  nRep = len(data[0])
  nCh = len(data[0][0][0])
  nFea = len(fems)
  if nFea > 1:
    rtn = np.empty((nStm, nRep, nFea, nSegments, nCh), dtype=float)
  else:
    if indexLast == 'features':
      raise ValueError(f"Only one feature was requested.")

    rtn = np.empty((nStm, nRep, nSegments, nCh), dtype=float)

  for s in range(len(data)):
    rmList = []
    for r in range(len(data[s])):
      # Evenly distribute voltage samples into subsegments such that the output matrix is the same size for all gesture samples
      window = data[s][r].copy()
      windowLength = len(window)
      if windowLength < nSegments:
        if report: 
          print(f'Window Length for [{s},{r}] is less than the number of segments {nSegments}.')
          print(windowLength)
          print(f'Padding using {padMode}...')
        window = array_ops.pad(window, nSegments, padMode)
        windowLength = len(window)

      stride = int(windowLength/nSegments)
      duration = int(windowLength-(nSegments*stride))
      
      # Get the Feature Matrix for each method
      if len(fems) > 1:
        for f, fem in enumerate(fems):
          for i in range(nSegments-1):
            if i >= duration:
              stp = duration*(stride+1)
              indx = (duration-i)
              strt = stp+(indx*stride)
              fnsh = stp+((indx+1)*stride)
            else:
              strt = i*(stride+1)
              fnsh = (i+1)*(stride+1)
            
            temp = window[strt:fnsh]
            rtn[s][r][f][i] = fem(temp, **kwargs)

          if np.isnan(rtn[s][r][f]).any():
            rmList.append([s, r, f]);
      else:
        for i in range(nSegments-1):
          if i >= duration:
            stp = duration*(stride+1)
            indx = (duration-i)
            strt = stp+(indx*stride)
            fnsh = stp+((indx+1)*stride)
          else:
            strt = i*(stride+1)
            fnsh = (i+1)*(stride+1)
          
          temp = window[strt:fnsh]
          rtn[s][r][i] = fems[0](temp, **kwargs)

          if np.isnan(rtn[s][r]).any():
            rmList.append([s, r]);


    if rmList and replace:
      if report:
        print("WARNING: The following [Stimulus, Repetitions] are empty: ")
        print(rmList)
        print("Replacing with similar repetition...")
      for rm in sorted(rmList, reverse=[True]):
        try:
          og = rtn[rm[0],rm[1],rm[2]-1]
        except:
          og = rtn[rm[0],rm[1],rm[2]+1]
        noise = np.random.normal(mu, std_dev, size=og.shape)
        rtn[rm[0],rm[1],rm[2]] = og+noise
  
  if indexLast == 'channels':
    return rtn
  elif indexLast == 'avs':
    return rtn.transpose((0,1,2,4,3))
  elif indexLast == 'features':
    return rtn.transpose((0,1,3,4,2))



def static_aggregation(data, fems:list=[], nSegments:int=200, **kwargs):
  raise NotImplementedError("Fixed windowing not yet implemented.")