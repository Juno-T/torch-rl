import math
import numpy as np

def DL_2_LD(DL):
  return [dict(zip(DL,t)) for t in zip(*DL.values())]

def LD_2_DL(LD):
  return {k: [dic[k] for dic in LD] for k in LD[0]}

def round_any(x, n=2):
  if isinstance(x, list):
    return [round_any(v, n) for v in x]
  if isinstance(x, dict):
    for k in x:
      x[k] = round_any(x[k], n)
    return x
  if isinstance(x, np.ndarray):
    return np.array([round_any(v) for v in x])
  return round_to_n(x, n)

def round_to_n(x, n): 
  if isinstance(x, bool):
    return x
  if isinstance(x, str):
    return x
  return x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))