import os
import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import check_grad
from scipy.special import logsumexp
from Bio import SeqIO
import random
import multiprocessing
import multiprocessing as mp
import ctypes
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import argparse
import sys
sys.path.append('src')