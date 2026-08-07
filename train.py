import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import BATCH, FORECAST_LR, RNN_LR, HORIZON, LAMBDA_E
from data import SequenceDataset
from pqc_forecaster import Forecaster, fidelity
from classical_baselines import RNNForecaster