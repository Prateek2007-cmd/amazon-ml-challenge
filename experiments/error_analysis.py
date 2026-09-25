"""
Error analysis of false positives and false negatives from LightGBM.
"""
import os, sys, time
import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb

sys.path.insert(0, "experiments")
from test_features_and_models import run_experiments

print("Analyzing top FP and FN patterns...")
