from extract_image_features import *
from extract_svd_features import *
from feature_engineering import *
from data_cleaning import *
from scoring import *

from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os
import re
import json
import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
import lightgbm as lgb
from functools import partial
from math import sqrt
import scipy as sp

from keras.applications.densenet import preprocess_input, DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error


pd.set_option('display.max_columns', None)
np.random.seed(1234)

train = pd.read_csv("../data/train/train.csv")
test = pd.read_csv("../data/test/test.csv")

train, test = prep_data(train, test)
results = run_cv_model(train, test, target, runLGB, params, rmse, 'lgb')
train_preds, qwk, conf_matrix = get_train_predictions(results)
test_preds = get_test_predictions(results)
