import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import cv2
import os
from pathlib import Path
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

from feature_engineering import *
from extract_image_features import *
from extract_svd_features import *

def clean_data(df):
    numeric_cols = ['Age', 'Quantity', 'Fee', 'PhotoAmt', 'name_len', 'AgeYrs',
                    'doc_sent_mag', 'doc_sent_score', 'label_scores', 'dominant_reds',
                    'dominant_greens', 'dominant_blues','dominant_scores',
                    'dominant_pixel_fracs', 'bounding_vertex_xs', 'bounding_vertex_ys',
                    'bounding_confidences', 'bounding_importance_fracs'] +\
                   [col for col in df.columns if col.startswith('pic') or col.startswith('svd')]

    cat_cols = list(set(df.columns) - set(numeric_cols))
    df[cat_cols] = df[cat_cols].astype('category')

    return df


def prep_data(train, test):
    for df in [train, test]:
        df = add_manual_features(df)
        df = add_breed_features(df)
        df = add_sent_scores(df)
        df = add_metadata(df)
    train, test = extract_image_features(train, test)
    train, test = transform_tfidf_svd(train, test)

    train = clean_data(train)
    train.to_csv("../data/processed/train_processed.csv", index=False)

    test = clean_data(test)
    test.to_csv("../data/processed/test_processed.csv", index=False)

    return train, test
