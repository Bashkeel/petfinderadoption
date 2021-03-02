import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size]) # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_train_image(pet_id):
    path = f'{Path(os.getcwd()).parents[0]}\\data\\train_images\\'
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image


def load_test_image(pet_id):
    path = f'{Path(os.getcwd()).parents[0]}\\data\\test_images\\'
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image

def extract_image_features(train, test):
    inp = Input((256,256,3))
    backbone = DenseNet121(input_tensor = inp, include_top = False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:,:,0])(x)
    m = Model(inp,out)

    # Train Images
    pet_ids = train['PetID']
    img_size = 256
    batch_size = 16
    n_batches = len(pet_ids) // batch_size + 1
    features = {}

    for b in range(n_batches):
        if b%10 == 0: 
            print(f'Processing Batch #{b}')
        start = b*batch_size
        end = (b+1)*batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
        for i,pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_train_image(pet_id)
            except:
                pass
        batch_preds = m.predict(batch_images)
        for i,pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]

    train_feats = pd.DataFrame.from_dict(features, orient='index')
    train_feats.columns = ['pic_'+str(i) for i in range(train_feats.shape[1])]
    train_feats['PetID'] = train_feats.index
    
    train = pd.merge(train, train_feats, on='PetID')
    train.to_csv("../data/processed/train_images.csv")


    #Test Images
    pet_ids = test['PetID']
    img_size = 256
    batch_size = 16
    n_batches = len(pet_ids) // batch_size + 1
    features = {}

    for b in range(n_batches):
        if b%10 == 0: 
            print(f'Processing Batch #{b}')
        start = b*batch_size
        end = (b+1)*batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
        for i,pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_test_image(pet_id)
            except:
                pass
        batch_preds = m.predict(batch_images)
        for i,pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]

    test_feats = pd.DataFrame.from_dict(features, orient='index')
    test_feats.columns = ['pic_'+str(i) for i in range(test_feats.shape[1])]
    test_feats['PetID'] = test_feats.index

    test = pd.merge(test, test_feats, on='PetID')
    test.to_csv("../data/processed/test_images.csv")

    return train, test