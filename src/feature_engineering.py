import numpy as np
import re
import pandas as pd
import json


def add_manual_features(df):
    # Add `nameornot` feature (0 indicates no name whereas 1 indicates name is available)
    df['NameorNot'] = np.where(df['Name'].isnull(), 0, 1)

    # Add `strangename` feature (0 indicates a normal name whereas 1 indicates name is considered strange)
    pattern = re.compile(r"[0-9\.:!]")
    df['Name'] = df['Name'].fillna('')
    df['strange_name'] = df['Name'].apply(lambda x: len(pattern.findall(x))>0).astype(np.int8)

    # Name Length
    df['name_len'] = df['Name'].apply(lambda x: len(x))

    # Age in Years
    df['AgeYrs'] = df['Age']//12

    return df


def add_breed_features(df):
    labels_breed = pd.read_csv('../data/breed_labels.csv')
    labels_breed.rename(index=str, columns={'BreedName':'Breed1Name'},inplace=True)
    labels_breed['Breed2Name'] = labels_breed['Breed1Name'].values

    df = df.merge(labels_breed[['BreedID','Breed1Name']], left_on='Breed1', right_on='BreedID', how='left')
    df.drop('BreedID',axis=1,inplace=True)
    df = df.merge(labels_breed[['BreedID','Breed2Name']], left_on='Breed2', right_on='BreedID', how='left')
    df.drop('BreedID',axis=1,inplace=True)
    df['Breed2Name'].fillna('',inplace=True)
    df['BreedName_full'] = df['Breed1Name'] + ' ' + df['Breed2Name']
    df['BreedName_full'].fillna('',inplace=True)
    df['BreedName_full'] = df['BreedName_full'].str.lower()
    df['breed_noname'] = df['BreedName_full'].isnull().astype(np.int8)
    df['breed_num'] = 1
    df['breed_num'] += df['Breed2'].apply(lambda x: 1 if x!=0 else 0)
    df['breed_mixed'] = df['BreedName_full'].apply(lambda x: x.find('mixed')>=0).astype(np.int8)
    df['breed_Domestic'] = df['BreedName_full'].apply(lambda x: x.find('domestic')>=0).astype(np.int8)
    df['pure_breed'] = ((df['breed_num']==1)&(df['breed_mixed']==0)).astype(np.int8)

    return df


def add_sent_scores(df):
    train_id = df['PetID']
    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in train_id:
        try:
            with open('../data/train_sentiment/' + pet + '.json', 'r', encoding='utf-8') as f:
                sentiment = json.load(f)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)

    df['doc_sent_mag'] = doc_sent_mag
    df['doc_sent_score'] = doc_sent_score

    return df


def add_metadata(df):
    pet_ids = df['PetID']

    label_descriptions = []
    label_scores = []
    dominant_reds = []
    dominant_greens = []
    dominant_blues = []
    dominant_scores = []
    dominant_pixel_fracs = []
    bounding_vertex_xs = []
    bounding_vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    num_missing_files = 0
    num_present_files = 0

    if 'AdoptionSpeed' in df.columns:
        metadata_folder = 'train_metadata'
    else:
        metadata_folder = 'test_metadata'

    for pet in pet_ids:
        try:
            with open(f'../data/{metadata_folder}/{pet}-1.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                num_present_files += 1

                if data.get('labelAnnotations'):
                    label_descriptions.append(data['labelAnnotations'][0]['description'])
                    label_scores.append(data['labelAnnotations'][0]['score'])
                else:
                    label_descriptions.append(-1)
                    label_scores.append(-1)

                if data.get('imagePropertiesAnnotation'):
                    dominant_reds.append(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red'])
                    dominant_greens.append(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green'])
                    dominant_blues.append(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue'])
                    dominant_scores.append(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score'])
                    dominant_pixel_fracs.append(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction'])
                else:
                    dominant_reds.append(-1)
                    dominant_greens.append(-1)
                    dominant_blues.append(-1)
                    dominant_scores.append(-1)
                    dominant_pixel_fracs.append(-1)

                if data.get('cropHintsAnnotation'):
                    bounding_vertex_xs.append(data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x'])
                    bounding_vertex_ys.append(data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y'])
                    bounding_confidences.append(data['cropHintsAnnotation']['cropHints'][0]['confidence'])
                else:
                    bounding_vertex_xs.append(-1)
                    bounding_vertex_ys.append(-1)
                    bounding_confidences.append(-1)

                if data.get('importanceFraction'):
                    bounding_importance_fracs.append(data['cropHintsAnnotation']['cropHints'][0]['importanceFraction'])
                else:
                    bounding_importance_fracs.append(-1)

        except FileNotFoundError:
            num_missing_files += 1
            label_descriptions.append(-1)
            label_scores.append(-1)
            dominant_reds.append(-1)
            dominant_greens.append(-1)
            dominant_blues.append(-1)
            dominant_scores.append(-1)
            dominant_pixel_fracs.append(-1)
            bounding_vertex_xs.append(-1)
            bounding_vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)

    print(f'# of Metadata files found: {num_present_files}')
    print(f'# of Metadata files missing: {num_missing_files}')

    df['label_descriptions'] = label_descriptions
    df['label_scores'] = label_scores
    df['dominant_reds'] = dominant_reds
    df['dominant_greens'] = dominant_greens
    df['dominant_blues'] = dominant_blues
    df['dominant_scores'] = dominant_scores
    df['dominant_pixel_fracs'] = dominant_pixel_fracs
    df['bounding_vertex_xs'] = bounding_vertex_xs
    df['bounding_vertex_ys'] = bounding_vertex_ys
    df['bounding_confidences'] = bounding_confidences
    df['bounding_importance_fracs'] = bounding_importance_fracs

    return df
