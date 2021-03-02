from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os

api = KaggleApi()
api.authenticate()
api.competition_download_files('petfinder-adoption-prediction')

zf = ZipFile('petfinder-adoption-prediction.zip')
zf.extractall('../data/')
zf.close()

os.remove("petfinder-adoption-prediction.zip")
print("Pet Adoption dataset is downloaded and ready to use!")


api.dataset_download_file(dataset='xhlulu/densenet-keras/',
                         file_name='DenseNet-BC-121-32-no-top.h5')

zf = ZipFile('DenseNet-BC-121-32-no-top.h5.zip')
zf.extractall('../models/')
zf.close()

os.remove("DenseNet-BC-121-32-no-top.h5.zip")
print("DenseNet Weights are downloaded and ready to use!")
