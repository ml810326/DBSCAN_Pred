import argparse
import os
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn.externals import joblib

from azureml.core import Run
from utils import load_data

import subprocess
import sys

# install pandas, azure storage, and tables component
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install('pandas')
install('azure-storage')
install('tables')

import pandas as pd

# let user feed in parameters, the location of the data files (from datastore),
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()

data_folder = args.data_folder
print('Data folder:', data_folder)

# load train and test set into numpy arrays
Data_training = pd.read_csv(os.path.join(data_folder, 'data.csv'))
Data_training = StandardScaler().fit_transform(Data_training)

# get hold of the current run
run = Run.get_context()

db = DBSCAN(eps=2, min_samples=10).fit(Data_training)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# print the estimated number
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Data_training, labels))

db.core_sample_indices_
db.components_

#save to model.csv
os.makedirs('outputs', exist_ok=True)
pd.DataFrame(db.components_).to_csv("outputs/model.csv", header=None, index=None)

#upload to Blob Storage
from azure.storage.blob import BlockBlobService
import tables

STORAGEACCOUNTNAME = "datatest123"
LOCALFILENAME = "outputs/model.csv"
STORAGEACCOUNTKEY = "<STORAGEACCOUNTKEY>"
CONTAINERNAME= "<CONTAINERNAME>"
BLOBNAME= "model/model.csv"

output_blob_service=BlockBlobService(account_name=STORAGEACCOUNTNAME,account_key=STORAGEACCOUNTKEY)    
localfileprocessed = os.path.join(os.getcwd(),LOCALFILENAME) #assuming file is in current working directory
try:
    output_blob_service.create_blob_from_path(CONTAINERNAME,BLOBNAME,localfileprocessed)
except:            
    print ("Something went wrong with uploading to the blob:"+ BLOBNAME)
