#!/usr/bin/env python
# coding: utf-8

# Setting azure ML Workspace
from azureml.core import Workspace
ws = Workspace.create(name='myworkspace', 
                      subscription_id='0fbd82a9-17d7-4692-ad62-4913c0ce69a1', 
                      resource_group='testml', 
                      create_resource_group=True, 
                      location='westus2')

ws.get_details()
ws.write_config()

import numpy as np
import azureml.core
from azureml.core import Workspace

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

# load workspace configuration from the config.json file in the current folder.
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, ws.location, sep='\t')

# Setting the experiment environment in workspace
experiment_name = 'dbscantest'
from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)

# Set the Azure cluster
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

# choose a name for your cluster
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpu-cluster1")
compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

# This example uses GPU VM. set SKU to STANDARD_NC6
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_NC6")

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found compute target. just use it. ' + compute_name)
else:
    print('creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                min_nodes = compute_min_nodes, 
                                                                max_nodes = compute_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
    
    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
     # For a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())

# Upload the data.csv (training data) to Blob Storage in Workspace
data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)

ds.upload(src_dir=data_folder, target_path='dbscndata', overwrite=True, show_progress=True)


import os
script_folder = os.path.join(os.getcwd(), "dbscn")
os.makedirs(script_folder, exist_ok=True)

# wrtie the train.py
get_ipython().run_cell_magic('writefile', '$script_folder/train.py', '\nimport argparse\nimport os\nimport numpy as np\n\nfrom sklearn.cluster import DBSCAN\nfrom sklearn import metrics\nfrom sklearn.datasets.samples_generator import make_blobs\nfrom sklearn.preprocessing import StandardScaler\n\nfrom sklearn.externals import joblib\n\nfrom azureml.core import Run\nfrom utils import load_data\n\nimport subprocess\nimport sys\n\ndef install(package):\n    subprocess.call([sys.executable, "-m", "pip", "install", package])\n\ninstall(\'pandas\')\ninstall(\'azure-storage\')\ninstall(\'tables\')\n\nimport pandas as pd\n\n# let user feed in parameters, the location of the data files (from datastore),\nparser = argparse.ArgumentParser()\nparser.add_argument(\'--data-folder\', type=str, dest=\'data_folder\', help=\'data folder mounting point\')\nargs = parser.parse_args()\n\ndata_folder = args.data_folder\nprint(\'Data folder:\', data_folder)\n\n# load train and test set into numpy arrays\nData_training = pd.read_csv(os.path.join(data_folder, \'data.csv\'))\nData_training = StandardScaler().fit_transform(Data_training)\n\n# get hold of the current run\nrun = Run.get_context()\n\ndb = DBSCAN(eps=2, min_samples=10).fit(Data_training)\ncore_samples_mask = np.zeros_like(db.labels_, dtype=bool)\ncore_samples_mask[db.core_sample_indices_] = True\nlabels = db.labels_\n\nn_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)\nn_noise_ = list(labels).count(-1)\n\nprint(\'Estimated number of clusters: %d\' % n_clusters_)\nprint(\'Estimated number of noise points: %d\' % n_noise_)\n\nprint("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Data_training, labels))\n\ndb.core_sample_indices_\ndb.components_\n\nos.makedirs(\'outputs\', exist_ok=True)\npd.DataFrame(db.components_).to_csv("outputs/model.csv", header=None, index=None)\n\nfrom azure.storage.blob import BlockBlobService\nimport tables\n\nSTORAGEACCOUNTNAME = "datatest123"\nLOCALFILENAME = "outputs/model.csv"\nSTORAGEACCOUNTKEY = "DHfLH+rw0qOUya7ihZQp5+7lA4Ezo1hdonfqsQZGw+HZ6vORqjMJpzgSQ/kxIiRDoWFEQzHI7P7xIzRlVWW08w=="\nCONTAINERNAME= "testconta"\nBLOBNAME= "model/model.csv"\n\noutput_blob_service=BlockBlobService(account_name=STORAGEACCOUNTNAME,account_key=STORAGEACCOUNTKEY)    \nlocalfileprocessed = os.path.join(os.getcwd(),LOCALFILENAME) #assuming file is in current working directory\ntry:\n    output_blob_service.create_blob_from_path(CONTAINERNAME,BLOBNAME,localfileprocessed)\nexcept:            \n    print ("Something went wrong with uploading to the blob:"+ BLOBNAME)\n\n# note file saved in the outputs folder is automatically uploaded into experiment record\n# joblib.dump(value=clf, filename=\'outputs/sklearn_mnist_model.pkl\')')

import shutil
shutil.copy('utils.py', script_folder)

from azureml.train.sklearn import SKLearn

script_params = {
    '--data-folder': ds.path('dbscndata').as_mount()
}

#establish the estimator for learning
est = SKLearn(source_directory=script_folder,
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py')

print(ds.path('dbscndata').as_mount())

# upload the estimator
run = exp.submit(config=est)
run

# start training process
from azureml.widgets import RunDetails
RunDetails(run).show()

# register model 
model = run.register_model(model_name='dbscan', model_path='outputs/model.csv')
print(model.name, model.id, model.version, sep='\t')
