

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# login into my google drive account
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# main functions are in this file
learning_lib = drive.CreateFile( {'id' : '1wwSN3AIl_dmayKENu5jnc1BRaNPe8BZc'}).GetContentFile("learning.py")

"""### Installing Required Libraries"""

# Commented out IPython magic to ensure Python compatibility.
# Install tensorflow
try:
    # tensorflow_version only exists in Colab
#     %tensorflow_version 2.2.0
except Exception:
    pass

# install required libraries
!pip install pyagrum==0.13.6
!pip install lime
!pip install shap

"""### Importing Libraries"""

# Commented out IPython magic to ensure Python compatibility.
from IPython.core.display import HTML
import numpy as np
import pandas as pd
import random as rn
import time

# current explanable algorithms
import lime
import shap
from lime import lime_tabular

# import auxiliary functions
from learning import *

# %matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""### Loading Black Box"""

# path to project folder
PATH = "/content/drive/My Drive/Colab Notebooks/"

"""#### Setting up data"""

# name of dataset
DATASET_NAME = "cancer.csv"
class_var = "Outcomes"

# load dataset
dataset_path = PATH + "datasets/" + DATASET_NAME
data = pd.read_csv( dataset_path )

# features
feature_names = data.drop([class_var], axis=1).columns.to_list()
print("Features")
print(feature_names)

# balance dataset
sampled_data = data.sample(frac=1)
sampled_data = sampled_data[ sampled_data["Outcomes"] == 0]

no_data = sampled_data.sample(frac=1)[0:268]
yes_data = data[ data["Outcomes"] == 1]

balanced_data = [no_data,yes_data]
balanced_data = pd.concat(balanced_data)

"""#### Load Trained Model"""

# apply one hot encoder to data
# standardize the input between 0 and 1
X, Y, encoder, scaler = encode_data( balanced_data, class_var)

n_features = X.shape[1]
n_classes = len(data[class_var].unique())

# load existing training data
print("Loading training data...")
X_train, Y_train, X_test, Y_test, X_validation, Y_validation= load_training_data( dataset_path )

# the best performing model was obtained with 5 hidden layers with 12 neurons each
model_name = "model_h5_N12"

# specify paths where the blackbox model was saved
path_serialisation_model = PATH + "training/" + DATASET_NAME.replace(".csv", "") + "/model/" 
path_serialisation_histr = PATH + "training/" + DATASET_NAME.replace(".csv", "") + "/history/" 

# load model and model performance history
print("Loading Blackbox model...")
model_history = load_model_history( model_name, path_serialisation_histr )
model = load_model( model_name, path_serialisation_model )

# check model
model.summary()

"""#### Evaluate Model"""

# evaluate loaded model on test and training data
optim = keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=1)
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)

print('\n[Accuracy] Train: %.3f, Test: %.3f' % (train_acc, test_acc))
print('[Loss] Train: %.3f, Test: %.3f' % (train_loss, test_loss))

"""### Searching for specific datapoints for local evaluation"""

# creates a dictionary with the following information:
# ground_truth
# index
# original_vector
# scaled_vector
# prediction_type
local_data_dict = generate_local_predictions( X_test, Y_test, model, scaler, encoder )

# separates vectors into true positives, true negatives
# false positives and false negatives
true_positives,true_negatives, false_positives, false_negatives = wrap_information( local_data_dict )

# add class variable to the feature list
feature_names.append("Relapse?")

# Example of a True Positve Instance
len(true_positives)

"""## Generating Explanations with Bayesian Networks"""

label_lst = ["No", "Yes"]
class_var = "Relapse?"

VAR = 0.1

feature_names_cp = feature_names.copy()
feature_names_cp.append("Relapse?")
feature_names.remove('Relapse?')

feature_names_cp

len(true_positives)

"""### Examples for Users

#### Example of Explanation for a Single True Positive
"""

instance =true_negatives[1]

[bn, inference, infoBN, markov_blanket] = generate_BN_explanationsMB(instance, label_lst, feature_names, class_var, 
                                                                       encoder, scaler, model, PATH, DATASET_NAME, variance = VAR)
inference = gnb.getInference(bn, evs={ 'Relapse?' : 'No' },targets=feature_names_cp )
gnb.sideBySide(*[inference, markov_blanket, infoBN ], captions=[ "Inference", "Markov Blanket", "Information BN" ])

#plt.tight_layout()
#explanation_plot.savefig('/content/drive/My Drive/Colab Notebooks/lime2.png', dpi=300, bbox_inches='tight')

gnb.showInference(bn,size="12")

MAX_FEAT = 5

start_time = time.time()

# LIME has one explainer for all the models
explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names= feature_names, 
                                                  class_names=["Relapse","No Replse"], verbose=False, mode='classification')

elapsed = time.time() - start_time

print ("Time taken to create explainer:", round(elapsed, 2), "seconds")

start_time = time.time()
# explain instance
patients_feat = np.array(instance['scaled_vector'])
exp = explainer.explain_instance(patients_feat, model.predict, num_features = MAX_FEAT)

# Show the predictions
exp.show_in_notebook(show_table=True)
elapsed = time.time() - start_time
print("Time taken to provide explanation", round (elapsed, 2), "seconds")

#Save the explanation
explanation_plot = exp.as_pyplot_figure()
#plt.tight_layout()
plt.savefig('/content/drive/My Drive/Colab Notebooks/lime_mrmr_zcube.png', dpi=300, bbox_inches='tight')
plt.show()
exp.save_to_file('/content/drive/My Drive/Colab Notebooks/lime_mrmr_zcube.html')

import shap

#Convert to dataframe
X_train_df = pd.DataFrame(data = X_train, columns = feature_names)

# load JS visualization code to notebook
shap.initjs()
shap_explainer = shap.KernelExplainer(model.predict, X_train_df)

instance_df = pd.DataFrame(data = [instance['scaled_vector']], columns = feature_names)

shap_values = shap_explainer.shap_values(instance_df)
exp = shap.force_plot(shap_explainer.expected_value[1], shap_values[1], np.around(instance_df, decimals=2), matplotlib = True, show = False)
plt.savefig('/content/drive/My Drive/Colab Notebooks/shap_cancer.png', dpi=300, bbox_inches='tight')

