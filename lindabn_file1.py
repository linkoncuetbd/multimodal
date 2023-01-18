

from google.colab import drive
drive.mount('/content/drive')

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# login into my google drive account
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

learning_lib = drive.CreateFile( {'id' : '1wwSN3AIl_dmayKENu5jnc1BRaNPe8BZc'}).GetContentFile("learning.py")

# Commented out IPython magic to ensure Python compatibility.
# Install tensorflow
try:
    # tensorflow_version only exists in Colab
#     %tensorflow_version 2.x
except Exception:
    pass

# library to deal with Bayesian Networks
!pip install pyagrum==0.13.6
!pip install lime
!pip install shap

import lime
from lime import lime_tabular

# Commented out IPython magic to ensure Python compatibility.
# for reproduciability reasons:
import numpy as np
import pandas as pd
import random as rn
import time

# %matplotlib inline

# import auxiliary functions
from learning import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""### Checking Dataset"""

# path to project folder
# please change to your own
#PATH = "/Users/catarina/Google Drive/Colab Notebooks/DDS/"
PATH = "/content/drive/My Drive/Colab Notebooks/"

# name of dataset
DATASET_NAME = "cancer.csv"

# variable containing the class labels in this case the dataset contains:
# 0 - if not High Risk
# 1 - if High Risk
class_var = "Outcomes"

# load dataset
dataset_path = PATH + "datasets/" + DATASET_NAME
data = pd.read_csv( dataset_path )
data

# features
feature_names = data.drop([class_var], axis=1).columns.to_list()

# check how balanced the classes are
data.groupby(class_var).count()

"""### Balanced Dataset"""

# balance dataset
sampled_data = data.sample(frac=1)
sampled_data = sampled_data[ sampled_data["Outcomes"] == 0]
no_data = sampled_data.sample(frac=1)[0:268]

yes_data = data[ data["Outcomes"] == 1]

balanced_data = [no_data,yes_data]
balanced_data = pd.concat(balanced_data)

# check how balanced the classes are
balanced_data.groupby(class_var).count()

"""#### Train a Model for the Balanced Dataset"""

# apply one hot encoder to data
# standardize the input between 0 and 1
X, Y, encoder, scaler = encode_data( balanced_data, class_var)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
n_features = X.shape[1]
n_classes = len(balanced_data[class_var].unique())
 
flag=True # DO NOT CHANGE! Data has already been generated. 
if flag:
    # save training, test and validation data
    generate_save_training_data( dataset_path, X, Y)
    
else:
    # load existing training data
    X_train, Y_train, X_test, Y_test, X_validation, Y_validation= load_training_data( dataset_path )

# generate models for grid search
if flag:
    models = grid_search_model_generator( n_features, n_classes )

    # perform grid_search
    HISTORY_DICT = perform_grid_search( models, PATH, DATASET_NAME.replace(".csv",""), 
                                   X_train, Y_train, 
                                   X_validation, Y_validation, X_test, Y_test, 
                                   batch_size=8, epochs=150 )

path_serialisation_model = PATH + "training/" + DATASET_NAME.replace(".csv", "") + "/model/" 
path_serialisation_histr = PATH + "training/" + DATASET_NAME.replace(".csv", "") + "/history/" 

# the best performing model was obtained with 5 hidden layers with 12 neurons each
model_name = "model_h5_N12"
    
if flag:
    
    # get respective model training history and model
    model_history = HISTORY_DICT[ model_name ][0]
    model = HISTORY_DICT[ model_name ][1]

    # save model and model history to file
    save_model_history(  model_history, model_name, path_serialisation_histr )
    save_model( model, model_name, path_serialisation_model )
else:
    model_history = load_model_history( model_name, path_serialisation_histr )
    model = load_model( model_name, path_serialisation_model )
    
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

local_data_dict = generate_local_predictions( X_train, Y_train, model, scaler, encoder )

# wrapping up information
true_positives = []
true_negatives = []
false_positives = []
false_negatives = []
for instance in local_data_dict:
    
    if( instance['prediction_type'] == 'TRUE POSITIVE'):
        true_positives.append(instance)

    if( instance['prediction_type'] == 'TRUE NEGATIVE' ):
        true_negatives.append(instance)
        
    if( instance['prediction_type'] == 'FALSE POSITIVE' ):
        false_positives.append(instance)
        
    if( instance['prediction_type'] == 'FALSE NEGATIVE' ):
        false_negatives.append(instance)

"""### Generating Explanations with Bayesian Networks"""

label_lst = ["No", "Yes"]
class_var = "Alzheimer?"

VAR = 0.8

"""#### TRUE POSITIVES"""

print([len(true_positives), len(true_negatives), len(false_positives), len(false_negatives)])

RULE1 = 0
RULE2 = 0
RULE3 = 0
RULE4 = 0
RULE5 = 0

for instance in true_positives:

    # get instance index
    indx = instance['index']
    print("INDEX = " + str(indx))
    
    [bn, inference, infoBN, markov_blanket] = generate_BN_explanationsMB(instance, label_lst, feature_names, 
                                                        class_var, encoder, scaler, model, PATH, DATASET_NAME, 
                                                        variance = VAR)
    ie=gum.LazyPropagation(bn)
    ie.makeInference()
    pos_class = ie.posterior(class_var)

    indx_yes = -1
    indx_no = -1

    if( len(markov_blanket.nodes()) == 1 ):
        RULE2 = RULE2+1    
        continue
    
    if (len(bn.variableFromName(class_var).labels()) == 1 ):

        if( bn.variableFromName(class_var).labels()[0] == "Yes" ):
            if( pos_class[0] >= 0.90 ):
                RULE1 = RULE1 + 1
                print("RULE 1")
        #if( bn.variableFromName(class_var).labels()[0] == "No" ):
        #    gnb.sideBySide(*[bn, inference, markov_blanket  ], captions=[ "Bayesian Network", "Inference", "Markov Blanket" ])

              
    if (len(bn.variableFromName(class_var).labels()) == 2 ):
        value_yes = pos_class[1]
        value_no = pos_class[0]

        if( value_yes >= 0.90 ):
            RULE1 = RULE1 + 1
            print("RULE 1")

        if( (value_no > value_yes) & (np.round(value_no) == 1) ):
            RULE3 = RULE3 + 1
            print("RULE 3")


        if( (value_yes > value_no) & (value_yes < 0.9) ):
            print("RULE 4 or 5")
            RULE4 = RULE4 + 1
            #gnb.sideBySide(*[bn, inference, markov_blanket  ], captions=[ "Bayesian Network", "Inference", "Markov Blanket" ])

print( [RULE1, RULE2, RULE3, RULE4] )

len(true_positives)

"""#### TRUE NEGATIVES"""

print([len(true_positives), len(true_negatives), len(false_positives), len(false_negatives)])

RULE1 = 0
RULE2 = 0
RULE3 = 0
RULE4 = 0
RULE5 = 0

for instance in true_negatives:

    # get instance index
    indx = instance['index']
    print("INDEX = " + str(indx))
    
    [bn, inference, infoBN, markov_blanket] = generate_BN_explanationsMB(instance, label_lst, feature_names, 
                                                        class_var, encoder, scaler, model, PATH, DATASET_NAME, 
                                                        variance = VAR)
    ie=gum.LazyPropagation(bn)
    ie.makeInference()
    pos_class = ie.posterior(class_var)

    indx_yes = -1
    indx_no = -1
    
    if( len(markov_blanket.nodes()) == 1 ):
        RULE2 = RULE2+1
        continue
        
    if (len(bn.variableFromName(class_var).labels()) == 1 ):

        if( bn.variableFromName(class_var).labels()[0] == "No" ):
            if( pos_class[0] >= 0.90 ):
                RULE1 = RULE1 + 1
                print("RULE 1")
        #if( bn.variableFromName(class_var).labels()[0] == "Yes" ):
        #    gnb.sideBySide(*[bn, inference, markov_blanket  ], captions=[ "Bayesian Network", "Inference", "Markov Blanket" ])

              
    if (len(bn.variableFromName(class_var).labels()) == 2 ):
        value_yes = pos_class[1]
        value_no = pos_class[0]

        if( value_no >= 0.90 ):
            RULE1 = RULE1 + 1
            print("RULE 1")

        if( (value_yes > value_no) & (np.round(value_yes) == 1) ):
            RULE3 = RULE3 + 1
            print("RULE 3")


        if( (value_no > value_yes) & (value_no < 0.9) ):
            print("RULE 4 or 5")
            RULE4 = RULE4 + 1
            #gnb.sideBySide(*[bn, inference, markov_blanket  ], captions=[ "Bayesian Network", "Inference", "Markov Blanket" ])

print( [RULE1, RULE2, RULE3, RULE4] )

"""#### FALSE POSITIVES"""

print([len(true_positives), len(true_negatives), len(false_positives), len(false_negatives)])

RULE1 = 0
RULE2 = 0
RULE3 = 0
RULE4 = 0
RULE5 = 0

for instance in false_positives:

    # get instance index
    indx = instance['index']
    print("INDEX = " + str(indx))
    
    [bn, inference, infoBN, markov_blanket] = generate_BN_explanationsMB(instance, label_lst, feature_names, 
                                                        class_var, encoder, scaler, model, PATH, DATASET_NAME, 
                                                        variance = VAR)
    ie=gum.LazyPropagation(bn)
    ie.makeInference()
    pos_class = ie.posterior(class_var)

    indx_yes = -1
    indx_no = -1

    if( len(markov_blanket.nodes()) == 1 ):
        RULE2 = RULE2+1
        continue
    
    if (len(bn.variableFromName(class_var).labels()) == 1 ):

        if( bn.variableFromName(class_var).labels()[0] == "Yes" ):
            if( pos_class[0] >= 0.90 ):
                RULE1 = RULE1 + 1
                print("RULE 1")
        #if( bn.variableFromName(class_var).labels()[0] == "No" ):
            #gnb.sideBySide(*[bn, inference, markov_blanket  ], captions=[ "Bayesian Network", "Inference", "Markov Blanket" ])

              
    if (len(bn.variableFromName(class_var).labels()) == 2 ):
        value_yes = pos_class[1]
        value_no = pos_class[0]

        if( value_yes >= 0.90 ):
            RULE1 = RULE1 + 1
            print("RULE 1")

        if( (value_no > value_yes) & (np.round(value_no) == 1) ):
            RULE3 = RULE3 + 1
            print("RULE 3")


        if( (value_yes > value_no) & (value_yes < 0.9) ):
            print("RULE 4 or 5")
            RULE4 = RULE4 + 1
            #gnb.sideBySide(*[bn, inference, markov_blanket  ], captions=[ "Bayesian Network", "Inference", "Markov Blanket" ])

print( [RULE1, RULE2, RULE3, RULE4] )

"""#### FALSE NEGATIVES"""

print([len(true_positives), len(true_negatives), len(false_positives), len(false_negatives)])

RULE1 = 0
RULE2 = 0
RULE3 = 0
RULE4 = 0
RULE5 = 0

for instance in false_negatives:

    # get instance index
    indx = instance['index']
    print("INDEX = " + str(indx))
    
    [bn, inference, infoBN, markov_blanket] = generate_BN_explanationsMB(instance, label_lst, feature_names, 
                                                        class_var, encoder, scaler, model, PATH, DATASET_NAME, 
                                                        variance = VAR)
    ie=gum.LazyPropagation(bn)
    ie.makeInference()
    pos_class = ie.posterior(class_var)

    indx_yes = -1
    indx_no = -1

    if( len(markov_blanket.nodes()) == 1 ):
        RULE2 = RULE2+1    
        continue
    
    if (len(bn.variableFromName(class_var).labels()) == 1 ):

        if( bn.variableFromName(class_va).labels()[0] == "No" ):
            if( pos_class[0] >= 0.90 ):
                RULE1 = RULE1 + 1
                print("RULE 1")
        #if( bn.variableFromName().labels()[0] == "Yes" ):
        #    gnb.sideBySide(*[bn, inference, markov_blanket  ], captions=[ "Bayesian Network", "Inference", "Markov Blanket" ])

              
    if (len(bn.variableFromName(class_var).labels()) == 2 ):
        value_yes = pos_class[1]
        value_no = pos_class[0]

        if( value_no >= 0.90 ):
            RULE1 = RULE1 + 1
            print("RULE 1")

        if( (value_yes > value_no) & (np.round(value_yes) == 1) ):
            RULE3 = RULE3 + 1
            print("RULE 3")


        if( (value_no > value_yes) & (value_no < 0.9) ):
            print("RULE 4 or 5")
            RULE4 = RULE4 + 1
            #gnb.sideBySide(*[bn, inference, markov_blanket  ], captions=[ "Bayesian Network", "Inference", "Markov Blanket" ])

print( [RULE1, RULE2, RULE3, RULE4] )

