import tensorflow as tf
import numpy as np
import random
from importlib import reload 
import utils
reload(utils)
from utils import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Function to get word2vec representations
#
# Arguments:
# reviews: A list of strings, each string represents a review
#
# Returns: mat (numpy.ndarray) of size (len(reviews), dim)
# mat is a two-dimensional numpy array containing vector representation for ith review (in input list reviews) in ith row
# dim represents the dimensions of word vectors, here dim = 300 for Google News pre-trained vectors
def w2v_rep(reviews):
    dim = 300
    mat = np.zeros((len(reviews), dim))
    w2v_dict = load_w2v()
    reviewIndex = 0
    for review in reviews:
            tokens = get_tokens(review)
            reviewWordCount = 0
            for token in tokens:
                if token in w2v_dict.keys():
                    mat[reviewIndex]=np.add(w2v_dict[token],mat[reviewIndex])
                    reviewWordCount+=1
            if reviewWordCount>0:
                mat[reviewIndex] = mat[reviewIndex]/np.array([reviewWordCount])
            reviewIndex+=1
    return mat
    


# Function to build a feed-forward neural network using tf.keras.Sequential model. building the sequential model
# by stacking up dense layers such that each hidden layer has 'relu' activation. Add an output dense layer in the end
# containing 1 unit, with 'sigmoid' activation, this is to ensure that we get label probability as output
#
# Arguments:
# params (dict): A dictionary containing the following parameter data:
#                    layers (int): Number of dense layers in the neural network
#                    units (int): Number of units in each dense layer
#                    loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#                    optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#
# Returns:
# model (tf.keras.Sequential), a compiled model created using the specified parameters
def build_nn(params):
    
    model = tf.keras.Sequential()
    for layer in range(params['layers']-1):
        # Adds a densely-connected layer with specified units to the model:
        model.add(tf.keras.layers.Dense(params['units'], activation='relu'))
    
    # Add a sigmoid layer with 1 output units:
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=params['optimizer'],
              loss=params['loss'],  
              metrics=['accuracy'])
    
    return model


# Function to select the best parameter combination based on accuracy by evaluating all parameter combinations
# This function should train on the training set (X_train, y_train) and evluate using the validation set (X_val, y_val)
#
# Arguments:
# params (dict): A dictionary containing parameter combinations to try:
#                    layers (list of int): Each element specifies number of dense layers in the neural network
#                    units (list of int): Each element specifies the number of units in each dense layer
#                    loss (list of string): Each element specifies the type of loss to optimize ('binary_crossentropy' or 'mse)
#                    optimizer (list of string): Each element specifies the type of optimizer to use while training ('sgd' or 'adam')
#                    epochs (list of int): Each element specifies the number of iterations over the training set
# X_train (numpy.ndarray): A matrix containing w2v representations for training set of shape (len(reviews), dim)
# y_train (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_train of shape (X_train.shape[0], )
# X_val (numpy.ndarray): A matrix containing w2v representations for validation set of shape (len(reviews), dim)
# y_val (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_val of shape (X_val.shape[0], )
#
# Returns:
# best_params (dict): A dictionary containing the best parameter combination:
#                        layers (int): Number of dense layers in the neural network
#                          units (int): Number of units in each dense layer
#                         loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#                        optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#                        epochs (int): Number of iterations over the training set
def find_best_params(params, X_train, y_train, X_val, y_val,test=0):
    
    maxAccuracy = 0.0
    if test != 1:
        for layer in params['layers']:
            for unit in params['units']:
                for lossValue in params['loss']:
                    for optimizerValue in params['optimizer']:
                        for epoch in params['epochs']:
                            param = {
                                'layers': layer,
                                'units': unit,
                                'loss': lossValue,
                                'optimizer':optimizerValue ,
                                'epochs': epoch
                                }
                            model = build_nn(param)
                            result = model.fit(X_train, y_train, epochs=epoch,validation_data=(X_val, y_val))
                            accuracy = result.history['val_accuracy'][-1]
                                

                            if maxAccuracy < accuracy:
                                print('-----setting max Value-----------')
                                best_params=param
                                maxAccuracy = accuracy 
    else :
        best_params = {'layers': 3, 'units': 16, 'loss': 'binary_crossentropy', 'optimizer': 'adam', 'epochs': 10}
    return best_params


# Function to convert probabilities into pos/neg labels
#
# Arguments:
# probs (numpy.ndarray): A numpy vector containing probability of being positive
#
# Returns:
# pred (numpy.ndarray): A numpy vector containing pos/neg labels such that ith value in probs is mapped to ith value in pred
#                         A value is mapped to pos label if it is >=0.5, neg otherwise
def translate_probs(probs,isLabel=1):
    if isLabel!=1:
        pred = np.array([1 if r >= 0.5 else 0 for r in probs])
    else:
        pred = np.array(['pos' if r >= 0.5 else 'neg' for r in probs])
    return pred

def get_review_list(data):
    return [r['text'] for r in data]

def get_label_list(data):
    return [r['label'] for r in data]
    
def convert_tags_to_number(data):
    return  np.array([1 if r['label'] == 'pos' else 0 for r in data])

def main():
    # Load dataset
    data = load_data('movie_reviews.csv')

    # Extract list of reviews from the training set
    # Note that since data is already sorted by review IDs, you do not need to sort it again for a subset
    train_data = list(filter(lambda x: x['split'] == 'train', data))
    val_data = list(filter(lambda x: x['split'] == 'val', data))
    test_data = list(filter(lambda x: x['split'] == 'test', data))
    
    reviews_train = get_review_list(train_data)
    reviews_test = get_review_list(test_data)
    reviews_val = get_review_list(val_data)

    # Compute the word2vec representation for training set
    X_train = w2v_rep(reviews_train)

    # Extracting representations for validation (X_val) and test (X_test) set
    # Also extract labels for training (y_train) and validation (y_val)
    # Used 1 to represent 'pos' label and 0 to represent 'neg' label
    X_val = w2v_rep(reviews_val)
    y_val = convert_tags_to_number(val_data)
    
    X_test = w2v_rep(reviews_test)
    y_train = convert_tags_to_number(train_data)

    # Possible parameters list
    params = {
        'layers': [1, 3],
        'units': [8, 16, 32],
        'loss': ['binary_crossentropy', 'mse'],
        'optimizer': ['sgd', 'adam'],
        'epochs': [1, 5, 10]
    }
    reset_seeds()
    best_params = find_best_params(params, X_train, y_train, X_val, y_val)

    # Build a model with best parameters and fit on the training set (uncomment the 2 lines below)
    model = build_nn(best_params)
    model.fit(X_train, y_train, epochs=best_params['epochs'])

    # Use the model to predict labels for the validation set (uncomment the line below)
    pred_val = model.predict(X_val)
    pred_val_label = translate_probs(pred_val,0)
    val_label = get_label_list(val_data)
    
    print(precision_score(y_val, pred_val_label,average=None))
    print(recall_score(y_val, pred_val_label,average=None))
    print(accuracy_score(y_val, pred_val_label))
    print(f1_score(y_val, pred_val_label,average=None))

    # Using the model to predict labels for the test set (uncomment the line below)
    pred = model.predict(X_test) 
    
    # Translate predicted probabilities into pos/neg labels
    pred = translate_probs(pred)


if __name__ == '__main__':
    main()
