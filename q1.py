# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2019
# Assignment 5
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import tensorflow as tf
import numpy as np
import random
from utils import *



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
	# [YOUR CODE HERE]
	return mat


# Function to build a feed-forward neural network using tf.keras.Sequential model. You should build the sequential model
# by stacking up dense layers such that each hidden layer has 'relu' activation. Add an output dense layer in the end
# containing 1 unit, with 'sigmoid' activation, this is to ensure that we get label probability as output
#
# Arguments:
# params (dict): A dictionary containing the following parameter data:
#					layers (int): Number of dense layers in the neural network
#					units (int): Number of units in each dense layer
#					loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#					optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#
# Returns:
# model (tf.keras.Sequential), a compiled model created using the specified parameters
def build_nn(params):
	model = tf.keras.Sequential()
	# [YOUR CODE HERE]
	return model


# Function to select the best parameter combination based on accuracy by evaluating all parameter combinations
# This function should train on the training set (X_train, y_train) and evluate using the validation set (X_val, y_val)
#
# Arguments:
# params (dict): A dictionary containing parameter combinations to try:
#					layers (list of int): Each element specifies number of dense layers in the neural network
#					units (list of int): Each element specifies the number of units in each dense layer
#					loss (list of string): Each element specifies the type of loss to optimize ('binary_crossentropy' or 'mse)
#					optimizer (list of string): Each element specifies the type of optimizer to use while training ('sgd' or 'adam')
#					epochs (list of int): Each element specifies the number of iterations over the training set
# X_train (numpy.ndarray): A matrix containing w2v representations for training set of shape (len(reviews), dim)
# y_train (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_train of shape (X_train.shape[0], )
# X_val (numpy.ndarray): A matrix containing w2v representations for validation set of shape (len(reviews), dim)
# y_val (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_val of shape (X_val.shape[0], )
#
# Returns:
# best_params (dict): A dictionary containing the best parameter combination:
#	    				layers (int): Number of dense layers in the neural network
#	 	     			units (int): Number of units in each dense layer
#	 					loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#						optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#						epochs (int): Number of iterations over the training set
def find_best_params(params, X_train, y_train, X_val, y_val):
	# [YOUR CODE HERE]
	best_params = {
		'layers': 1,
		'units': 8,
		'loss': 'binary_crossentropy',
		'optimizer': 'adam',
		'epochs': 1
	}
	return best_params


# Function to convert probabilities into pos/neg labels
#
# Arguments:
# probs (numpy.ndarray): A numpy vector containing probability of being positive
#
# Returns:
# pred (numpy.ndarray): A numpy vector containing pos/neg labels such that ith value in probs is mapped to ith value in pred
# 						A value is mapped to pos label if it is >=0.5, neg otherwise
def translate_probs(probs):
	# [YOUR CODE HERE]
	pred = np.repeat('pos', probs.shape[0])
	return pred


# Use the main function to test your code when running it from a terminal
# Sample code is provided to assist with the assignment, it is recommended
# that you do not change the code in main function for this assignment
# You can run the code from termianl as: python3 q3.py
# It should produce the following output and 2 files (q1-train-rep.npy, q1-pred.npy):
#
# $ python3 q1.py
# Best parameters: {'layers': 1, 'units': 8, 'loss': 'binary_crossentropy', 'optimizer': 'adam', 'epochs': 1}

def main():
	# Load dataset
	data = load_data('movie_reviews.csv')

	# Extract list of reviews from the training set
	# Note that since data is already sorted by review IDs, you do not need to sort it again for a subset
	train_data = list(filter(lambda x: x['split'] == 'train', data))
	reviews_train = [r['text'] for r in train_data]

	# Compute the word2vec representation for training set
	X_train = w2v_rep(reviews_train)
	# Save these representations in q1-train-rep.npy for submission
	np.save('q1-train-rep.npy', X_train)

	# Write your code here to extract representations for validation (X_val) and test (X_test) set
	# Also extract labels for training (y_train) and validation (y_val)
	# Use 1 to represent 'pos' label and 0 to represent 'neg' label
	# You may look at q2.ipynb for help
	# [YOUR CODE HERE]
	X_val = None
	X_test = None
	y_train = None
	y_val = None


	# Build a feed forward neural network model with build_nn function
	params = {
		'layers': 1,
		'units': 8,
		'loss': 'binary_crossentropy',
		'optimizer': 'adam'
	}
	model = build_nn(params)

	# Function to choose best parameters
	# You should use build_nn function in find_best_params function
	params = {
		'layers': [1, 3],
		'units': [8, 16, 32],
		'loss': ['binary_crossentropy', 'mse'],
		'optimizer': ['sgd', 'adam'],
		'epochs': [1, 5, 10]
	}
	# reset_seeds function must be called immediately before find_best_params function
	reset_seeds()
	best_params = find_best_params(params, X_train, y_train, X_val, y_val)

	# Save the best parameters in q1-params.csv for submission
	print("Best parameters: {0}".format(best_params))

	# Build a model with best parameters and fit on the training set
	# reset_seeds function must be called immediately before build_nn and model.fit function
	reset_seeds()
	# Uncomment the following 2 lines to call the necessary functions
	# model = build_nn(best_params)
	# model.fit(X_train, y_train, epochs=best_params['epochs'])

	# Use the model to predict labels for the validation set (uncomment the line below)
	# pred = model.predict(X_val)
	
	# Write code here to evaluate model performance on the validation set
	# You should compute precision, recall, f1, accuracy
	# Save these results in q1-res.csv for submission
	# Can you use translate_probs function to facilitate the conversions before comparison?
	# [YOUR CODE HERE]

	# Just dummy data to avoid errors
	pred = np.zeros((10))
	# Use the model to predict labels for the test set (uncomment the line below)
	# pred = model.predict(X_test) 
	
	# Translate predicted probabilities into pos/neg labels
	pred = translate_probs(pred)
	# Save the results for submission
	np.save('q1-pred.npy', pred)


if __name__ == '__main__':
	main()
