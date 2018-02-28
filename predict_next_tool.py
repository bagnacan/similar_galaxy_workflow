"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""
import sys
import numpy as np
import random
import collections
import time
import math
from random import shuffle

# machine learning library
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier


import prepare_data

class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.test_data_share = 0.2

    @classmethod
    def divide_train_test_data( self ):
        """
        Divide data into train and test sets in a random way
        """
        data = prepare_data.PrepareData()
        complete_data, labels, dictionary, reverse_dictionary = data.read_data()
        complete_data = complete_data[ :len( complete_data ) - 1 ]
        labels = labels[ :len( labels ) - 1 ]
        len_data = len( complete_data )
        len_test_data = int( self.test_data_share * len_data )
        dimensions = len( complete_data[ 0 ] )
        # take random positions from the complete data to create test data
        data_indices = range( len_data )
        shuffle( data_indices )
        test_positions = data_indices[ :len_test_data ]
        train_positions = data_indices[ len_test_data: ]

        # create test and train data and labels
        train_data = np.zeros( [ len_data - len_test_data, dimensions ] ) 
        train_labels = np.zeros( [ len_data - len_test_data, dimensions ] )
        test_data = np.zeros( [ len_test_data, dimensions ] )
        test_labels = np.zeros( [ len_test_data, dimensions ] )

        for i, item in enumerate( train_positions ):
            train_data[ i ] = complete_data[ item ]
            train_labels[ i ] = labels[ item ]

        for i, item in enumerate( test_positions ):
            test_data[ i ] = complete_data[ item ]
            test_labels[ i ] = labels[ item ]
        return train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary
        
    @classmethod
    def evaluate_keras_classifier( self ):
        """
        Create neural network classifier
        """
        print "Dividing data..."
        train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary = self.divide_train_test_data()
        train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
        train_labels = np.reshape(train_labels, (train_labels.shape[0], 1, train_labels.shape[1]))
        test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
        test_labels = np.reshape(test_labels, (test_labels.shape[0], 1, test_labels.shape[1]))
        train_data_shape = train_data.shape

        model = Sequential()
        model.add( Dense( 500, input_shape=( train_data_shape[ 1 ], train_data_shape[ 2 ] ), activation='relu' ) )
        model.add( Dropout( 0.3 ) )
        model.add( Dense( dimensions ) )
	model.add( Activation( 'softmax' ) )
        model.compile( loss='categorical_crossentropy', optimizer='rmsprop' )
        print "Start training..."
        model.fit( train_data, train_labels, epochs=50, batch_size=10 )
        print "Start predicting..."
        self.see_predicted_tools( model, dictionary, reverse_dictionary, test_data, test_labels, dimensions )

    @classmethod
    def see_predicted_tools( self, trained_model, dictionary, reverse_dictionary, test_data, test_labels, dimensions ):
        """
        Use trained model to predict next tool
        """
        # predict random input sequences
        num_predict = 50 #len( test_data )
        num_predictions = 5
        prediction_accuracy = 0
        print "Get top 5 predictions for each test input..."
        for i in range( num_predict ):
            input_seq = test_data[ i ][ 0 ]
            label = test_labels[ i ][ 0 ]
            tool_pos = np.where( input_seq > 0.0 )[ 0 ]
            input_tool_text = reverse_dictionary[ tool_pos[ 0 ] ]
            label_pos = np.where( label > 0.0 )[ 0 ]
            label_text = reverse_dictionary[ label_pos[ 0 ] ]
            print "Input sequence: %s " % input_tool_text
            print "Actual next tool: %s" % label_text
            input_seq_reshaped = np.reshape( input_seq, ( 1, 1, dimensions ) )
            # predict the next tool using the trained model
            prediction = trained_model.predict( input_seq_reshaped, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions ) )
            # take prediction in reverse order, best ones first
            prediction_pos = np.argsort( prediction, axis=0 )
            # get top n predictions
            top_prediction_pos = prediction_pos[ -num_predictions: ]
            # get tool names for the predicted positions
            top_predictions = [ reverse_dictionary[ pred_pos ] for pred_pos in top_prediction_pos ]
            top_predicted_tools_text = " ".join( top_predictions )
            if label_text in top_predictions:
                prediction_accuracy += 1
            print "Predicted next tools: %s" % top_predicted_tools_text
            print "=========================================="
        print "Prediction accuracy: %s" % str( float( prediction_accuracy ) / num_predict )


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    start_time = time.time()
    predict_tool = PredictNextTool()
    predict_tool.evaluate_keras_classifier()
    end_time = time.time()
    print "Program finished in %s seconds" % str( end_time - start_time  )
