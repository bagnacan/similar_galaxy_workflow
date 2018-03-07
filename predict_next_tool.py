"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""
import sys
import numpy as np
import random
import collections
import time
import math
import os
import h5py as h5
from random import shuffle

# machine learning library
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import gensim
from gensim.models import Word2Vec

import prepare_data
import evaluate_top_results

class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.sequence_file = self.current_working_dir + "/data/train_data_sequence.txt"
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.weights_path = self.current_working_dir + "/data/weights/trained_model.h5"
        self.loss_path = self.current_working_dir + "/data/loss_history.txt"
        self.accuracy_path = self.current_working_dir + "/data/accuracy_history.txt"
        self.val_loss_path = self.current_working_dir + "/data/val_loss_history.txt"
        self.val_accuracy_path = self.current_working_dir + "/data/val_accuracy_history.txt"
        self.epoch_weights_path = self.current_working_dir + "/data/weights/weights-epoch-{epoch:02d}.hdf5"
        self.test_data_path = self.current_working_dir + "/data/test_data.hdf5"
        self.test_labels_path = self.current_working_dir + "/data/test_labels.hdf5"
        self.doc2vec_model_path = self.current_working_dir + "/data/doc2vec_model.hdf5"

    @classmethod
    def divide_train_test_data( self ):
        """
        Divide data into train and test sets in a random way
        """
        test_data_share = 0.33
        seed = 0
        node_vectors = dict()
        
        data = prepare_data.PrepareData()
        complete_data, labels, dictionary, reverse_dictionary, graph_documents, nodes_len = data.read_data()
        
        node_dimensions = 100
        model = Word2Vec( graph_documents, min_count=1, size=node_dimensions, iter=20 )
        nodes = list( model.wv.vocab )
        nodes_vec = model[ model.wv.vocab ]
        complete_input_vector = np.zeros( [ complete_data.shape[ 0 ], complete_data.shape[ 1 ], node_dimensions ] )
        for i in range( len( nodes ) ):
            node_vectors[ nodes[ i ] ] = nodes_vec[ i ]
        for index, path in enumerate( complete_data ):
             for node_idx, node_pos in enumerate( path ):
                 if node_pos > 0:
                     node_name = reverse_dictionary[ node_pos ]
                     complete_input_vector[ index ][ node_idx ][ : ] = node_vectors[ node_name ]
        print (complete_input_vector.shape)
        model.save('word2vec_model.bin')
        np.random.seed( seed )
        dimensions = len( dictionary )
        train_data, test_data, train_labels, test_labels = train_test_split( complete_input_vector, labels, test_size=test_data_share, random_state=seed )
        print(train_data.shape)
        print(train_labels.shape)
        # write the test data and labels to files for further evaluation
        with h5.File( self.test_data_path, "w" ) as test_data_file:
            test_data_file.create_dataset( "testdata", test_data.shape, data=test_data )
        with h5.File( self.test_labels_path, "w" ) as test_labels_file:
            test_labels_file.create_dataset( "testlabels", test_labels.shape, data=test_labels )
        return train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary, nodes_len

    @classmethod
    def evaluate_LSTM_network( self ):
        """
        Create LSTM network and evaluate performance
        """
        print ("Dividing data...")
        n_epochs = 200
        num_predictions = 5
        batch_size = 40
        dropout = 0.2
        node_dimensions = 100
        train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary, nodes_len = self.divide_train_test_data()
        # reshape train and test data
        train_data = np.reshape( train_data, ( train_data.shape[0], train_data.shape[1], train_data.shape[2] ) )
        train_labels = np.reshape( train_labels, (train_labels.shape[0], 1, train_labels.shape[1] ) )
        test_data = np.reshape(test_data, ( test_data.shape[0], test_data.shape[1], test_data.shape[2] ) )
        test_labels = np.reshape( test_labels, ( test_labels.shape[0], 1, test_labels.shape[1] ) )
        train_data_shape = train_data.shape
        print (train_data_shape)
        optimizer = Adam( lr=0.0001 )
        # define recurrent network
        model = Sequential()
        model.add( LSTM( 256, input_shape=( train_data_shape[ 1 ], train_data_shape[ 2 ] ), return_sequences=True, recurrent_dropout=dropout ) )
        #model.add(Embedding( train_data_shape[1], node_dimensions, input_length=train_data_shape[1] ) )
        #model.add( LSTM( 256, return_sequences=True) )
        model.add( Dropout( dropout ) )
        #model.add( LSTM( 512, return_sequences=True, recurrent_dropout=dropout ) )
        #model.add( Dropout( dropout ) )
        #model.add( LSTM( 256, recurrent_dropout=dropout ) )
        #model.add( Dense( 256 ) )
        #model.add( Dropout( dropout ) )
        model.add( Dense( dimensions ) )
        model.add( Activation( 'softmax' ) )
        model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=[ 'accuracy' ] )

        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( self.epoch_weights_path, verbose=1, mode='max' )
        callbacks_list = [ checkpoint ]

        print ("Start training...")
        model_fit_callbacks = model.fit( train_data, train_labels, validation_data=( test_data, test_labels ), epochs=n_epochs, batch_size=batch_size, callbacks=callbacks_list, shuffle=True )
        loss_values = model_fit_callbacks.history[ "loss" ]
        accuracy_values = model_fit_callbacks.history[ "acc" ]
        validation_loss = model_fit_callbacks.history[ "val_loss" ]
        validation_acc = model_fit_callbacks.history[ "val_acc" ]

        np.savetxt( self.loss_path, np.array( loss_values ), delimiter="," )
        np.savetxt( self.accuracy_path, np.array( accuracy_values ), delimiter="," )
        np.savetxt( self.val_loss_path, np.array( validation_loss ), delimiter="," )
        np.savetxt( self.val_accuracy_path, np.array( validation_acc ), delimiter="," )

        # save the network as json
        model_json = model.to_json()
        with open( self.network_config_json_path, "w" ) as json_file:
            json_file.write(model_json)
        # save the learned weights to h5 file
        model.save_weights( self.weights_path )
        print ("Training finished")

    @classmethod
    def load_saved_model( self, network_config_path, weights_path ):
        """
        Load the saved trained model using the saved network and its weights
        """
        with open( network_config_path, 'r' ) as network_config_file:
            loaded_model = network_config_file.read()
        # load the network
        loaded_model = model_from_json(loaded_model)
        # load the saved weights into the model
        loaded_model.load_weights( weights_path )
        return loaded_model

    @classmethod
    def get_raw_paths( self ):
        """
        Read training data and its labels files
        """
        training_samples = list()
        training_labels = list()
        train_file = open( self.sequence_file, "r" )
        train_file = train_file.read().split( "\n" )
        for item in train_file:
            tools = item.split( "," )
            train_tools = tools[ :len( tools) - 1 ]
            train_tools = ",".join( train_tools )
            training_samples.append( train_tools )
            training_labels.append( tools[ -1 ] )
        return training_samples, training_labels
 
    @classmethod
    def see_predicted_tools( self, trained_model, test_data, dictionary, reverse_dictionary, dimensions ):
        """
        Use trained model to predict next tool
        """
        # predict random input sequences
        num_predict = len( test_data )
        num_predictions = 5
        train_data, train_labels = self.get_raw_paths()
        prediction_accuracy = self.get_top_predictions( num_predictions, test_data, train_labels, dimensions, trained_model, reverse_dictionary )
        print ("No. total test inputs: %d" % num_predict)
        print ("No. correctly predicted: %d" % prediction_accuracy)
        print ("Prediction accuracy: %s" % str( float( prediction_accuracy ) / num_predict ))
  

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    start_time = time.time()
    predict_tool = PredictNextTool()
    predict_tool.evaluate_LSTM_network()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
