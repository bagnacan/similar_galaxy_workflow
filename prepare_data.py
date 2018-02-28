"""
Predict nodes in graphichal data (Galaxy workflows) using Recurrent Neural Network (LSTM)
"""
import sys
import random
import collections
import time
import numpy as np


class PrepareData:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.raw_file = "data/workflow_steps.txt"
        self.train_file = "data/train_data.txt"

    @classmethod
    def process_processed_data( self, fname ):
        """
        Get all the tools and complete set of individual paths for each workflow
        """
        tokens = list()
        raw_paths = list()
        with open( fname ) as f:
            data = f.readlines()
        raw_paths = [ x.replace( "\n", '' ) for x in data ]
        for item in raw_paths:
            split_items = item.split( " " )
            for token in split_items:
                if token not in tokens:
                    tokens.append( token )
        tokens = np.array( tokens ) 
        tokens = np.reshape( tokens, [ -1, ] )
        return tokens, raw_paths

    @classmethod
    def create_data_dictionary( self, words ):
        """
        Create two dictionaries having tools names and their indexes
        """
        count = collections.Counter( words ).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len( dictionary )  
        reverse_dictionary = dict(zip( dictionary.values(), dictionary.keys() ) )
        return dictionary, reverse_dictionary
    
    @classmethod
    def create_train_labels_file( self, dictionary, raw_paths ):
        """
        Create training data with its labels with varying window sizes
        """
        len_dict = len( dictionary )
        train_data = list()
        train_data_sequence = list()
        print "preparing downstream data..."
        # prepare training data using a pair of tools - one trainig and the next one as its label
        for index, item in enumerate( raw_paths ):
            tools = item.split(" ")
            for window in range( 0, len( tools ) - 1 ):
                training_tool = tools[ window ]
                label_tool = tools[ window + 1 ]
                tools_pair = str( dictionary[ training_tool ] )  + "," + str( dictionary[ label_tool ] )
                if tools_pair not in train_data:
                    train_data.append( tools_pair )
                    train_data_sequence.append( training_tool + "," + label_tool )
            print "Path %d processed" % ( index + 1 )

        with open( self.train_file, "w" ) as train_file:
            for item in train_data:
                train_file.write( "%s\n" % item )

        with open( "data/train_data_sequence.txt", "w" ) as train_seq:
            for item in train_data_sequence:
                train_seq.write( "%s\n" % item )

        print "Training data and labels files written"

    @classmethod
    def prepare_train_test_data( self ):
        """
        Read training data and its labels files
        """
        training_data = list()
        train_file = open( self.train_file, "r" )
        train_file = train_file.read().split( "\n" )
        for item in train_file:
            training_data.append( item )
        return training_data

    @classmethod
    def read_data( self ):
        """
        Convert the data into corresponding arrays
        """
        processed_data, raw_paths = self.process_processed_data( self.raw_file )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        #self.create_train_labels_file( dictionary, raw_paths )
        # all the nodes/tools are classes as well
        num_classes = len( dictionary )
        train_data_array = np.zeros([num_classes])
        train_data = self.prepare_train_test_data()
        # initialize the training data matrix
        train_data_array = np.zeros( [ len( train_data ), num_classes ] )
        train_label_array = np.zeros( [ len( train_data ), num_classes ] )
        for index, item in enumerate( train_data ):
           if item:
               positions = item.split( "," )
               train_data_array[ index ][ int( positions[ 0 ] ) ] = 1.0
               train_label_array[ index ][ int( positions[ 1 ] ) ] = 1.0
        return train_data_array, train_label_array, dictionary, reverse_dictionary
