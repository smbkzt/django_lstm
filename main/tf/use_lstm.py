import os

import numpy as np
import tensorflow as tf

from . import config
from .train_and_test import PrepareData
from django_lstm.settings import BASE_DIR


class TryLstm():
    def __init__(self):
        print("Creating LSTM instance")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.maxSeqLength = config.maxSeqLength
        self.batchSize = config.batchSize
        self.load_gloves()
        self.restore_models()

    def load_gloves(self):
        """Loads GloVes model (wordsList)"""
        print("Loading gloves model...")
        filepath = BASE_DIR + '/main/tf/data/wordsList.npy'
        self.wordsList = np.load(filepath).tolist()
        self.wordsList = [word for word in self.wordsList]

    def restore_models(self):
        """Restores the TF session"""
        tf.reset_default_graph()
        self.sess = tf.Session()

        # Restoring the meta and latest model
        filepath = BASE_DIR + "/main/tf/models/"
        path = ".".join([tf.train.latest_checkpoint(filepath), "meta"])
        saver = tf.train.import_meta_graph(path)
        saver.restore(self.sess, tf.train.latest_checkpoint(filepath))

        # Restoring the tf variables
        self.input_data = tf.get_collection("input_data")[0]
        self.prediction = tf.get_collection("prediction")[0]

    def predict(self, inputText):
        inputMatrix = self.getSentenceMatrix(inputText)
        predictedSentiment = self.sess.run(self.prediction,
                                           {self.input_data: inputMatrix}
                                           )[0]
        print(f"Agreement coefficient:",
              "{0:.2f}".format(predictedSentiment[0]))
        print(f"Disagreement coefficient:",
              "{0:.2f}".format(predictedSentiment[1]))
        if (predictedSentiment[0] > predictedSentiment[1]):
            answer = "The comment message has agreement sentiment"
        else:
            answer = "The comment message has disagreement sentiment"
        return answer

    def getSentenceMatrix(self, sentence):
        sentenceMatrix = np.zeros([self.batchSize, self.maxSeqLength],
                                  dtype='int32')
        cleanedSentence = PrepareData.clean_string(sentence)
        split = cleanedSentence.split()
        for indexCounter, word in enumerate(split):
            try:
                sentenceMatrix[0, indexCounter] = self.wordsList.index(word)
            except ValueError:
                # Vector for unknown words
                sentenceMatrix[0, indexCounter] = 000
        return sentenceMatrix
