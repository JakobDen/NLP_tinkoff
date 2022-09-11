import numpy as np
import argparse
import string
from keras.models import Model
from keras.layers import Dense, Concatenate, Input, Activation
from keras.layers import Dropout
from keras.layers import LSTM, TimeDistributed
from gensim.models.word2vec import Word2Vec
from keras.optimizers import Adam
import os

class text_generator():
    def __init__(self):
        
        args = self.data_loading()
        
        self.input_directory = args.input_dir
        self.output_directory = args.model
        
        self.max_sentence_len = 15
        self.batch_size = 16
        
        self.stop_symbols = string.punctuation + '…»«—'
        self.START_CHAR = '\b'
        self.END_CHAR = '\t'
        self.PADDING_CHAR = '\a'
        self.special_symbols = [self.START_CHAR, self.END_CHAR, self.PADDING_CHAR]
        
        self.Data = self.data_mining()
        self.w2v_model = Word2Vec(self.Data, vector_size=128, window=5, min_count=5, workers=4)

        test_indices = np.random.choice(range(len(self.Data)), int(len(self.Data) * 0.05))
        Data_test = [self.Data[x] for x in test_indices]
        X_test, y_test = self.get_matrices(Data_test)

        self.Data_train = [self.Data[x] for x in set(range(len(self.Data))) - set(test_indices)]
        self.Data_train = sorted(self.Data_train, key = lambda x : len(x))

        self.create_model()
        self.Fit()
        
        self.model.save(self.output_directory)
        self.w2v_model.save(os.join(self.output_directory, 'word2vec.model'))
        
    def data_loading(self):
        parser = argparse.ArgumentParser(description='training DNN model')
        parser.add_argument('--input_dir', type=str, help='Input dir for texts')
        parser.add_argument('--model', type=str, help='Output dir for model weights')
        args = parser.parse_args()
        return args

    def data_mining(self):
        Data = []
        files = [os.path.join(self.input_directory, f) for f in os.listdir(self.input_directory) \
                if os.path.isfile(os.path.join(self.input_directory, f))]
    
        for path in files:
            with open(path) as file_:
                docs = file_.readlines()
            sentences = [[word for word in doc.lower().translate({ord(x): ' ' for x in self.stop_symbols}).split()\
                        [ : self.max_sentence_len]] for doc in docs]
            Data += [x for x in sentences if x and len(x) > 3]
        Data.append(self.special_symbols * 6)
        return Data
    
    def get_matrices(self, Data):
            X = np.zeros((len(Data), self.max_sentence_len, 128), dtype=np.float32)
            y = np.zeros((len(Data), self.max_sentence_len, 128), dtype=np.float32)
            for i, sentence in enumerate(Data):
                word_seq = ([self.START_CHAR] + sentence + [self.END_CHAR])
                diff = self.max_sentence_len - len(word_seq) + 1
                while diff > 0:
                    word_seq += [self.PADDING_CHAR]
                    diff -= 1
                for t in range(self.max_sentence_len):
                    if(word_seq[t + 1] in self.w2v_model.wv) and (word_seq[t] in self.w2v_model.wv):
                        X[i, t, :] = self.w2v_model.wv[word_seq[t]]
                        y[i, t, :] = self.w2v_model.wv[word_seq[t + 1]]
            return X, y
       
    def Fit(self):
        self.model.fit_generator(
            self.generate_batch(),
            int(len(self.Data_train) / self.batch_size) * self.batch_size, epochs=5)
    
    def generate_batch(self):
            while True:
                for i in range( int(len(self.Data_train) / self.batch_size) ):
                    Data_batch = self.Data_train[i * self.batch_size : (i + 1) * self.batch_size]
                    yield self.get_matrices(Data_batch)
    
    def create_model(self):
        vec = Input(shape=(self.max_sentence_len, 128))
        l1 = LSTM(128, activation='tanh', return_sequences=True)(vec)
        l1_d = Dropout(0.2)(l1)
        input2 = Concatenate()([vec, l1_d])
        l2 = LSTM(128, activation='tanh', return_sequences=True)(input2)
        l2_d = Dropout(0.2)(l2)
        input3 = Concatenate()([vec, l2_d])
        l3 = LSTM(128, activation='tanh', return_sequences=True)(input3)
        l3_d = Dropout(0.2)(l3)
        input_d = Concatenate()([l1_d, l2, l3_d])
        dense3 = TimeDistributed(Dense(128))(input_d)
        output_res = Activation('sigmoid')(dense3)
        self.model = Model(inputs=vec, outputs=output_res)
        self.model.compile(loss='mse',optimizer=Adam(clipnorm = 1), metrics=['accuracy'])

tg = text_generator()
