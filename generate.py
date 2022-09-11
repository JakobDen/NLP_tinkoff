import argparse
import random
import pickle
import numpy as np
from keras.models import load_model
from gensim.models import Word2Vec

length = 10

w2v_model = Word2Vec.load("data/word2vec.model")
model = load_model('data/tgen')
END_CHAR = '\t'
max_sentence_len = 15
with open('data/X_test.pickle', 'rb') as f:
    X_test = pickle.load(f)
with open('data/sentences_test.pickle', 'rb') as f:
    sentences_test = pickle.load(f)

def sample_one(length):
    rind = random.randint(0, len(X_test) - 1)
    result = ' '.join(sentences_test[rind])
    Xsampled = X_test[rind]
    ysampled = model.predict(np.reshape(Xsampled, (-1, max_sentence_len, 128)), verbose=0)
    selected_word = w2v_model.wv.most_similar(ysampled[0][-1])[random.randint(0, 2)][0]
    
    i = 0
    while i < length:
        result += ' ' + selected_word
        Xsampled = np.append(Xsampled, w2v_model.wv[selected_word])
        Xsampled = Xsampled[128:]
        ysampled = model.predict(np.reshape(Xsampled, (-1, max_sentence_len, 128)), verbose=0)
        selected_word = w2v_model.wv.most_similar(ysampled[0][-1])[random.randint(0, 2)][0]
        if selected_word == END_CHAR:
            break
        i += 1
    return result

print(sample_one(length))
