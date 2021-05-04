import zipfile
import os
import wget
import numpy as np
import os.path
from os import path

url = 'danielwillgeorge/glove6b100dtxt'

def download_GloveDataset(url):
    import zipfile
    import os
    import wget

    if not os.path.isdir ( 'glove_data' ):
        os.mkdir ( 'glove_data' )
    else:
        pass

    for afile in os.listdir ( os.path.join ( str ( os.getcwd () ), 'glove_data' ) ):
        if afile.split ( '.' )[0] == 'glove':
            pass
        else:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('danielwillgeorge/glove6b100dtxt', path= str(os.getcwd())+"/glove_data/", force = True, unzip=True)

        print ( 'File already exists!')

    glove_path = os.path.join(os.getcwd(), 'glove_data', str(os.listdir('glove_data')[0]))
    return glove_path
    
# Embedding the indexed data

def embedding_GloveDataset(glove_path, word_index):

    embedding_index = {}
    while len(embedding_index) == 0:
        with open(glove_path, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs

        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, 100))

        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    print ('Glove data has been embedded.')
    return embedding_matrix, embedding_index
