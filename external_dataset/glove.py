import zipfile
import os
import wget
import numpy as np
import os.path
from os import path

url = "https://storage.googleapis.com/kaggle-data-sets/715814/1246668/compressed/glove.6B.100d.txt.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210425%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210425T151647Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=340005cb614f61b7f64c0d53500e0074044c397f712856fe8a86ae0bdc7aaad7523fd08feb95345e2daaef9f699740d78459711451ae47685ca622b4ebce560f6affb3b3f441a10bac612d6f0878350179a0496d75b6b9a15293f11d275868cd90351ed3e4e8ca210db7e2ed14ae891a8ed3af37afc149602f33d801f54b577113ce92efda6cf4ab63912159827a20466bc175d11aa1a117403fddaafdcc26bf83f2545d0414a0358c88c07acf936ffe8e5f8a8395cd504e3f7cfa108f2bfd06fc54fcab2417915707ed64e6697dffa02b9b6a95505290e1c7313c2371d8be3fa0bbaeb077e0af123a9fc6634e38f9c82e772ae35499f758fe6953cd9e92d9c0"


def download_GloveDataset(url):
    import zipfile
    import os
    import wget

    if not os.path.isdir('glove_data'):
        os.mkdir('glove_data')
    else:
        pass
    
    print('Directory created. Now downloading the data. This may take a minute or two.')
    
    if os.path.isfile('glove_data/glove.6B.100d.txt'):
        print('Glove_data exists! Now embedding the dataset...')
    else:
        wget.download(url, os.path.join(str(os.getcwd()), 'glove_data/'))
        downloaded_file = os.path.join(os.getcwd(), 'glove_data', str(os.listdir('glove_data')[0]))
        with zipfile.ZipFile(downloaded_file) as zip_file:
            zip_file.extractall(os.path.join(os.getcwd(), 'glove_data/'))
        print('Glove data is now downloaded and extracted.')
    
        os.remove(downloaded_file)

    glove_path = os.path.join(os.getcwd(), 'glove_data', str(os.listdir('glove_data')[0]))
    return glove_path
    
# Embedding the indexed data

def embedding_GloveDataset(filepath, clean_word_index):

    if not os.path.isfile('glove_data/glove.6B.100d.txt'):
        print('Glove dataset needs to be download and unzipped. Re-run the download_GloveDataset function.')

    else:
        embedding_index = {}
        with open(filepath, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs

        vocab_size = len(clean_word_index) + 1
        embedding_matrix = np.zeros((vocab_size, 100))

        for word, i in clean_word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    print ('Glove data has been embedded.')
    return embedding_index
