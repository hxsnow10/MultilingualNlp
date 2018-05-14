# coding: utf-8

# # So, show me how to align two vector spaces for myself!

# No problem. We're going to run through the example given in the README again, and show you how to learn your own transformation to align the French vector space to the Russian vector space.
# 
# First, let's define a few simple functions...

# In[1]:


import numpy as np
from fasttext import FastVector

# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)


# Now we load the French and Russian word vectors, and evaluate the similarity of "chat" and "кот":

# In[2]:


fr_dictionary = FastVector(vector_file='zh_vec.txt')
ru_dictionary = FastVector(vector_file='en_vec.txt')

fr_vector = fr_dictionary["chat"]
ru_vector = ru_dictionary["кот"]
print(FastVector.cosine_similarity(fr_vector, ru_vector))


# "chat" and "кот" both mean "cat", so they should be highly similar; clearly the two word vector spaces are not yet aligned. To align them, we need a bilingual dictionary of French and Russian translation pairs. As it happens, this is a great opportunity to show you something truly amazing...
# 
# Many words appear in the vocabularies of more than one language; words like "alberto", "london" and "presse". These words usually mean similar things in each language. Therefore we can form a bilingual dictionary, by simply extracting every word that appears in both the French and Russian vocabularies.

# In[3]:


ru_words = set(ru_dictionary.word2id.keys())
fr_words = set(fr_dictionary.word2id.keys())
overlap = list(ru_words & fr_words)
bilingual_dictionary = [(entry, entry) for entry in overlap]


# Let's align the French vectors to the Russian vectors, using only this "free" dictionary that we acquired without any bilingual expert knowledge.

# In[ ]:


# form the training matrices
source_matrix, target_matrix = make_training_matrices(
    fr_dictionary, ru_dictionary, bilingual_dictionary)

# learn and apply the transformation
transform = learn_transformation(source_matrix, target_matrix)
fr_dictionary.apply_transform(transform)


# Finally, we re-evaluate the similarity of "chat" and "кот":

# In[4]:


fr_vector = fr_dictionary["chat"]
ru_vector = ru_dictionary["кот"]
print(FastVector.cosine_similarity(fr_vector, ru_vector))


# "chat" and "кот" are pretty similar after all :)
# 
# Use this simple "identical strings" trick to align other language pairs for yourself, or prepare your own expert bilingual dictionaries for optimal performance.
