"""
LING 380/780: Neural Network Models of Linguistic Structure
Assignment 0: Working with Python and Word Vectors
Due January 23, 2020

Your Name: Philos Kim
Your NetID: pk497
"""
from typing import Tuple, List

import numpy as np

def load_embeddings(filename: str) -> Tuple[List[str], np.ndarray]:
    """
    Loads word embeddings from a file.

    :param filename: The filename 
    :return: The words and their vectors
    """
    words = []
    vecs = []
    with open(filename, "r") as f:
        for line in f:
            word, vec = line.split(" ", 1)
            words.append(word)
            vecs.append(np.fromstring(vec, sep=" "))
    return words, np.array(vecs)

def top_k(w: str, k: int, words: List[str], vecs: np.ndarray) -> List[str]:
    """
    Question 3: Find the k words most similar to w according to cosine
    similarity, not including w itself.

    :param w: The target word
    :param k: The number of words to return
    :param words: All words in the vocabulary
    :param vecs: The matrix of word vectors
    :return: The k most similar words to w, not including w
    """
    vecs = np.array([v / np.linalg.norm(v) for v in vecs])
    w_vec = vecs[words.index(w)]

    arr = np.dot(vecs, w_vec)

    top_k = [words[i] for i in np.argsort(arr)[-k-1:] if words[i] != w]

    return np.flip(top_k)


def sim(l: List[str], words: List[str], vecs: np.ndarray) -> str:
    """
    Question 5: Find the word most similar to a collection of words.

    :param l: A collection of words
    :param words: All words in the vocabulary
    :param vecs: The matrix of word vectors
    :return: The word that is the most similar to the words in l
    """
    l_index = [words.index(word) for word in l]
    new_words = np.delete(words, l_index)

    l_sum = np.zeros(np.size(vecs[0]))
    for i in l_index:
        l_sum += vecs[i]
    l_sum = l_sum / np.linalg.norm(l_sum)

    vecs = np.array([v / np.linalg.norm(v) for v in vecs])

    arr = np.dot(vecs, l_sum)
    new_arr = np.delete(arr, l_index)

    return new_words[np.argmax(new_arr)]


if __name__ == "__main__":
    # Test your code here
    words, embeddings = load_embeddings("bow2.words")
    print("Loaded the embeddings!")

    # Answer to Problem 4:
    print("Answer to Problem 4:")
    for word in ["dog", "house", "france", "february", "school"]:
        print("Most similar words to {}:".format(word), top_k(word, 5, words, embeddings))

    # Answer to Problem 6:
    print("Demonstrative examples for problem 6:")
    for l in [["dogs","cat","dachshund","puppy","pig"], ["diabetes", "obesity"], ["sad", "happy"]]:
        print("Most similar word to {}:".format(l), sim(l, words, embeddings))