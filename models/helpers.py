import numpy as np
import random


def sigmoid(value):
    """simple sigmoid function"""
    return 1 / (1 + np.exp(-value))


def initialize_embeddings(vocabulary_size, embedding_size):
    """
    create the input and output embedding matrices
    we initialize them with small random values
    """

    input_embeddings = np.random.uniform(
        low=-0.5,
        high=0.5,
        size=(vocabulary_size, embedding_size)
    )

    output_embeddings = np.random.uniform(
        low=-0.5,
        high=0.5,
        size=(vocabulary_size, embedding_size)
    )

    return input_embeddings, output_embeddings


def sample_negative_words(vocabulary_size, target_word_id, negative_sample_count):
    """
    randomly sample words that are not the true context word
    """

    negative_word_ids = []

    while len(negative_word_ids) < negative_sample_count:

        random_id = random.randint(0, vocabulary_size - 1)

        if random_id != target_word_id:
            negative_word_ids.append(random_id)

    return negative_word_ids