import pickle
from pathlib import Path
import numpy as np

from helpers import sigmoid
from helpers import initialize_embeddings
from helpers import sample_negative_words


PAIR_FILE = Path("data/processed/skipgram_pairs.pkl")
VOCAB_FILE = Path("data/processed/vocabulary.pkl")

MODEL_OUTPUT = Path("models/word2vec_embeddings.pkl")

EMBEDDING_SIZE = 50
LEARNING_RATE = 0.025
NEGATIVE_SAMPLE_COUNT = 5
EPOCHS = 2


def load_data():
    """load vocabulary and skipgram pairs"""

    print("loading vocabulary and training pairs...")

    with open(VOCAB_FILE, "rb") as f:
        word_to_id, id_to_word = pickle.load(f)

    with open(PAIR_FILE, "rb") as f:
        training_pairs = pickle.load(f)

    vocabulary_size = len(word_to_id)

    print("vocabulary size:", vocabulary_size)
    print("number of training pairs:", len(training_pairs))

    return word_to_id, id_to_word, training_pairs, vocabulary_size


def train_word2vec(training_pairs, vocabulary_size):
    """
    train skip-gram word2vec with negative sampling
    """

    input_embeddings, output_embeddings = initialize_embeddings(
        vocabulary_size,
        EMBEDDING_SIZE
    )

    for epoch_index in range(EPOCHS):

        print("starting epoch:", epoch_index + 1)

        total_loss = 0

        for center_word_id, context_word_id in training_pairs:

            center_vector = input_embeddings[center_word_id]
            context_vector = output_embeddings[context_word_id]

            negative_word_ids = sample_negative_words(
                vocabulary_size,
                context_word_id,
                NEGATIVE_SAMPLE_COUNT
            )

            positive_score = np.dot(center_vector, context_vector)
            positive_probability = sigmoid(positive_score)

            positive_loss = -np.log(positive_probability)

            negative_loss = 0

            for negative_word_id in negative_word_ids:

                negative_vector = output_embeddings[negative_word_id]

                negative_score = np.dot(center_vector, negative_vector)

                negative_probability = sigmoid(-negative_score)

                negative_loss += -np.log(negative_probability)

            loss = positive_loss + negative_loss

            total_loss += loss

            positive_gradient = positive_probability - 1

            input_embeddings[center_word_id] -= (
                LEARNING_RATE * positive_gradient * context_vector
            )

            output_embeddings[context_word_id] -= (
                LEARNING_RATE * positive_gradient * center_vector
            )

            for negative_word_id in negative_word_ids:

                negative_vector = output_embeddings[negative_word_id]

                negative_score = np.dot(center_vector, negative_vector)

                negative_probability = sigmoid(-negative_score)

                negative_gradient = 1 - negative_probability

                input_embeddings[center_word_id] -= (
                    LEARNING_RATE * negative_gradient * negative_vector
                )

                output_embeddings[negative_word_id] -= (
                    LEARNING_RATE * negative_gradient * center_vector
                )

        average_loss = total_loss / len(training_pairs)

        print("epoch finished")
        print("average loss:", average_loss)

    return input_embeddings


def save_model(input_embeddings, id_to_word):
    """save trained embeddings"""

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump((input_embeddings, id_to_word), f)

    print("model saved to:", MODEL_OUTPUT)


def main():

    word_to_id, id_to_word, training_pairs, vocabulary_size = load_data()

    input_embeddings = train_word2vec(training_pairs, vocabulary_size)

    save_model(input_embeddings, id_to_word)


if __name__ == "__main__":
    main()