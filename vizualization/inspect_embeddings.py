import pickle
import numpy as np
from pathlib import Path


MODEL_FILE = Path("models/word2vec_embeddings.pkl")


def cosine_similarity(vector_a, vector_b):
    """compute cosine similarity between two vectors"""

    dot_product = np.dot(vector_a, vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return dot_product / (norm_a * norm_b)


def find_nearest_words(query_word, embeddings, word_to_id, id_to_word, top_k=10):
    """find nearest words using cosine similarity"""

    if query_word not in word_to_id:
        print("word not in vocabulary")
        return

    query_id = word_to_id[query_word]
    query_vector = embeddings[query_id]

    similarities = []

    for word_id in range(len(embeddings)):

        if word_id == query_id:
            continue

        word_vector = embeddings[word_id]

        similarity = cosine_similarity(query_vector, word_vector)

        similarities.append((similarity, id_to_word[word_id]))

    similarities.sort(reverse=True)

    print("nearest words to:", query_word)

    for score, word in similarities[:top_k]:
        print(word, score)


def main():

    with open(MODEL_FILE, "rb") as f:
        embeddings, id_to_word = pickle.load(f)

    word_to_id = {word: i for i, word in id_to_word.items()}

    find_nearest_words("garlic", embeddings, word_to_id, id_to_word)
    print("=======================")
    find_nearest_words("chicken", embeddings, word_to_id, id_to_word)
    print("=======================")
    find_nearest_words("sugar", embeddings, word_to_id, id_to_word)


if __name__ == "__main__":
    main()