import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


MODEL_FILE = Path("models/word2vec_embeddings.pkl")
FIGURE_FOLDER = Path("reports/figures")


def load_embeddings():
    """load trained embeddings"""

    print("loading embeddings...")

    with open(MODEL_FILE, "rb") as f:
        embeddings, id_to_word = pickle.load(f)

    print("number of embeddings:", len(embeddings))

    return embeddings, id_to_word


def select_words(id_to_word, word_list):
    """
    find the ids of words we want to visualize
    """

    word_ids = []
    words_found = []

    for word_id, word in id_to_word.items():

        if word in word_list:
            word_ids.append(word_id)
            words_found.append(word)

    print("words used for visualization:", words_found)

    return word_ids, words_found


def plot_pca(embeddings, word_ids, words):

    print("creating pca visualization...")

    vectors = embeddings[word_ids]

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    plt.figure(figsize=(8, 6))

    for i, word in enumerate(words):

        x = reduced_vectors[i, 0]
        y = reduced_vectors[i, 1]

        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word)

    plt.title("word2vec embeddings (pca)")
    plt.xlabel("component 1")
    plt.ylabel("component 2")

    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

    plt.savefig(FIGURE_FOLDER / "embedding_pca.png")
    plt.close()

    print("pca figure saved")


def plot_tsne(embeddings, word_ids, words):

    print("creating tsne visualization...")

    vectors = embeddings[word_ids]

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)

    reduced_vectors = tsne.fit_transform(vectors)

    plt.figure(figsize=(8, 6))

    for i, word in enumerate(words):

        x = reduced_vectors[i, 0]
        y = reduced_vectors[i, 1]

        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word)

    plt.title("word2vec embeddings (tsne)")
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")

    plt.savefig(FIGURE_FOLDER / "embedding_tsne.png")
    plt.close()

    print("tsne figure saved")


def main():

    embeddings, id_to_word = load_embeddings()

    # chose only a few words so that the plot is clean! (with no overlaps of words!) but we can still get the point
    words_to_plot = [
        "garlic",
        "onion",
        "basil",
        "oregano",
        "parsley",
        "pepper",
        "salt",
        "chicken",
        "turkey",
        "beef",
        "sugar",
        "cocoa",
        "vanilla",
        "egg",
        "butter"
    ]

    word_ids, words_found = select_words(id_to_word, words_to_plot)

    plot_pca(embeddings, word_ids, words_found)

    plot_tsne(embeddings, word_ids, words_found)


if __name__ == "__main__":
    main()