import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


TOKEN_FILE = Path("data/interim/tokenized_recipes.pkl")
FIGURE_FOLDER = Path("reports/figures")


def load_tokens():
    """load tokenized recipes"""

    print("loading tokenized recipes...")

    with open(TOKEN_FILE, "rb") as f:
        token_lists = pickle.load(f)

    print("number of recipes:", len(token_lists))

    return token_lists


def count_words(token_lists):
    """count word frequencies"""

    print("counting word frequencies...")

    word_counter = Counter()

    for tokens in token_lists:
        for word in tokens:
            word_counter[word] += 1

    return word_counter


def plot_frequency_distribution(word_counter):

    print("creating frequency plot...")

    # sort words by frequency
    sorted_counts = sorted(word_counter.values(), reverse=True)

    # create rank values
    ranks = list(range(1, len(sorted_counts) + 1))

    plt.figure(figsize=(8, 6))

    plt.plot(ranks, sorted_counts)

    plt.title("word frequency distribution (zipf's law)")
    plt.xlabel("word rank")
    plt.ylabel("frequency")

    plt.yscale("log")
    plt.xscale("log")

    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

    plt.savefig(FIGURE_FOLDER / "word_frequency_distribution.png")
    plt.close()

    print("frequency plot saved")


def main():

    token_lists = load_tokens()

    word_counter = count_words(token_lists)

    plot_frequency_distribution(word_counter)


if __name__ == "__main__":
    main()