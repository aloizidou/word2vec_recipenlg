import pickle
from pathlib import Path
from collections import Counter


TOKEN_FILE = Path("data/interim/tokenized_recipes.pkl")

VOCAB_FILE = Path("data/processed/vocabulary.pkl")
PAIR_FILE = Path("data/processed/skipgram_pairs.pkl")

WINDOW_SIZE = 2
MIN_WORD_COUNT = 5


def load_tokens():
    """load tokenized recipes"""
    print("loading tokenized recipes...")

    with open(TOKEN_FILE, "rb") as f:
        token_lists = pickle.load(f)

    print("number of recipes loaded:", len(token_lists))

    return token_lists


def build_vocabulary(token_lists):
    """
    build a simple vocabulary
    we count all words and remove very rare ones
    """
    print("building vocabulary...")

    word_counter = Counter()

    for tokens in token_lists:
        for word in tokens:
            word_counter[word] += 1

    vocabulary = []

    for word, count in word_counter.items():
        if count >= MIN_WORD_COUNT:
            vocabulary.append(word)

    print("vocabulary size:", len(vocabulary))

    return vocabulary


def build_word_mappings(vocabulary):
    """
    create word to id and id to word dictionaries
    """
    word_to_id = {}
    id_to_word = {}

    for index, word in enumerate(vocabulary):
        word_to_id[word] = index
        id_to_word[index] = word

    return word_to_id, id_to_word


def convert_tokens_to_ids(token_lists, word_to_id):
    """
    convert recipes from words to word ids
    """
    print("converting words to ids...")

    recipe_id_lists = []

    for tokens in token_lists:

        word_ids = []

        for word in tokens:
            if word in word_to_id:
                word_ids.append(word_to_id[word])

        if len(word_ids) > 0:
            recipe_id_lists.append(word_ids)

    return recipe_id_lists


def build_skipgram_pairs(recipe_id_lists):
    """
    create skip-gram training pairs
    """
    print("building skip-gram pairs...")

    pairs = []

    for recipe in recipe_id_lists:

        for i in range(len(recipe)):

            center_word_id = recipe[i]

            start = max(0, i - WINDOW_SIZE)
            end = min(len(recipe), i + WINDOW_SIZE + 1)

            for j in range(start, end):

                if i == j:
                    continue

                context_word_id = recipe[j]

                pairs.append((center_word_id, context_word_id))

    print("number of training pairs:", len(pairs))

    return pairs


def save_results(vocabulary, word_to_id, id_to_word, pairs):
    """save vocabulary and training pairs"""

    VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(VOCAB_FILE, "wb") as f:
        pickle.dump((word_to_id, id_to_word), f)

    with open(PAIR_FILE, "wb") as f:
        pickle.dump(pairs, f)

    print("saved vocabulary and pairs")


def main():

    token_lists = load_tokens()

    vocabulary = build_vocabulary(token_lists)

    word_to_id, id_to_word = build_word_mappings(vocabulary)

    recipe_id_lists = convert_tokens_to_ids(token_lists, word_to_id)

    pairs = build_skipgram_pairs(recipe_id_lists)

    save_results(vocabulary, word_to_id, id_to_word, pairs)


if __name__ == "__main__":
    main()