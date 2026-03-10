import pandas as pd
import re
from pathlib import Path
import pickle


RAW_DATA_FILE = Path("data/raw/RecipeNLG_dataset.csv")
OUTPUT_FILE = Path("data/interim/tokenized_recipes.pkl")

MAX_ROWS = 10000  # we are using a small subset so training stays fast and easy to debug


def load_recipe_data():
    """load a small subset of the recipe dataset"""
    print("loading dataset...")
    recipe_table = pd.read_csv(RAW_DATA_FILE, nrows=MAX_ROWS)
    print("rows loaded:", len(recipe_table))
    return recipe_table


def build_recipe_text(row):
    """
    combine the main recipe text fields into one string
    we keep this very simple and just join the columns
    """
    title = str(row.get("title", ""))
    ingredients = str(row.get("ingredients", ""))
    directions = str(row.get("directions", ""))

    combined_text = title + " " + ingredients + " " + directions
    return combined_text


def clean_text(text):
    """
    basic cleaning
    we lowercase the text and remove unusual characters
    """
    text = text.lower()

    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    """
    split the cleaned text into individual words
    """
    tokens = text.split(" ")
    return tokens


def preprocess_recipes():
    """
    main preprocessing pipeline

    here we:
    1. load a small subset of the dataset
    2. combine the recipe text fields
    3. clean the text
    4. tokenize the words
    5. save the token lists for later steps
    """
    recipe_table = load_recipe_data()

    token_lists = []

    for _, row in recipe_table.iterrows():

        # build a single text string for the recipe
        combined_text = build_recipe_text(row)

        # apply simple cleaning
        cleaned_text = clean_text(combined_text)

        # split into tokens
        tokens = tokenize_text(cleaned_text)

        if len(tokens) > 0:
            token_lists.append(tokens)

    print("number of recipes processed:", len(token_lists))

    # create the folder if it does not exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # save tokenized recipes so we can reuse them later
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(token_lists, f)

    print("tokenized recipes saved to:", OUTPUT_FILE)


def main():
    preprocess_recipes()


if __name__ == "__main__":
    main()