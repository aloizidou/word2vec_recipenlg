# Word2Vec from Scratch (NumPy) – RecipeNLG

This project implements **Word2Vec (Skip-Gram with Negative Sampling)** from scratch using **pure NumPy**.

The goal is to understand how word embeddings are learned by implementing the full training pipeline without using deep learning frameworks such as PyTorch or TensorFlow.

The model is trained on a subset of the **RecipeNLG dataset**, allowing the embeddings to learn relationships between ingredients and cooking terms.

---

# How to Run the Project

The project includes a **Makefile** that runs each stage of the pipeline.

### Show available commands
make help


### Run the full pipeline
make run_all


### Main steps
make prepare_data
make build_pairs
make train_model
make inspect_embeddings
make plot_embeddings
make plot_word_frequency
make visualize_training_diagram


---

# Implementation Overview

The Word2Vec implementation follows the standard **Skip-Gram with Negative Sampling** approach.

### 1. Text preprocessing

Recipes are cleaned and tokenized to create sequences of words.

Example:
garlic chicken pasta olive oil


---

### 2. Vocabulary creation

All unique words are collected and mapped to integer indices.

These indices allow words to be used in matrix operations during training.

---

### 3. Skip-Gram training pairs

For each word in a sentence, the model generates training pairs with nearby words within a fixed context window.

Example sentence:
garlic chicken pasta olive oil


Training pairs:

(pasta → garlic)
(pasta → chicken)
(pasta → olive)
(pasta → oil)


---

### 4. Word2Vec model

The model learns two embedding matrices:
1. input embeddings
2. output embeddings


During training:
center word embedding → predicts surrounding words


The embeddings are optimized using **stochastic gradient descent**.

---

### 5. Negative sampling

Instead of computing probabilities over the entire vocabulary, the model samples a few random words as negative examples.

This significantly reduces computation while still learning meaningful embeddings.

---

# Training Example

The diagram below illustrates a Skip-Gram training step.

A center word predicts surrounding context words.

![Skip-Gram Training](reports/figures/skipgram_training_example.png)

---

# Word Frequency Distribution

The dataset follows **Zipf's law**, where a few words appear very frequently while most words appear rarely.

![Word Frequency](reports/figures/word_frequency_distribution.png)

---

# Word Embedding Visualization

The learned embeddings can be visualized using dimensionality reduction.

### PCA projection

![PCA](reports/figures/embedding_pca.png)

### t-SNE projection

![t-SNE](reports/figures/embedding_tsne.png)

Related ingredients and cooking terms cluster together in the embedding space.

---

# Technologies Used

- Python
- NumPy
- Matplotlib
- scikit-learn

No deep learning frameworks were used.

---

# Author

Andrea Loizidou