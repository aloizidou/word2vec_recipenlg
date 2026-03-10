# Word2Vec Skip-Gram (NumPy Implementation)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-orange)
![Model](https://img.shields.io/badge/Model-Word2Vec-lightgrey)
![Variant](https://img.shields.io/badge/Variant-SkipGram-yellow)
![Training](https://img.shields.io/badge/Optimization-SGD-green)
![Dataset](https://img.shields.io/badge/Dataset-RecipeNLG-9cf)

---

## Overview

This repository implements the **core training loop of Word2Vec using pure NumPy**.  
No deep learning frameworks such as PyTorch or TensorFlow are used.

The project focuses on implementing the full **optimization procedure**, including:

- forward pass
- loss computation
- gradient calculation
- parameter updates with stochastic gradient descent

The model follows the **Skip-Gram with Negative Sampling** approach and is trained on a subset of the **RecipeNLG dataset**.

The goal is to demonstrate a clear understanding of how word embeddings are learned from text through context prediction.

---

## Logic

The training pipeline follows these steps:

1. Load and preprocess recipe text
2. Build the vocabulary and map each word to an index
3. Generate Skip-Gram training pairs from tokenized sentences
4. Initialize input and output embedding matrices
5. Compute prediction scores using dot products
6. Apply the negative sampling loss
7. Compute gradients and update parameters using SGD
8. Save trained embeddings and visualize the results

---

## System Flow

```mermaid
flowchart LR
    A[RecipeNLG Dataset] --> B[Text Preprocessing]
    B --> C[Tokenized Recipes]
    C --> D[Skip-Gram Training Pairs]
    D --> E[Word2Vec Training Loop]
    E --> F[Trained Embeddings]
    F --> G[Embedding Visualization]