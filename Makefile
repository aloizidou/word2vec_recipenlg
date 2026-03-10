.PHONY: help download_data prepare_data build_pairs train_model inspect_embeddings plot_embeddings plot_word_frequency visualize_training_diagram run_all

help:
	@echo "available commands:"
	@echo "  make download_data              - download the recipenlg dataset from kaggle into data/raw"
	@echo "  make prepare_data               - clean recipe text and save tokenized recipes"
	@echo "  make build_pairs                - build the vocabulary and create skip-gram training pairs"
	@echo "  make train_model                - train the word2vec skip-gram model with negative sampling"
	@echo "  make inspect_embeddings         - print nearest neighbors for some example words"
	@echo "  make plot_embeddings            - create pca and tsne visualizations of word embeddings"
	@echo "  make plot_word_frequency        - plot the word frequency distribution (zipf's law)"
	@echo "  make visualize_training_diagram - generate a simple diagram explaining skip-gram training"
	@echo "  make run_all                    - run the full pipeline"


download_data:
	@echo "downloading the recipenlg dataset..."
	python dataset/download_data.py


prepare_data:
	@echo "cleaning recipes and creating tokenized text..."
	python dataset/prepare_data.py


build_pairs:
	@echo "building vocabulary and generating skip-gram training pairs..."
	python dataset/build_pairs.py


train_model:
	@echo "training word2vec model using numpy implementation..."
	python models/train_word2vec.py


inspect_embeddings:
	@echo "printing nearest words from the trained embeddings..."
	python visualization/inspect_embeddings.py


plot_embeddings:
	@echo "creating pca and tsne plots of selected word embeddings..."
	python visualization/plot_embeddings.py


plot_word_frequency:
	@echo "creating word frequency distribution plot..."
	python visualization/plot_word_frequency.py


visualize_training_diagram:
	@echo "creating diagram explaining skip-gram training..."
	python visualization/word2vec_training_diagram.py


run_all:
	@echo "running the full word2vec pipeline..."
	$(MAKE) prepare_data
	$(MAKE) build_pairs
	$(MAKE) train_model
	$(MAKE) inspect_embeddings
	$(MAKE) plot_embeddings
	$(MAKE) plot_word_frequency
	$(MAKE) visualize_training_diagram