import matplotlib.pyplot as plt


def visualize_skipgram(sentence, window_size=2):
    """
    visualize how skipgram pairs are generated
    """

    words = sentence.split()

    print("sentence:")
    print(words)
    print()

    pairs = []

    for i in range(len(words)):

        center_word = words[i]

        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)

        for j in range(start, end):

            if i == j:
                continue

            context_word = words[j]

            pairs.append((center_word, context_word))

    print("generated pairs:")
    for pair in pairs:
        print(pair)

    return pairs


def plot_skipgram(sentence):

    words = sentence.split()

    x_positions = list(range(len(words)))
    y_positions = [0] * len(words)

    plt.figure(figsize=(10, 2))

    for i, word in enumerate(words):
        plt.scatter(i, 0)
        plt.text(i, 0.05, word, ha="center")

    window_size = 2

    for i in range(len(words)):

        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)

        for j in range(start, end):

            if i == j:
                continue

            plt.plot([i, j], [0, 0], alpha=0.2)

    plt.title("skipgram context connections")
    plt.yticks([])
    plt.xlabel("word position")

    plt.show()


def main():

    sentence = "garlic chicken pasta olive oil salt pepper"

    visualize_skipgram(sentence)

    plot_skipgram(sentence)


if __name__ == "__main__":
    main()