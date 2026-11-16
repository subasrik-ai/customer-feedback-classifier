
# ============================================================
# TASK 3: CLASSICAL NLP BASELINE AND EMBEDDINGS
# Temporary Python + PyTorch implementation
# ============================================================

import os
import csv
import re
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# STEP 1: REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# STEP 2: CREATE A LABELLED TEXT DATASET
# ============================================================

positive_starts = [
    "I love",
    "I really like",
    "I am happy with",
    "I am satisfied with",
    "I highly recommend",
    "I enjoyed",
    "I appreciate",
    "I am impressed by"
]

negative_starts = [
    "I hate",
    "I really dislike",
    "I am unhappy with",
    "I am disappointed with",
    "I cannot recommend",
    "I regret",
    "I am frustrated with",
    "I am annoyed by"
]

subjects = [
    "the customer service",
    "the product quality",
    "the delivery service",
    "the mobile application",
    "the website"
]

positive_endings = [
    "it works perfectly",
    "the experience was excellent",
    "everything was smooth",
    "the staff were helpful",
    "the service was fantastic"
]

negative_endings = [
    "it does not work properly",
    "the experience was terrible",
    "everything was frustrating",
    "the staff were unhelpful",
    "the service was extremely poor"
]

dataset = []

for start in positive_starts:
    for subject in subjects:
        for ending in positive_endings:
            dataset.append((
                f"{start} {subject} because {ending}",
                1
            ))

for start in negative_starts:
    for subject in subjects:
        for ending in negative_endings:
            dataset.append((
                f"{start} {subject} because {ending}",
                0
            ))

random.shuffle(dataset)

print("\nTotal messages:", len(dataset))


# ============================================================
# STEP 3: CLEAN AND TOKENIZE TEXT
# ============================================================

def tokenize(text):

    text = text.lower()

    tokens = re.findall(r"\b[a-z]+\b", text)

    return tokens


cleaned_dataset = []

for text, label in dataset:

    tokens = tokenize(text)

    cleaned_dataset.append((tokens, label))


# Save dataset
os.makedirs("data", exist_ok=True)

with open(
    "data/task3_dataset.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(["text", "label"])

    for tokens, label in cleaned_dataset:

        writer.writerow([" ".join(tokens), label])


print("Dataset saved!")
print("Positive:", sum(label == 1 for _, label in cleaned_dataset))
print("Negative:", sum(label == 0 for _, label in cleaned_dataset))


# ============================================================
# STEP 4: SPLIT DATA INTO TRAIN, VALIDATION AND TEST
# ============================================================

positive_data = [
    item for item in cleaned_dataset if item[1] == 1
]

negative_data = [
    item for item in cleaned_dataset if item[1] == 0
]

random.shuffle(positive_data)
random.shuffle(negative_data)


def split_class(items):

    train_end = int(len(items) * 0.70)
    val_end = int(len(items) * 0.85)

    return (
        items[:train_end],
        items[train_end:val_end],
        items[val_end:]
    )


pos_train, pos_val, pos_test = split_class(positive_data)
neg_train, neg_val, neg_test = split_class(negative_data)

train_data = pos_train + neg_train
val_data = pos_val + neg_val
test_data = pos_test + neg_test

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

print("\nData split:")
print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Testing:", len(test_data))


# ============================================================
# STEP 5: BUILD TF-IDF VECTORIZER WITHOUT SCIKIT-LEARN
# ============================================================

# Build vocabulary from training data only.

vocabulary = sorted(set(
    word
    for tokens, label in train_data
    for word in tokens
))

word_to_index = {
    word: index
    for index, word in enumerate(vocabulary)
}

vocab_size = len(vocabulary)

# Calculate document frequency.

document_frequency = Counter()

for tokens, label in train_data:

    for word in set(tokens):

        document_frequency[word] += 1


# Calculate inverse document frequency.

number_of_documents = len(train_data)

idf = {}

for word in vocabulary:

    idf[word] = (
        math.log(
            (1 + number_of_documents) /
            (1 + document_frequency[word])
        ) + 1
    )


# Convert text into TF-IDF vectors.

def create_tfidf_vector(tokens):

    vector = [0.0] * vocab_size

    word_counts = Counter(tokens)

    total_words = len(tokens)

    for word, count in word_counts.items():

        if word in word_to_index:

            tf = count / total_words

            tfidf = tf * idf[word]

            index = word_to_index[word]

            vector[index] = tfidf

    # L2 normalization
    norm = math.sqrt(sum(value * value for value in vector))

    if norm > 0:
        vector = [value / norm for value in vector]

    return vector


def make_tfidf_dataset(samples):

    features = [
        create_tfidf_vector(tokens)
        for tokens, label in samples
    ]

    labels = [
        label
        for tokens, label in samples
    ]

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )


X_train, y_train = make_tfidf_dataset(train_data)
X_val, y_val = make_tfidf_dataset(val_data)
X_test, y_test = make_tfidf_dataset(test_data)

print("\nTF-IDF information:")
print("Vocabulary size:", vocab_size)
print("Training feature shape:", X_train.shape)


# ============================================================
# STEP 6: CREATE PYTORCH LOGISTIC REGRESSION CLASSIFIER
# ============================================================

# A single Linear layer with two outputs acts as
# a multinomial logistic regression classifier.

class TextClassifier(nn.Module):

    def __init__(self, input_size, number_of_classes):

        super().__init__()

        self.linear = nn.Linear(
            input_size,
            number_of_classes
        )

    def forward(self, x):

        return self.linear(x)


def train_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=40
):

    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    model = TextClassifier(
        X_train.shape[1],
        2
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_state = None

    patience = 6
    no_improvement = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        model.eval()

        with torch.no_grad():

            val_outputs = model(X_val.to(device))

            val_predictions = val_outputs.argmax(dim=1)

            val_accuracy = (
                val_predictions.cpu() == y_val
            ).float().mean().item()

        average_loss = total_loss / len(train_dataset)

        if epoch == 0 or (epoch + 1) % 5 == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {average_loss:.4f} | "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

            no_improvement = 0

        else:

            no_improvement += 1

        if no_improvement >= patience:

            print("Early stopping triggered!")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


print("\nTraining TF-IDF classifier...")

tfidf_model = train_classifier(
    X_train,
    y_train,
    X_val,
    y_val
)


# ============================================================
# STEP 7: EVALUATION METRICS
# ============================================================

def evaluate_predictions(actual, predicted):

    actual = actual.tolist()
    predicted = predicted.tolist()

    # Confusion matrix:
    # Rows = actual class
    # Columns = predicted class

    matrix = [[0, 0], [0, 0]]

    for true_label, predicted_label in zip(actual, predicted):

        matrix[true_label][predicted_label] += 1

    total_correct = matrix[0][0] + matrix[1][1]

    accuracy = total_correct / len(actual)

    class_f1_scores = []

    for class_id in [0, 1]:

        tp = matrix[class_id][class_id]

        fp = sum(
            matrix[row][class_id]
            for row in [0, 1]
            if row != class_id
        )

        fn = sum(
            matrix[class_id][column]
            for column in [0, 1]
            if column != class_id
        )

        precision = tp / (tp + fp) if tp + fp else 0

        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        class_f1_scores.append(f1)

    macro_f1 = sum(class_f1_scores) / 2

    return accuracy, macro_f1, matrix


tfidf_model.eval()

with torch.no_grad():

    test_outputs = tfidf_model(X_test.to(device))

    tfidf_predictions = (
        test_outputs.argmax(dim=1).cpu()
    )


tfidf_accuracy, tfidf_macro_f1, tfidf_matrix = (
    evaluate_predictions(y_test, tfidf_predictions)
)

print("\n========== TF-IDF RESULTS ==========")

print(f"Accuracy: {tfidf_accuracy:.4f}")
print(f"Macro F1: {tfidf_macro_f1:.4f}")

print("\nConfusion Matrix:")

print("                 Predicted Negative  Predicted Positive")
print(
    f"Actual Negative       {tfidf_matrix[0][0]:5d}"
    f"                {tfidf_matrix[0][1]:5d}"
)

print(
    f"Actual Positive       {tfidf_matrix[1][0]:5d}"
    f"                {tfidf_matrix[1][1]:5d}"
)


# ============================================================
# STEP 8: PRINT TOP 10 IMPORTANT WORDS PER CLASS
# ============================================================

weights = tfidf_model.linear.weight.detach().cpu()

# Class 0 = negative
# Class 1 = positive

print("\n========== IMPORTANT WORDS ==========")

for class_id, class_name in [
    (0, "Negative"),
    (1, "Positive")
]:

    word_scores = []

    for word, index in word_to_index.items():

        score = weights[class_id][index].item()

        word_scores.append((word, score))

    word_scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    print(f"\nTop 10 words for {class_name} class:")

    for word, score in word_scores[:10]:

        print(f"{word:20s} {score:.4f}")


# ============================================================
# STEP 9: TRAIN SIMPLE WORD2VEC-STYLE EMBEDDINGS
# ============================================================

# These are locally trained skip-gram embeddings.
# They are NOT pretrained GloVe vectors.

# Add a small analogy demonstration corpus.
# This is only for practising word-vector arithmetic.

analogy_sentences = [
    "king man queen woman",
    "king is a man queen is a woman",
    "a king rules a kingdom",
    "a queen rules a kingdom",
    "man is a person woman is a person",
    "prince is a boy princess is a girl",
    "king queen prince princess man woman"
] * 30

embedding_sentences = [
    tokens for tokens, label in train_data
]

embedding_sentences += [
    tokenize(sentence)
    for sentence in analogy_sentences
]

embedding_vocabulary = sorted(set(
    word
    for sentence in embedding_sentences
    for word in sentence
))

embedding_word_to_index = {
    word: index
    for index, word in enumerate(embedding_vocabulary)
}

embedding_vocab_size = len(embedding_vocabulary)

# Create skip-gram centre/context pairs.

skipgram_pairs = []

window_size = 2

for sentence in embedding_sentences:

    for center_position, center_word in enumerate(sentence):

        start = max(0, center_position - window_size)

        end = min(
            len(sentence),
            center_position + window_size + 1
        )

        for context_position in range(start, end):

            if context_position == center_position:
                continue

            context_word = sentence[context_position]

            skipgram_pairs.append((
                embedding_word_to_index[center_word],
                embedding_word_to_index[context_word]
            ))


# Skip-gram model with negative sampling.

class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim):

        super().__init__()

        self.input_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.output_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        nn.init.uniform_(
            self.input_embeddings.weight,
            -0.5 / embedding_dim,
            0.5 / embedding_dim
        )

        nn.init.zeros_(
            self.output_embeddings.weight
        )

    def forward(self, center, context, negative):

        center_vectors = self.input_embeddings(center)

        context_vectors = self.output_embeddings(context)

        negative_vectors = self.output_embeddings(negative)

        positive_scores = (
            center_vectors * context_vectors
        ).sum(dim=1)

        negative_scores = torch.bmm(
            negative_vectors,
            center_vectors.unsqueeze(2)
        ).squeeze(2)

        positive_loss = F.logsigmoid(positive_scores)

        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)

        return -(positive_loss + negative_loss).mean()


embedding_dim = 50

skipgram_model = SkipGram(
    embedding_vocab_size,
    embedding_dim
).to(device)

embedding_optimizer = torch.optim.Adam(
    skipgram_model.parameters(),
    lr=0.01
)

print("\nTraining skip-gram embeddings...")

pair_count = len(skipgram_pairs)

batch_size = 512
embedding_epochs = 5
negative_samples = 5

for epoch in range(embedding_epochs):

    random.shuffle(skipgram_pairs)

    total_embedding_loss = 0.0

    for start in range(0, pair_count, batch_size):

        batch = skipgram_pairs[start:start + batch_size]

        centers = torch.tensor(
            [pair[0] for pair in batch],
            dtype=torch.long,
            device=device
        )

        contexts = torch.tensor(
            [pair[1] for pair in batch],
            dtype=torch.long,
            device=device
        )

        negatives = torch.randint(
            0,
            embedding_vocab_size,
            (len(batch), negative_samples),
            device=device
        )

        embedding_optimizer.zero_grad()

        loss = skipgram_model(
            centers,
            contexts,
            negatives
        )

        loss.backward()

        embedding_optimizer.step()

        total_embedding_loss += loss.item()

    print(
        f"Embedding Epoch {epoch + 1}/{embedding_epochs} | "
        f"Loss: {total_embedding_loss:.4f}"
    )


# Use input embeddings as word vectors.

word_vectors = (
    skipgram_model.input_embeddings.weight
    .detach()
    .cpu()
)


# ============================================================
# STEP 10: FIND NEAREST WORDS
# ============================================================

def nearest_words(target_word, top_k=5):

    if target_word not in embedding_word_to_index:

        print(f"Word '{target_word}' not found.")
        return

    target_index = embedding_word_to_index[target_word]

    target_vector = word_vectors[target_index]

    similarities = F.cosine_similarity(
        word_vectors,
        target_vector.unsqueeze(0),
        dim=1
    )

    similarities[target_index] = -1

    values, indices = torch.topk(
        similarities,
        min(top_k, embedding_vocab_size - 1)
    )

    print(f"\nFive nearest words to '{target_word}':")

    for score, index in zip(values, indices):

        word = embedding_vocabulary[index.item()]

        print(f"{word:20s} {score.item():.4f}")


nearest_words("service", 5)
nearest_words("product", 5)


# ============================================================
# STEP 11: WORD ANALOGY
# king - man + woman = queen
# ============================================================

def word_analogy(word_a, word_b, word_c):

    required_words = [word_a, word_b, word_c]

    for word in required_words:

        if word not in embedding_word_to_index:

            print(f"Missing word: {word}")
            return

    vector = (
        word_vectors[embedding_word_to_index[word_a]]
        - word_vectors[embedding_word_to_index[word_b]]
        + word_vectors[embedding_word_to_index[word_c]]
    )

    normalized_vectors = F.normalize(
        word_vectors,
        p=2,
        dim=1
    )

    normalized_target = F.normalize(
        vector.unsqueeze(0),
        p=2,
        dim=1
    )

    similarities = (
        normalized_vectors @ normalized_target.T
    ).squeeze(1)

    for word in required_words:

        similarities[embedding_word_to_index[word]] = -1

    best_index = torch.argmax(similarities).item()

    answer = embedding_vocabulary[best_index]

    print(
        f"\n{word_a} - {word_b} + {word_c} -> {answer}"
    )

    print(
        "Similarity:",
        round(similarities[best_index].item(), 4)
    )


word_analogy("king", "man", "woman")


# ============================================================
# STEP 12: CREATE SENTENCE VECTORS
# ============================================================

def sentence_vector(tokens):

    vectors = []

    for word in tokens:

        if word in embedding_word_to_index:

            index = embedding_word_to_index[word]

            vectors.append(word_vectors[index])

    if not vectors:

        return torch.zeros(embedding_dim)

    return torch.stack(vectors).mean(dim=0)


def make_embedding_dataset(samples):

    features = [
        sentence_vector(tokens).tolist()
        for tokens, label in samples
    ]

    labels = [
        label
        for tokens, label in samples
    ]

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )


X_train_embeddings, y_train_embeddings = (
    make_embedding_dataset(train_data)
)

X_val_embeddings, y_val_embeddings = (
    make_embedding_dataset(val_data)
)

X_test_embeddings, y_test_embeddings = (
    make_embedding_dataset(test_data)
)


# ============================================================
# STEP 13: TRAIN CLASSIFIER ON AVERAGED EMBEDDINGS
# ============================================================

print("\nTraining averaged-embedding classifier...")

embedding_classifier = train_classifier(
    X_train_embeddings,
    y_train_embeddings,
    X_val_embeddings,
    y_val_embeddings
)

embedding_classifier.eval()

with torch.no_grad():

    embedding_outputs = embedding_classifier(
        X_test_embeddings.to(device)
    )

    embedding_predictions = (
        embedding_outputs.argmax(dim=1).cpu()
    )


embedding_accuracy, embedding_macro_f1, embedding_matrix = (
    evaluate_predictions(
        y_test_embeddings,
        embedding_predictions
    )
)


# ============================================================
# STEP 14: COMPARE BOTH REPRESENTATIONS
# ============================================================

print("\n========== FINAL MODEL COMPARISON ==========")

print("\nTF-IDF + PyTorch classifier")

print(f"Accuracy: {tfidf_accuracy:.4f}")
print(f"Macro F1: {tfidf_macro_f1:.4f}")

print("\nAverage word embeddings + PyTorch classifier")

print(f"Accuracy: {embedding_accuracy:.4f}")
print(f"Macro F1: {embedding_macro_f1:.4f}")


# ============================================================
# STEP 15: SAVE MODELS
# ============================================================

os.makedirs("models", exist_ok=True)

torch.save(
    tfidf_model.state_dict(),
    "models/task3_tfidf_classifier.pth"
)

torch.save(
    embedding_classifier.state_dict(),
    "models/task3_embedding_classifier.pth"
)

torch.save(
    skipgram_model.state_dict(),
    "models/task3_skipgram_embeddings.pth"
)

print("\nModels saved successfully!")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== TASK 3 SUMMARY ==========")

print("Dataset size:", len(cleaned_dataset))
print("Vocabulary size:", vocab_size)

print(f"TF-IDF accuracy: {tfidf_accuracy:.4f}")
print(f"TF-IDF macro F1: {tfidf_macro_f1:.4f}")

print(f"Embedding accuracy: {embedding_accuracy:.4f}")
print(f"Embedding macro F1: {embedding_macro_f1:.4f}")

