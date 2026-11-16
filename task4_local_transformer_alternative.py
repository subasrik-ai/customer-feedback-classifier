
import re
import csv
import json
import random
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


# ============================================================
# CONFIGURATION
# ============================================================

random.seed(42)
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "task4_sentiment_model.pth"
VOCAB_PATH = MODEL_DIR / "task4_vocabulary.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# ============================================================
# 1. TOKENIZATION
# ============================================================

def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def demonstrate_tokenization():
    print("\n" + "=" * 55)
    print("1. TOKENIZER DEMONSTRATION")
    print("=" * 55)

    sentences = [
        "The customer loved the new product!",
        "The delivery was slow and disappointing."
    ]

    for sentence in sentences:
        tokens = tokenize(sentence)

        # Demonstration IDs, not pretrained Transformer IDs
        token_ids = list(range(1, len(tokens) + 1))
        attention_mask = [1] * len(tokens)

        print("\nSentence:", sentence)
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)
        print("Attention Mask:", attention_mask)


# ============================================================
# 2. CREATE SAMPLE SENTIMENT DATASET
# ============================================================

def create_dataset():
    positive_templates = [
        "I love the {}",
        "The {} was excellent",
        "Very happy with the {}",
        "Amazing {} and great support",
        "I am satisfied with the {}",
        "The {} works perfectly",
        "Excellent experience with the {}",
        "The {} was very helpful",
        "I recommend the {}",
        "Really enjoyed the {}"
    ]

    negative_templates = [
        "I hate the {}",
        "The {} was terrible",
        "Very unhappy with the {}",
        "Poor {} and bad support",
        "I am disappointed with the {}",
        "The {} does not work",
        "Terrible experience with the {}",
        "The {} was very frustrating",
        "I do not recommend the {}",
        "Really disliked the {}"
    ]

    subjects = [
        "product", "service", "delivery", "purchase",
        "customer support", "application", "website",
        "order", "experience", "device"
    ]

    texts = []
    labels = []

    for template in positive_templates:
        for subject in subjects:
            texts.append(template.format(subject))
            labels.append(1)

    for template in negative_templates:
        for subject in subjects:
            texts.append(template.format(subject))
            labels.append(0)

    combined = list(zip(texts, labels))
    random.shuffle(combined)

    texts, labels = zip(*combined)

    return list(texts), list(labels)


# ============================================================
# 3. BUILD VOCABULARY
# ============================================================

def build_vocabulary(texts):
    word_counts = Counter()

    for text in texts:
        word_counts.update(tokenize(text))

    vocabulary = {"<PAD>": 0, "<UNK>": 1}

    for word in sorted(word_counts):
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)

    return vocabulary


def encode_text(text, vocabulary):
    tokens = tokenize(text)

    token_ids = [
        vocabulary.get(token, vocabulary["<UNK>"])
        for token in tokens
    ]

    return token_ids


# ============================================================
# 4. DATASET AND DATALOADER
# ============================================================

class SentimentDataset(Dataset):

    def __init__(self, texts, labels, vocabulary):
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        token_ids = encode_text(text, self.vocabulary)

        # Fixed length for batch processing
        max_length = 20
        token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        while len(token_ids) < max_length:
            token_ids.append(0)
            attention_mask.append(0)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


# ============================================================
# 5. PYTORCH SENTIMENT CLASSIFIER
# ============================================================

class SentimentClassifier(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim=32):
        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=0
        )

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, token_ids, attention_mask):

        embeddings = self.embedding(token_ids)

        mask = attention_mask.unsqueeze(-1).float()

        masked_embeddings = embeddings * mask

        summed = masked_embeddings.sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1)

        average_embeddings = summed / lengths

        return self.network(average_embeddings)


# ============================================================
# 6. METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    classes = [0, 1]

    confusion = [[0, 0], [0, 0]]

    for true_label, predicted_label in zip(actual, predicted):
        confusion[true_label][predicted_label] += 1

    accuracy = sum(
        1 for a, p in zip(actual, predicted) if a == p
    ) / len(actual)

    f1_scores = []

    for class_id in classes:
        tp = confusion[class_id][class_id]

        fp = sum(
            confusion[row][class_id]
            for row in classes if row != class_id
        )

        fn = sum(
            confusion[class_id][column]
            for column in classes if column != class_id
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)

    return accuracy, macro_f1, confusion


# ============================================================
# 7. TRAINING
# ============================================================

def train_model():

    print("\n" + "=" * 55)
    print("2. DATASET PREPARATION")
    print("=" * 55)

    texts, labels = create_dataset()

    total = len(texts)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_texts = texts[:train_end]
    train_labels = labels[:train_end]

    val_texts = texts[train_end:val_end]
    val_labels = labels[train_end:val_end]

    test_texts = texts[val_end:]
    test_labels = labels[val_end:]

    vocabulary = build_vocabulary(train_texts)

    print("Total samples:", total)
    print("Training samples:", len(train_texts))
    print("Validation samples:", len(val_texts))
    print("Test samples:", len(test_texts))
    print("Vocabulary size:", len(vocabulary))

    train_dataset = SentimentDataset(
        train_texts, train_labels, vocabulary
    )

    val_dataset = SentimentDataset(
        val_texts, val_labels, vocabulary
    )

    test_dataset = SentimentDataset(
        test_texts, test_labels, vocabulary
    )

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=16
    )

    test_loader = DataLoader(
        test_dataset, batch_size=16
    )

    model = SentimentClassifier(len(vocabulary)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.005
    )

    print("\nModel parameters:",
          sum(p.numel() for p in model.parameters()))

    print("\n" + "=" * 55)
    print("3. MODEL TRAINING")
    print("=" * 55)

    start_time = time.time()

    best_val_accuracy = 0
    best_state = None

    epochs = 20

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for token_ids, masks, targets in train_loader:

            token_ids = token_ids.to(DEVICE)
            masks = masks.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(token_ids, masks)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()

        val_actual = []
        val_predicted = []

        with torch.no_grad():

            for token_ids, masks, targets in val_loader:

                outputs = model(
                    token_ids.to(DEVICE),
                    masks.to(DEVICE)
                )

                predictions = outputs.argmax(dim=1)

                val_actual.extend(targets.tolist())
                val_predicted.extend(predictions.cpu().tolist())

        val_accuracy = sum(
            a == p for a, p in zip(val_actual, val_predicted)
        ) / len(val_actual)

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Loss: {average_loss:.4f} "
            f"| Validation accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    training_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    # ========================================================
    # 8. TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 55)
    print("4. TEST EVALUATION")
    print("=" * 55)

    model.eval()

    actual = []
    predicted = []

    inference_start = time.time()

    with torch.no_grad():

        for token_ids, masks, targets in test_loader:

            outputs = model(
                token_ids.to(DEVICE),
                masks.to(DEVICE)
            )

            predictions = outputs.argmax(dim=1)

            actual.extend(targets.tolist())
            predicted.extend(predictions.cpu().tolist())

    inference_time = time.time() - inference_start

    accuracy, macro_f1, confusion = calculate_metrics(
        actual, predicted
    )

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print("Confusion matrix:")
    print(confusion)

    print(f"Training time: {training_time:.4f} seconds")
    print(f"Test inference time: {inference_time:.4f} seconds")

    print("\nCOMPARISON TABLE")
    print("-" * 65)
    print(f"{'Metric':<25}{'Task 4 PyTorch model':<25}{'Task 3 baseline'}")
    print("-" * 65)
    print(f"{'Accuracy':<25}{accuracy:.4f}")
    print(f"{'Macro F1':<25}{macro_f1:.4f}")
    print(f"{'Training time (sec)':<25}{training_time:.4f}")
    print(f"{'Inference time (sec)':<25}{inference_time:.4f}")
    print("Task 3 values: enter your actual measured results separately.")

    # ========================================================
    # 9. SAVE MODEL AND VOCABULARY
    # ========================================================

    torch.save(model.state_dict(), MODEL_PATH)

    with open(VOCAB_PATH, "w", encoding="utf-8") as file:
        json.dump(vocabulary, file, indent=4)

    print("\nModel saved to:", MODEL_PATH)
    print("Vocabulary saved to:", VOCAB_PATH)

    return model, vocabulary


# ============================================================
# 10. SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text, model, vocabulary):

    model.eval()

    token_ids = encode_text(text, vocabulary)[:20]

    attention_mask = [1] * len(token_ids)

    while len(token_ids) < 20:
        token_ids.append(0)
        attention_mask.append(0)

    ids_tensor = torch.tensor(
        [token_ids], dtype=torch.long
    ).to(DEVICE)

    mask_tensor = torch.tensor(
        [attention_mask], dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model(ids_tensor, mask_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_class = probabilities.argmax().item()

    label = "positive" if predicted_class == 1 else "negative"

    confidence = probabilities[predicted_class].item()

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }


# ============================================================
# 11. SIMPLE NAMED ENTITY RECOGNITION DEMONSTRATION
# ============================================================

def simple_ner(text):

    print("\nNER INPUT:", text)

    # Demonstration only: this is not a trained NER model.
    patterns = {
        "PERSON": r"\b(?:John|Mary|David|Sarah)\b",
        "LOCATION": r"\b(?:India|Chennai|London|Paris|New York)\b",
        "ORGANIZATION": r"\b(?:Google|Microsoft|Amazon|OpenAI)\b"
    }

    entities = []

    for entity_type, pattern in patterns.items():

        for match in re.finditer(pattern, text, re.IGNORECASE):

            entities.append({
                "text": match.group(),
                "label": entity_type
            })

    print("Detected entities:", entities)

    return entities


# ============================================================
# 12. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Feedback Sentiment API",
    description="PyTorch sentiment classification demonstration",
    version="1.0"
)


class PredictionRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="Customer feedback text"
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("Text cannot be empty or whitespace")
        return value


class PredictionResponse(BaseModel):
    label: str
    confidence: float


api_model = None
api_vocabulary = None


def load_api_model():

    global api_model, api_vocabulary

    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        raise RuntimeError(
            "Model files are missing. Run this Python file first to train the model."
        )

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        api_vocabulary = json.load(file)

    api_model = SentimentClassifier(
        len(api_vocabulary)
    ).to(DEVICE)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    api_model.load_state_dict(state)
    api_model.eval()

    print("API model loaded successfully.")


@app.on_event("startup")
def startup_event():
    load_api_model()


@app.get("/")
def home():
    return {
        "message": "Customer Feedback Sentiment API is running",
        "endpoint": "POST /predict"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_api(request: PredictionRequest):

    result = predict_sentiment(
        request.text,
        api_model,
        api_vocabulary
    )

    return result


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    demonstrate_tokenization()

    trained_model, trained_vocabulary = train_model()

    print("\n" + "=" * 55)
    print("5. SENTIMENT PREDICTION DEMONSTRATION")
    print("=" * 55)

    examples = [
        "I love this excellent product",
        "The service was terrible and disappointing"
    ]

    for text in examples:

        result = predict_sentiment(
            text, trained_model, trained_vocabulary
        )

        print("\nText:", text)
        print("Prediction:", result)

    print("\n" + "=" * 55)
    print("6. NER DEMONSTRATION")
    print("=" * 55)

    simple_ner(
        "John visited Chennai and works at Google."
    )

    print("\nTraining completed.")
    print("To start the API, run:")
    print(
        "uvicorn src.task4_local_transformer_alternative:app --reload"
    )
