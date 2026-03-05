import torch
import numpy
import torch.nn as nn


class ShakespeareClassification(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(ShakespeareClassification, self).__init__()

        self.embed_size = embed_size

        self.W_embedding = nn.Parameter(torch.randn(vocab_size, embed_size)*0.1)
        self.W_query = nn.Parameter(torch.randn(embed_size, embed_size)*0.1)
        self.W_key = nn.Parameter(torch.randn(embed_size, embed_size)*0.1)
        self.W_value = nn.Parameter(torch.randn(embed_size, embed_size)*0.1)

        self.fc = nn.Parameter(torch.randn(embed_size, num_classes)*0.1)
    def forward(self, x):
        # x is (batch_size, seq_length)
        embedding = self.W_embedding[x] # (batch_size, seq_length, embed_size)
        Q = embedding @ self.W_query # (batch_size, seq_length, embed_size)
        K = embedding @ self.W_key # (batch_size, seq_length, embed_size)
        V = embedding @ self.W_value # (batch_size, seq_length, embed_size)

        attention_score = Q @ K.transpose(-2, -1) / self.embed_size**0.5 # (batch_size, seq_length, seq_length)
        attention_norm = torch.softmax(attention_score, dim=-1) # (batch_size, seq_length, seq_length)
        out = attention_norm @ V # (batch_size, seq_length, embed_size)

        sentence_vector = out.mean(dim=1) # (batch_size, embed_size)
        logits = sentence_vector @ self.fc # (batch_size, num_classes)
        return logits
    

def encode(string):
    return [char_to_idx[char] for char in string if char in char_to_idx]

def get_batch(dataset, batch_size, seq_length):
    inputs = []
    labels = []
    for _ in range(batch_size):
        sample = dataset[numpy.random.randint(0, len(dataset))]
        encoded = encode(sample[0])
        if len(encoded) < seq_length:
            encoded += [0] * (seq_length - len(encoded))
        else:
            encoded = encoded[:seq_length]
        inputs.append(encoded)
        labels.append(sample[1])
    return torch.tensor(inputs), torch.tensor(labels)


def predict_speaker(model, text, seq_length):
    model.eval()
    with torch.no_grad():
        encoded = encode(text.lower())
        if len(encoded) < seq_length:
            encoded += [0] * (seq_length - len(encoded))
        else:
            encoded = encoded[:seq_length]
        input_tensor = torch.tensor([encoded])
        logits = model(input_tensor)
        predicted_class = torch.argmax(logits, dim=-1).item()
        return "ROMEO" if predicted_class == 0 else "JULIET"


if __name__ == "__main__":
    with open("input.txt", "r") as f:
        text = f.read()
    blocks = text.split("\n\n")
    print(f"Total blocks: {len(blocks)}")

    dataset = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        label = lines[0].strip()
        content = " ".join(lines[1:]).strip().lower()
        if label == "ROMEO:":
            dataset.append((content, 0))
        elif label == "JULIET:":
            dataset.append((content, 1))
    print(f"Total samples: {len(dataset)}")

    all_text = " ".join([sample[0] for sample in dataset])
    chars = sorted(list(set(all_text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    embed_size = 128
    num_classes = 2
    model = ShakespeareClassification(vocab_size, embed_size, num_classes)  
    seq_length = 32
    batch_size = 8
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #starting training
    for step in range(500):
        X, y = get_batch(dataset, batch_size, seq_length)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

    print("Training Complete! The Encoder has learned to identify Romeo and Juliet.")

    print("\n--- Testing the Model ---")
    
    test_sentences = [
        "O, swear not by the moon, the inconstant moon!", # Actual Juliet line
        "With love's light wings did I o'er-perch these walls", # Actual Romeo line
        "I bring thee tidings of the prince's doom.", # Unseen text
        "My heart is breaking for thee.", # Made up text
    ]

    for sentence in test_sentences:
        prediction = predict_speaker(model, sentence, seq_length)
        print(f"Sentence: '{sentence}'")
        print(f"Predicted Speaker: {prediction}\n")
