import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.i2h = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.i2o = nn.Linear(embedding_dim + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        embedded = self.embedding(input_tensor)
        combined = torch.cat((embedded, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def train(model, X, y):
    model.to(device)  # Move the model to the correct device
    X_train_post_list = X["numerical_post"].tolist()
    y_train_list = y.tolist()

    for epoch in range(num_epochs):
        total_loss = 0
        for index in range(len(X)):
            input_tensor = torch.tensor(X_train_post_list[index], dtype=torch.long).unsqueeze(0).to(device)
            label_tensor = torch.tensor([y_train_list[index]], dtype=torch.long).to(device)
            hidden = model.init_hidden().to(device)

            optimizer.zero_grad()

            outputs = []
            for word in input_tensor[0]:
                word_tensor = word.unsqueeze(0).to(device)
                output, hidden = model(word_tensor, hidden)
                outputs.append(output)

            output = torch.stack(outputs).mean(dim=0)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'rnn_model.pth')


def tokenize(text):
    return nltk.word_tokenize(text)  # tokenize


def build_vocab(corpus):
    """Build a vocabulary mapping tokens to indices."""
    vocab = {"<unk>": 0}
    for text in corpus:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def text_to_indices(text, vocab):
    """Convert text into a list of indices based on the vocabulary."""
    return [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]


def test(model, X, y):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device

    X_test_post_list = X["numerical_post"].tolist()
    y_test_list = y.tolist()

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for index in range(len(X)):
            input_tensor = torch.tensor(X_test_post_list[index], dtype=torch.long).unsqueeze(0).to(device)
            label_tensor = torch.tensor([y_test_list[index]], dtype=torch.long).to(device)
            hidden = model.init_hidden().to(device)

            outputs = []
            for word in input_tensor[0]:
                word_tensor = word.unsqueeze(0).to(device)
                output, hidden = model(word_tensor, hidden)
                outputs.append(output)

            output = torch.stack(outputs).mean(dim=0)
            predicted_label = output.argmax(dim=1).item()  # Get the predicted class

            if predicted_label == label_tensor.item():
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    # Hyperparameters
    embedding_dim = 50
    hidden_size = 128
    output_size = 3
    learning_rate = 0.05
    num_epochs = 10

    df = pd.read_csv('./data/cleaned.csv')
    df = df.sample(2000, random_state=42)
    print(df.describe())

    # Build Vocabulary
    vocab = build_vocab(df["processed_post"])

    df["numerical_post"] = df["processed_post"].apply(lambda x: text_to_indices(x, vocab))

    # Instantiate the model, loss function, and optimizer
    vocab_size = len(vocab)
    model = RNN(vocab_size, embedding_dim, hidden_size, output_size)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Move it to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df[['numerical_post', 'nr_of_words', 'nr_of_characters']],
                                                        df['label'], test_size=0.2, random_state=42)

    train(model, X_train, y_train)

    # Load the model weights
    # model.load_state_dict(torch.load('rnn_model_2.pth'))
    test(model, X_test, y_test)