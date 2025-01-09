import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm


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
            # print(input_tensor.shape)
            optimizer.zero_grad()

            outputs = []
            for word in input_tensor[0]:
                word_tensor = word.unsqueeze(0).to(device)
                # print(word_tensor.shape, hidden.shape)
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

def build_vocabulary(df1, df2):
    # Build Vocabulary
    df = pd.concat([df1, df2])
    vocab = build_vocab(df["processed_post"])
    df1["numerical_post"] = df1["processed_post"].apply(lambda x: text_to_indices(x, vocab))
    df2["numerical_post"] = df2["processed_post"].apply(lambda x: text_to_indices(x, vocab))
    return df1, df2, len(vocab)

def test(model, X, y):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device

    X_test_post_list = X["numerical_post"].tolist()
    y_test_list = y.tolist()

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for index in tqdm(range(len(X)), desc="Testing"):
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




# Function to initialize the model and optimizer
def initialize_model_and_optimizer(vocab_size, output_size, params, device):
    model = RNN(vocab_size, params['embedding_dim'], params['hidden_size'], output_size).to(device)

    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    return model, optimizer


# Function to train the model for one epoch
def train_one_epoch(model, optimizer, criterion, X_train, y_train, params, device, number):
    # model.train()
    model.to(device)  # Move the model to the correct device
    total_loss = 0
    batch_size = params['batch_size']
    X_train_post_list = X_train["numerical_post"].tolist()
    y_train_list = y_train.tolist()

    for i in tqdm(range(0, len(X_train), batch_size),  desc=f"Epoch {number + 1}"):
        input_batch = X_train_post_list[i:i + batch_size]
        label_batch = y_train_list[i:i + batch_size]
        optimizer.zero_grad()

        input_tensor, label_tensor = process_batch(input_batch, label_batch, device)
        loss = compute_loss(model, criterion, input_tensor, label_tensor, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {number + 1}, Training Loss: {total_loss:.4f}")
    return total_loss


# Function to process a batch into tensors
def process_batch(input_batch, label_batch, device):
    input_tensor = rnn_utils.pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in input_batch],
        batch_first=True
    ).unsqueeze(0).to(device)
    label_tensor = torch.tensor(label_batch, dtype=torch.long).to(device)
    return input_tensor, label_tensor


# Function to compute loss for a batch
def compute_loss(model, criterion, input_tensor, label_tensor, device):
    hidden = model.init_hidden().to(device)  # Initialize hidden state
    _, batch_size, seq_length = input_tensor.size()
    # print(input_tensor.shape)

    num_classes = model.i2o.out_features  # Get the number of output classes

    outputs = torch.zeros(batch_size, num_classes, device=device)  # Initialize outputs for the batch

    for word in input_tensor[0]:  # Loop over the sequence length
        for t in word:
            word_tensor = t.unsqueeze(0).to(device)  # Shape: (batch_size, 1)
            # print(word_tensor.shape, hidden.shape)
            output, hidden = model(word_tensor, hidden)  # Output shape: (batch_size, num_classes)
            # print(output)
            outputs += output.squeeze(1)  # Accumulate outputs, ensuring shape is (batch_size, num_classes)

    outputs /= seq_length  # Average over the sequence dimension
    loss = criterion(outputs, label_tensor)  # Compute loss
    return loss


# Function to evaluate the model
def evaluate_model(model, X_val, y_val, device):
    model.eval()
    X_val_post_list = X_val["numerical_post"].tolist()
    y_val_list = y_val.tolist()
    all_preds = []

    with torch.no_grad():
        for index in tqdm(range(len(X_val)),  desc="Evaluating..."):
            input_tensor = torch.tensor(X_val_post_list[index], dtype=torch.long).unsqueeze(0).to(device)
            hidden = model.init_hidden().to(device)

            outputs = []
            for word in input_tensor[0]:
                output, hidden = model(word.unsqueeze(0).to(device), hidden)
                outputs.append(output)

            output = torch.stack(outputs).mean(dim=0)
            predicted_label = output.argmax(dim=1).item()
            all_preds.append(predicted_label)

    accuracy = accuracy_score(y_val_list, all_preds)
    return accuracy


# Main function to train and evaluate with given parameters
def train_and_evaluate(params, X_train, y_train, X_val, y_val, vocab_size, output_size, device):
    model, optimizer = initialize_model_and_optimizer(vocab_size, output_size, params, device)
    criterion = nn.NLLLoss()
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, criterion, X_train, y_train, params, device, epoch)

    val_accuracy = evaluate_model(model, X_val, y_val, device)
    return val_accuracy


# Perform random search
def perform_random_search(param_space, num_samples, X_train, y_train, X_val, y_val, vocab_size, output_size, device):
    best_params = None
    best_accuracy = 0

    for _ in range(num_samples):
        sampled_params = {key: random.choice(values) for key, values in param_space.items()}
        print(f"Testing parameters: {sampled_params}")

        val_accuracy = train_and_evaluate(
            sampled_params, X_train, y_train, X_val, y_val, vocab_size, output_size, device
        )
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = sampled_params

    print(f"Best Parameters: {best_params}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")





if __name__ == '__main__':
    output_size = 3
    label_column = 'label'
    mode = 'hypertune'

    train_df = pd.read_csv('./data/train.csv')
    train_df = train_df.sample(2000, random_state=42)
    X_train = train_df.drop(columns=[label_column])
    y_train = train_df[label_column]
    if mode == 'hypertune':
        val_df = pd.read_csv('./data/val.csv')
        X_val = val_df.drop(columns=[label_column])
        y_val = val_df[label_column]

        # Hyperparameters
        param_space = {
            'embedding_dim': [50, 100, 200],
            'hidden_size': [64, 128, 256],
            'learning_rate': [0.01, 0.05, 0.1],
            'optimizer': ['SGD', 'Adam'],
            'batch_size': [64, 128, 256]
        }
        num_samples = 10
        num_epochs = 3

        # Load your data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        X_train, X_val, vocab_size = build_vocabulary(X_train, X_val)
        perform_random_search(param_space, num_samples, X_train, y_train, X_val, y_val, vocab_size, output_size, device)
    else:
        # Hyperparameters
        embedding_dim = 50
        hidden_size = 128
        learning_rate = 0.05
        num_epochs = 10

        test_df = pd.read_csv('./data/test.csv')
        X_test = test_df.drop(columns=[label_column])
        y_test = test_df[label_column]

        X_train, X_test, vocab_size = build_vocabulary(X_train, X_test)
        model = RNN(vocab_size, embedding_dim, hidden_size, output_size)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Move it to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        train(model, X_train, y_train)

        # Load the model weights
        # model.load_state_dict(torch.load('rnn_model_2.pth'))
        test(model, X_test, y_test)



