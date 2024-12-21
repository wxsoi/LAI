import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


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


# Convert text to numerical data
def text_pipeline(text):
    return vocab(tokenizer(text))


if __name__ == '__main__':
    torchtext.disable_torchtext_deprecation_warning()

    # Hyperparameters
    embedding_dim = 50
    hidden_size = 128
    output_size = 3
    learning_rate = 0.01
    num_epochs = 10

    df = pd.read_csv('./data/clean.csv')
    df = df.head(100)

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    tokenizer = get_tokenizer("basic_english")

    # Build Vocabulary
    vocab = build_vocab_from_iterator(df["clean_post"], specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    df["numerical_post"] = df["clean_post"].apply(text_pipeline)

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

    # Training loop
    X_train_post_list = X_train["numerical_post"].tolist()
    y_train_list = y_train.tolist()

    for epoch in range(num_epochs):
        total_loss = 0
        for index in range(len(X_train)):
            input_tensor = torch.tensor(X_train_post_list[index], dtype=torch.long).unsqueeze(0).to(device)
            label_tensor = torch.tensor([y_train_list[index]], dtype=torch.long).to(device)
            hidden = model.init_hidden().to(device)

            optimizer.zero_grad()

            outputs = []
            for word in input_tensor[0]:
                output, hidden = model(word.unsqueeze(0), hidden)
                outputs.append(output)

            output = torch.stack(outputs).mean(dim=0)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'rnn_model.pth')
