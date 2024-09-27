import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import DataLoader, TensorDataset
from Tokenizer import Tokenize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
torch.manual_seed(0)

def skip_gram(lst, window):
    half = window // 2
    skip_grams = []
    for i in range(half, len(lst) - half):
        for j in range(i - half, i + half + 1):
            if i != j:
                skip_grams.append((lst[i], lst[j]))
    return skip_grams

class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer1 = nn.Linear(embedding_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, vocab_size)
    
    def forward(self, context):
        context_embedding = self.embedding(context)
        out = self.layer1(context_embedding)
        out = func.relu(out)
        out = self.layer2(out)
        out = func.relu(out)
        out = self.layer3(out)
        return out

window = 2
embedding_size = 50
epochs = 50
batch_size = 2048  # Adjust batch size as needed

print('Tokenizing...')
text = ''.join(open('test.txt', 'r').readlines())
tokenizer = Tokenize()
tokenizer.bp_encoding(text)
tokens = tokenizer.encode(text)
print('Tokenized')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Prepare the skip-grams as a tensor dataset
skip_grams = skip_gram(tokens, window)
contexts, targets = zip(*skip_grams)
contexts = torch.tensor(contexts, dtype=torch.long).to(device)
targets = torch.tensor(targets, dtype=torch.long).to(device)

# Create a DataLoader for batching
dataset = TensorDataset(contexts, targets)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TokenEmbeddings(tokenizer.vocab_size, embedding_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    print(f'Training epoch [{epoch+1}/{epochs}] :-', end='\t')
    total_loss = 0
    for batch_contexts, batch_targets in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_contexts)
        
        # Flatten the outputs and targets for loss computation
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print("Total Loss:", total_loss)

#   P   L   O   T   T   I   N   G         T   H   E         E   M   B   E   D   D   I   N   G   S
def get_embeddings(model):
    return model.embedding.weight.data.cpu().numpy()

def reduce_dimensions(embeddings):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def plot_embeddings(reduced_embeddings, tokens):
    plt.figure(figsize=(10, 10))
    for i in range(len(reduced_embeddings)):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(tokens[i], xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                     textcoords='offset points', xytext=(0, 5), ha='center', fontsize=8)
    plt.title("t-SNE of Token Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid()
    plt.show()

# After training, call the functions to plot the embeddings
embeddings = get_embeddings(model)
reduced_embeddings = reduce_dimensions(embeddings)
plot_embeddings(reduced_embeddings, tokens)