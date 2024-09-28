# src/model.py

import torch.nn as nn
import torch.nn.functional as func

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
