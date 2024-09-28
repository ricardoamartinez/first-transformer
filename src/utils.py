# src/utils.py

import torch
from sklearn.manifold import TSNE
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def skip_gram(lst, window):
    skip_grams = []
    for i in range(len(lst)):
        # Define the context window boundaries
        start = max(0, i - window)
        end = min(len(lst), i + window + 1)
        for j in range(start, end):
            if i != j:
                skip_grams.append((lst[i], lst[j]))
    logger.info(f"Generated {len(skip_grams)} skip-gram pairs with window size {window}.")
    return skip_grams

def get_embeddings(model):
    embeddings = model.embedding.weight.data.cpu().numpy()
    logger.info(f"Retrieved embeddings with shape: {embeddings.shape}")
    return embeddings

def get_gradients(model):
    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads.append(param.grad.cpu().numpy().flatten())
    if grads:
        gradients = np.concatenate(grads)
        logger.info(f"Retrieved gradients with shape: {gradients.shape}")
        return gradients
    else:
        logger.warning("No gradients found.")
        return np.array([])

def reduce_dimensions(embeddings, n_components=3, random_state=0):
    try:
        tsne = TSNE(n_components=n_components, random_state=random_state)
        reduced_embeddings = tsne.fit_transform(embeddings)
        return reduced_embeddings
    except Exception as e:
        logger.error(f"Error during dimensionality reduction: {str(e)}")
        return np.array([])
