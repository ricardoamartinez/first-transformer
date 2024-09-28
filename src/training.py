# src/training.py

import os
import time
import traceback
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.tokenizer import Tokenize  # Ensure this module exists and is correctly implemented
from src.model import TokenEmbeddings  # Ensure this module exists and is correctly implemented
from src.utils import skip_gram, get_embeddings, get_gradients, reduce_dimensions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    text_path,
    window,
    embedding_size,
    epochs,
    batch_size,
    lr,
    update_queue  # Queue for communication
):
    try:
        logger.info("Starting training process.")
        window = window
        embedding_size = embedding_size
        epochs = epochs
        batch_size = batch_size

        # Tokenization
        tokenizer = Tokenize(update_queue)
        if not os.path.exists(text_path):
            error_msg = f"Error: '{text_path}' not found."
            logger.error(error_msg)
            update_queue.put({'type': 'error', 'message': error_msg})
            print("Error message sent to queue.")
            return
        with open(text_path, 'r', encoding='utf-8') as f:
            text = ''.join(f.readlines())
        logger.info("Starting BPE encoding.")
        update_queue.put({'type': 'status', 'message': "Starting BPE encoding..."})
        print("BPE encoding started.")
        tokenizer.bp_encoding(text)
        tokens = tokenizer.encode(text)
        logger.info("BPE encoding completed.")
        update_queue.put({'type': 'status', 'message': "BPE encoding completed."})
        print("BPE encoding completed and message sent to queue.")

        # Device Configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        status_msg = f"Using device: {device}"
        logger.info(status_msg)
        update_queue.put({'type': 'status', 'message': status_msg})
        print("Device information sent to queue.")

        # Prepare the skip-grams as a tensor dataset
        skip_grams_pairs = skip_gram(tokens, window)
        if not skip_grams_pairs:
            error_msg = "Error: No skip-gram pairs generated."
            logger.error(error_msg)
            update_queue.put({'type': 'error', 'message': error_msg})
            print("Error message sent to queue due to no skip-gram pairs.")
            return
        contexts, targets = zip(*skip_grams_pairs)
        contexts = torch.tensor(contexts, dtype=torch.long).to(device)
        targets = torch.tensor(targets, dtype=torch.long).to(device)

        # Create a DataLoader for batching
        dataset = TensorDataset(contexts, targets)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = TokenEmbeddings(tokenizer.vocab_size, embedding_size)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            status_msg = f"Training Epoch {epoch + 1}/{epochs}..."
            logger.info(status_msg)
            update_queue.put({'type': 'status', 'message': status_msg})
            print(f"Training epoch {epoch + 1} started and message sent to queue.")
            total_loss = 0
            for batch_contexts, batch_targets in data_loader:
                optimizer.zero_grad()
                outputs = model(batch_contexts)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            status_msg = f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}"
            logger.info(status_msg)
            update_queue.put({'type': 'status', 'message': status_msg})
            print(f"Training epoch {epoch + 1} completed and message sent to queue.")

            # Get embeddings and gradients
            embeddings = get_embeddings(model)
            if embeddings.size == 0:
                error_msg = "Error: Embeddings are empty."
                logger.error(error_msg)
                update_queue.put({'type': 'error', 'message': error_msg})
                print("Error message sent to queue due to empty embeddings.")
                return
            gradients = get_gradients(model)
            if gradients.size == 0:
                error_msg = "Error: Gradients are empty."
                logger.error(error_msg)
                update_queue.put({'type': 'error', 'message': error_msg})
                print("Error message sent to queue due to empty gradients.")
                return

            # Apply t-SNE reduction for visualization
            try:
                if len(embeddings) > 500:
                    sample_size = 500
                    indices = torch.randint(0, len(embeddings), (sample_size,)).tolist()
                    sampled_embeddings = embeddings[indices]
                    labels = [tokenizer.mapping[i] for i in indices]
                else:
                    sampled_embeddings = embeddings
                    labels = [tokenizer.mapping[i] for i in range(len(embeddings))]

                reduced_embeddings = reduce_dimensions(sampled_embeddings)
                if reduced_embeddings.size == 0:
                    raise ValueError("Dimensionality reduction failed, resulting in empty embeddings.")

            except Exception as e:
                error_msg = f"Error during dimensionality reduction: {str(e)}"
                logger.error(error_msg)
                update_queue.put({'type': 'error', 'message': error_msg})
                print("Error message sent to queue due to dimensionality reduction failure.")
                return

            # Prepare data to send
            embed_data = {
                'x': reduced_embeddings[:, 0].tolist(),
                'y': reduced_embeddings[:, 1].tolist(),
                'z': reduced_embeddings[:, 2].tolist(),
                'labels': labels
            }

            grad_data = gradients.tolist()

            # Put the updated data into the queue
            update_queue.put({
                'type': 'update',
                'embeddings': embed_data,
                'gradients': grad_data,
                'epoch': epoch + 1
            })
            print(f"Update message for epoch {epoch + 1} sent to queue.")

            # Small sleep to allow Streamlit to process updates
            time.sleep(0.1)

        # After training is done
        complete_msg = 'Training completed!'
        logger.info(complete_msg)
        update_queue.put({'type': 'complete', 'message': complete_msg})
        print("Complete message sent to queue.")

    except Exception as e:
        error_message = f"An exception occurred during training: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        update_queue.put({'type': 'error', 'message': error_message})
        print("Exception message sent to queue.")
