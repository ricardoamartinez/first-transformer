# app.py

import streamlit as st
import threading
from queue import Queue, Empty
from streamlit_autorefresh import st_autorefresh  # Ensure this package is installed
import logging

from src.training import train_model  # Ensure this module exists and is correctly implemented
from src.utils import reduce_dimensions  # Ensure this module exists and is correctly implemented
import plotly.express as px
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(page_title="🚀 Real-Time Token Embedding Visualization", layout="wide")

# ---------------------------
# Initialize Session State
# ---------------------------
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'gradients' not in st.session_state:
    st.session_state.gradients = None
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'update_queue' not in st.session_state:
    st.session_state.update_queue = Queue()
if 'bpe_progress' not in st.session_state:
    st.session_state.bpe_progress = 0
if 'bpe_merges' not in st.session_state:
    st.session_state.bpe_merges = []
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# ---------------------------
# Define Visualization Function
# ---------------------------
def visualize_bpe_merges(merges):
    if not merges:
        st.write("No BPE merges to display yet.")
        return

    G = nx.Graph()
    for merge in merges:
        pair = merge['merged_pair']
        new_token = merge['new_token']
        G.add_node(new_token, color='red')  # Highlight new merged tokens
        if not G.has_node(pair[0]):
            G.add_node(pair[0], color='blue')
        if not G.has_node(pair[1]):
            G.add_node(pair[1], color='blue')
        G.add_edge(pair[0], new_token, weight=1)
        G.add_edge(pair[1], new_token, weight=1)

    net = Network(height='400px', width='100%', notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        components.html(open(tmp_file.name).read(), height=400)

# ---------------------------
# Function to Start Training
# ---------------------------
def start_training_thread(text_path, window, embedding_size, epochs, batch_size, lr, update_queue):
    training_thread = threading.Thread(
        target=train_model,
        args=(text_path, window, embedding_size, epochs, batch_size, lr, update_queue),
        daemon=True  # Ensures thread exits when main program does
    )
    training_thread.start()
    st.session_state.training_thread = training_thread
    logger.info("Training thread started.")
    print("Training thread started.")

# ---------------------------
# Function to Process Queue Messages
# ---------------------------
def process_queue():
    while True:
        try:
            message = st.session_state.update_queue.get_nowait()
        except Empty:
            break
        else:
            if message['type'] == 'status':
                st.session_state.status_messages.append(message['message'])
                logger.info(f"Processed status message: {message['message']}")
                print(f"Processed status message: {message['message']}")
                # Update BPE progress based on status messages
                if "BPE Iteration" in message['message']:
                    try:
                        # Extract iteration number from message
                        iteration_num = int(message['message'].split(":")[0].split()[-1])
                        st.session_state.bpe_progress = iteration_num
                    except:
                        pass
            elif message['type'] == 'update':
                st.session_state.embeddings = message['embeddings']
                st.session_state.gradients = message['gradients']
                st.session_state.current_epoch = message['epoch']
                logger.info(f"Processed update message for epoch {message['epoch']}")
                print(f"Processed update message for epoch {message['epoch']}")
            elif message['type'] == 'bpe_merge':
                st.session_state.bpe_merges.append(message)
                logger.info(f"BPE Merge: {message['merged_pair']} -> {message['new_token']}")
                print(f"BPE Merge: {message['merged_pair']} -> {message['new_token']}")
            elif message['type'] == 'error':
                st.session_state.status_messages.append(message['message'])
                st.session_state.status_messages.append("Training failed.")
                st.session_state.training_complete = True
                logger.error(f"Processed error message: {message['message']}")
                print(f"Processed error message: {message['message']}")
                st.stop()  # Halt further execution
            elif message['type'] == 'complete':
                st.session_state.status_messages.append(message['message'])
                st.session_state.training_complete = True
                logger.info(f"Processed complete message: {message['message']}")
                print(f"Processed complete message: {message['message']}")
            elif message['type'] == 'progress':
                st.session_state.progress = float(message['message'].split(":")[-1].replace("%", ""))
                progress_bar.progress(st.session_state.progress / 100)

# ---------------------------
# Auto-Refresh Setup
# ---------------------------
# Refresh the app every 2 seconds to process new queue messages
if not st.session_state.training_complete:
    st_autorefresh(interval=2000, limit=None, key="autorefresh")

# ---------------------------
# Streamlit Layout
# ---------------------------
st.title("🚀 Real-Time Token Embedding Visualization")

# Sidebar for Configuration
st.sidebar.header("Configuration")
text_path = st.sidebar.text_input("Path to Text File", "data/test.txt")
window = st.sidebar.slider("Window Size (Context Tokens per Side)", min_value=1, max_value=10, value=1, step=1)
embedding_size = st.sidebar.slider("Embedding Size", min_value=10, max_value=300, value=10, step=10)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=100, value=1, step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=64, max_value=8192, value=64, step=64)
lr = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=0.001, step=1e-4, format="%.5f")

start_training = st.sidebar.button("Start Training")

# Visualization Placeholders
embedding_plot_placeholder = st.empty()
gradient_plot_placeholder = st.empty()

# Status Placeholder
status_placeholder = st.empty()

# Log Messages Placeholder
log_placeholder = st.empty()

# Start Training Button Logic
if start_training:
    if not st.session_state.training_complete and st.session_state.training_thread is None:
        start_training_thread(text_path, window, embedding_size, epochs, batch_size, lr, st.session_state.update_queue)
        st.session_state.status_messages.append("Training started...")
    else:
        st.warning("Training is already in progress or has completed.")

# Process any new messages from the training thread
process_queue()

# Update Status Messages
with log_placeholder.container():
    st.markdown("### Status Logs:")
    for msg in st.session_state.status_messages:
        st.write(f"- {msg}")

# BPE Progress Indicator
st.subheader("🛠️ Byte Pair Encoding (BPE) Progress")
max_merges = 100  # Ensure this matches the `max_merges` in tokenizer.py
progress_percentage = min(st.session_state.bpe_progress / max_merges, 1.0)
progress_bar = st.progress(progress_percentage)
st.write(f"**BPE Iteration:** {st.session_state.bpe_progress}/{max_merges}")

# Visualization Section
if st.session_state.embeddings and len(st.session_state.embeddings['x']) > 0:
    st.subheader("🔍 3D t-SNE Embeddings")
    fig = px.scatter_3d(
        x=st.session_state.embeddings['x'],
        y=st.session_state.embeddings['y'],
        z=st.session_state.embeddings['z'],
        text=st.session_state.embeddings['labels'],
        labels={'x': 'TSNE 1', 'y': 'TSNE 2', 'z': 'TSNE 3'},
        title=f"Epoch {st.session_state.current_epoch}",
        width=800,
        height=600
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
    embedding_plot_placeholder.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No embeddings available for visualization.")

if st.session_state.gradients is not None and len(st.session_state.gradients) > 0:
    st.subheader("📈 Gradient Magnitude")
    grad_magnitude = np.linalg.norm(st.session_state.gradients)
    fig_grad = px.scatter_3d(
        x=[0],  # Dummy coordinates
        y=[0],
        z=[0],
        marker=dict(
            size=grad_magnitude * 10,  # Scale size based on magnitude
            color='red',
            opacity=0.6
        ),
        title=f"Gradient Magnitude: {grad_magnitude:.2f}"
    )
    fig_grad.update_layout(showlegend=False)
    gradient_plot_placeholder.plotly_chart(fig_grad, use_container_width=True)
else:
    st.warning("No gradient data available.")

# BPE Merge Visualization
with st.expander("📈 BPE Merge Visualization"):
    if st.session_state.bpe_merges:
        visualize_bpe_merges(st.session_state.bpe_merges)
    else:
        st.write("No BPE merges to display yet.")

# Training Progress Information
if st.session_state.training_complete:
    st.success("🎉 Training Completed!")
else:
    if st.session_state.current_epoch > 0:
        st.info(f"🛠️ Training in progress: Epoch {st.session_state.current_epoch}/{epochs}")
    else:
        st.info("⚙️ Awaiting training to start.")

# Optional: Display Raw Embeddings and Gradients (for debugging)
with st.expander("📂 Show Raw Data"):
    if st.session_state.embeddings:
        st.write("### Embeddings (First 10 Samples)")
        st.write("#### X Coordinates")
        st.write(st.session_state.embeddings['x'][:10])
        st.write("#### Y Coordinates")
        st.write(st.session_state.embeddings['y'][:10])
        st.write("#### Z Coordinates")
        st.write(st.session_state.embeddings['z'][:10])
    if st.session_state.gradients is not None:
        st.write("### Gradients (First 10 Values)")
        st.write(st.session_state.gradients[:10])

# Debugging Section (Optional)
with st.expander("🔍 Debugging Information"):
    queue_size = st.session_state.update_queue.qsize()
    st.write(f"**Current Queue Size:** {queue_size}")
    training_thread_alive = st.session_state.training_thread.is_alive() if st.session_state.training_thread else 'No Thread Started'
    st.write(f"**Training Thread Alive:** {training_thread_alive}")
