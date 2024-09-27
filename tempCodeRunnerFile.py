model = TokenEmbeddings(tokenizer.vocab_size, embedding_size)
# model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
# criterion = nn.CrossEntropyLoss()

# for epoch in range(epochs):
#     print(f'Training epoch [{epoch+1}/{epochs}] :-', end='\t')
#     total_loss = 0
#     for batch_contexts, batch_targets in data_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_contexts)
        
#         # Flatten the outputs and targets for loss computation
#         loss = criterion(outputs, batch_targets)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
#     print("Total Loss:", total_loss)

# #   P   L   O   T   T   I   N   G         T   H   E         E   M   B   E   D   D   I   N   G   S
# def get_embeddings(model):
#     return model.embedding.weight.data.cpu().numpy()

# def reduce_dimensions(embeddings):
#     tsne = TSNE(n_components=2, random_state=0)
#     reduced_embeddings = tsne.fit_transform(embeddings)
#     return reduced_embeddings

# def plot_embeddings(reduced_embeddings, tokens):
#     plt.figure(figsize=(10, 10))
#     for i in range(len(reduced_embeddings)):
#         plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
#         plt.annotate(tokens[i], xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
#                      textcoords='offset points', xytext=(0, 5), ha='center', fontsize=8)
#     plt.title("t-SNE of Token Embeddings")
#     plt.xlabel("Dimension 1")
#     plt.ylabel("Dimension 2")
#     plt.grid()
#     plt.show()

# # After training, call the functions to plot the embeddings
# embeddings = get_embeddings(model)
# reduced_embeddings = reduce_dimensions(embeddings)
# plot_embeddings(reduced_embeddings, tokens)
