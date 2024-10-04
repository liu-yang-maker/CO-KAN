import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
from pykan import KolmogorovArnoldNetwork  # Import the new KAN implementation from pykan

# Generate a random graph for the Max-Cut problem
def generate_graph(num_nodes, edge_prob):
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    adj_matrix = nx.to_numpy_array(G)
    return G, adj_matrix

# Loss function for Max-Cut
# The objective is to maximize the weight of edges between the two sets of nodes
# which is equivalent to minimizing the negative cut value
def maxcut_loss(adj_matrix, assignments):
    cut_value = 0
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                cut_value += adj_matrix[i, j] * assignments[i] * (1 - assignments[j])
    return -cut_value

# Training the KAN to solve Max-Cut
def train_kan_for_maxcut(adj_matrix, num_epochs=1000, learning_rate=0.01):
    num_nodes = adj_matrix.shape[0]
    model = KolmogorovArnoldNetwork(input_dim=num_nodes, hidden_dim=128, output_dim=num_nodes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        # Random input to the network
        x = torch.randn(num_nodes)
        # Forward pass to get the node assignments (0 or 1)
        assignments = model(x)
        assignments = torch.round(assignments)  # Round to 0 or 1
        
        # Compute the loss (negative cut value)
        loss = maxcut_loss(adj_matrix, assignments)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Return the final assignments
    return assignments.detach().numpy()

if __name__ == "__main__":
    # Generate a random graph with 10 nodes and edge probability of 0.5
    G, adj_matrix = generate_graph(num_nodes=10, edge_prob=0.5)
    
    # Train KAN to solve the Max-Cut problem
    assignments = train_kan_for_maxcut(adj_matrix)
    
    # Visualize the result
    color_map = ['red' if assignments[i] == 1 else 'blue' for i in range(len(assignments))]
    nx.draw(G, node_color=color_map, with_labels=True)
    import matplotlib.pyplot as plt
    plt.show()