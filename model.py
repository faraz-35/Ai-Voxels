import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Output Visualization
def visualize_voxel_grid(voxel_grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of the non-zero voxels
    x, y, z = np.nonzero(voxel_grid)

    # Plot the voxels
    ax.scatter(x, y, z, zdir='z', c='red')

    plt.show()

# Model Definition
class TextToVoxelModel(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(TextToVoxelModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 512)  # First dense layer
        self.fc2 = nn.Linear(512, 4096)  # Second dense layer (4096 for 16x16x16 voxel grid)
        self.fc3 = nn.Linear(4096, output_dim)  # Final output layer to match voxel grid size
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Pass through the first layer
        x = self.relu(self.fc2(x))  # Pass through the second layer
        x = torch.sigmoid(self.fc3(x))  # Output layer with sigmoid for voxel values (0 to 1)
        x = x.view(-1, 32, 32, 32)  # Reshape to match voxel dimensions
        return x

# Load Cube Data from Dataset Directory
def load_cube_data(dataset_dir, filename):
    file_path = os.path.join(dataset_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    cube_data = np.load(file_path)
    return cube_data

# Training the Model
dataset_dir = 'dataset'
cube_filename = 'cube.npy'
cube_data = load_cube_data(dataset_dir, cube_filename)

# Create a random embedding for the cube (for simplicity)
embedding = torch.randn(1, 768)  # Assume this is the "cube" embedding
voxel_grid = torch.tensor(cube_data, dtype=torch.float32).unsqueeze(0)

model = TextToVoxelModel(embedding_dim=768, output_dim=32*32*32)
criterion = nn.MSELoss()  # Mean Squared Error for voxel grid comparison
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):  # Example: 10 epochs
    optimizer.zero_grad()  # Clear gradients
    output = model(embedding)  # Forward pass
    loss = criterion(output, voxel_grid)  # Calculate loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model weights
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the trained model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/text_to_voxel_model.pth')

# Load the trained model
model = TextToVoxelModel(embedding_dim=768, output_dim=32*32*32)
model.load_state_dict(torch.load('models/text_to_voxel_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    predicted_voxel_grid = model(embedding)
    print(predicted_voxel_grid)  # Predict the voxel grid for "cube"

    # Convert the tensor to a numpy array
    predicted_voxel_grid_np = predicted_voxel_grid.cpu().numpy().squeeze()

    # Ensure the voxel grid is 3-dimensional
    if predicted_voxel_grid_np.ndim != 3:
        raise ValueError("Predicted voxel grid must be 3-dimensional")

    # Save the numpy array as a .npy file in the dataset directory
    np.save('dataset/cube_predicted.npy', predicted_voxel_grid_np)

# Load the saved .npy file
voxel_grid = np.load('dataset/cube_predicted.npy')
if voxel_grid.ndim != 3:
    raise ValueError("Loaded voxel grid must be 3-dimensional")

# Visualize the voxel grid
visualize_voxel_grid(voxel_grid)
