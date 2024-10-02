import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_voxel_grid(grid: np.ndarray, title: str = "Voxel Grid"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Explicitly cast ax to Axes3D to avoid type checker errors
    ax = plt.gca()

    ax.voxels(grid, edgecolor='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.show()

def load_voxel_grids(directory: str):
    voxel_grids = []
    filenames = []

    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            filepath = os.path.join(directory, filename)
            grid = np.load(filepath)
            voxel_grids.append(grid)
            filenames.append(filename)

    return voxel_grids, filenames

def main():
    dataset_directory = "dataset"

    # Load voxel grids from the dataset directory
    voxel_grids, filenames = load_voxel_grids(dataset_directory)

    for i, (grid, filename) in enumerate(zip(voxel_grids, filenames)):
        print(f"Visualizing {filename}")
        plot_voxel_grid(grid, title=f"Sample {i}: {filename}")

if __name__ == "__main__":
    main()
