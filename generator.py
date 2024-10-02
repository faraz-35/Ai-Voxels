import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import os

@dataclass
class Shape:
    shape_type: str
    position: Tuple[int, int, int]
    size: int

class VoxelGenerator:
    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size

    def create_empty_grid(self) -> np.ndarray:
        return np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint8)

    def add_cube(self, grid: np.ndarray, position: Tuple[int, int, int], size: int) -> np.ndarray:
        x, y, z = position
        grid[x:x+size, y:y+size, z:z+size] = 1
        return grid

    def add_sphere(self, grid: np.ndarray, position: Tuple[int, int, int], radius: int) -> np.ndarray:
        x, y, z = position
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                for dz in range(-radius, radius+1):
                    if dx**2 + dy**2 + dz**2 <= radius**2:
                        px, py, pz = x+dx, y+dy, z+dz
                        if 0 <= px < self.grid_size and 0 <= py < self.grid_size and 0 <= pz < self.grid_size:
                            grid[px, py, pz] = 1
        return grid

    def generate_random_shape(self) -> Tuple[np.ndarray, Shape]:
        grid = self.create_empty_grid()

        shape_type = np.random.choice(['cube', 'sphere'])
        size = np.random.randint(4, 10)
        position = tuple(np.random.randint(0, self.grid_size - size, size=3))

        shape = Shape(shape_type=shape_type, position=position, size=size)

        if shape_type == 'cube':
            grid = self.add_cube(grid, position, size)
        else:  # sphere
            grid = self.add_sphere(grid, position, size // 2)

        return grid, shape

    def generate_dataset(self, num_samples: int) -> Tuple[np.ndarray, List[Shape]]:
        dataset = np.zeros((num_samples, self.grid_size, self.grid_size, self.grid_size), dtype=np.uint8)
        shapes = []

        for i in range(num_samples):
            grid, shape = self.generate_random_shape()
            dataset[i] = grid
            shapes.append(shape)

        return dataset, shapes

def save_dataset(dataset: np.ndarray, shapes: List[Shape], directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, (grid, shape) in enumerate(zip(dataset, shapes)):
        filename = os.path.join(directory, f"sample_{i}_{shape.shape_type}_{shape.position}_{shape.size}.npy")
        np.save(filename, grid)


# Example usage
def main():
    generator = VoxelGenerator(grid_size=32)
    num_samples = 10
    dataset, shapes = generator.generate_dataset(num_samples)

    # Save the dataset to disk
    save_directory = "dataset"
    save_dataset(dataset, shapes, save_directory)
    print(f"Dataset saved to {save_directory}")

if __name__ == "__main__":
    main()
