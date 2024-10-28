import numpy as np
import matplotlib.pyplot as plt

class Checker():
    def __init__(self, resolution, tile_size):
        # Set class attributes
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None
        
    def draw(self):
        # Build tile_size x tile_size matrices of ones and zeros
        black = np.zeros((self.tile_size, self.tile_size), dtype=int)
        white = np.ones((self.tile_size, self.tile_size), dtype=int)

        # Construct the odd and even rows
        rowpattern1 = np.tile(np.hstack([black, white]), (1, self.resolution // (2 * self.tile_size)))
        rowpattern2 = np.tile(np.hstack([white, black]), (1, self.resolution // (2 * self.tile_size)))

        # Combine odd and even pattern together into the output
        self.output = np.tile(np.vstack([rowpattern1, rowpattern2]), (self.resolution // (2 * self.tile_size), 1))
        return self.output.copy()
        
    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.title("Checker Pattern")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=True)  


class Circle:
    def __init__(self, resolution=100, radius=20, position=(0, 0)):
        # Set class attributes
        self.radius = radius
        self.resolution = resolution
        self.position = position
        self.output = None

    def draw(self):
        # Initialize the output to a zero matrix
        self.output = np.zeros((self.resolution, self.resolution), dtype=int)

        # Create a grid of (x, y) coordinates
        x, y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))

        # Calculate the distance from the center
        cx, cy = self.position[0], self.position[1]
        distance = (x - cx) ** 2 + (y - cy) ** 2

        # Set pixels within the circle to 1
        self.output[distance <= self.radius**2] = 1
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.title("Circle Pattern")
        plt.show(block=True)


class Spectrum:
    def __init__(self, resolution=100):
        # Set class attribute
        self.resolution = resolution

    def draw(self):

        # Build and initialize the three channels
        self.r = np.linspace(0, 1, self.resolution)
        self.r = np.tile(self.r, (self.resolution, 1))
        self.g = np.linspace(0, 1, self.resolution)
        self.g = np.tile(self.g, (self.resolution, 1)).T
        self.b = np.linspace(1, 0, self.resolution)
        self.b = np.tile(self.b, (self.resolution, 1))

        # Assemble the output
        self.output=np.stack((self.r, self.g, self.b), axis=-1)
        return self.output.copy()

    def show(self):
        plt.imshow(np.stack((self.r, self.g, self.b), axis=-1))
        plt.title("Spectrum Pattern")
        plt.axis("equal")
        plt.show(block=True)
