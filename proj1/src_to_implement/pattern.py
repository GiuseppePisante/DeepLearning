import numpy as np
import matplotlib.pyplot as plt


class Circle:
    def __init__(self, radius=20, resolution=100, position=(0, 0)):
        self.radius = radius
        self.resolution = resolution
        self.position = position
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        cx, cy = self.resolution // 2, self.resolution // 2

        # Create a grid of (x, y) coordinates
        y, x = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))

        # Calculate the distance from the center
        distance = (x - cx) ** 2 + (y - cy) ** 2

        # Set pixels within the circle to 1
        self.output[distance <= self.radius**2] = 1

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.title("Circle Pattern")
        plt.axis("equal")
        plt.show(block=True)


class Spectrum:
    def __init__(self, resolution=100):
        self.resolution = resolution

    def draw(self):
        self.r = np.linspace(0, 1, self.resolution)
        self.r = np.tile(self.r, (self.resolution, 1))
        self.g = np.linspace(0, 1, self.resolution)
        self.g = np.tile(self.g, (self.resolution, 1)).T
        self.b = np.linspace(1, 0, self.resolution)
        self.b = np.tile(self.b, (self.resolution, 1))

    def show(self):
        plt.imshow(np.stack((self.r, self.g, self.b), axis=-1))
        plt.title("Spectrum Pattern")
        plt.axis("equal")
        plt.show(block=True)
