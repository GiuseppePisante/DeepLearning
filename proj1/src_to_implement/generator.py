import os.path
import json
import scipy.misc
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.labels = self._load_labels()
        
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, f))]
        if self.shuffle:
            np.random.shuffle(self.image_files)
        self.index = 0
        self.epoch = 0
    # Load labels
    def _load_labels(self):
        with open(self.label_path, 'r') as file:
            labels = json.load(file)
        return labels

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        batch_images = []
        batch_labels = []

        for _ in range(self.batch_size):
            
            # Shuffle if self.shuffle is true + every time we move to a new epoch
            if self.shuffle and self.index == 0:
                np.random.shuffle(self.image_files)
            if self.index >= len(self.image_files):
                self.index = 0
                self.epoch += 1

            image_file = self.image_files[self.index]
            image_path = os.path.join(self.file_path, image_file)
            image = np.load(image_path)
            image_key = os.path.splitext(image_file)[0]  # Strip the .npy extension
            

            # Resize the image using PIL
            image = Image.fromarray(image)
            image = image.resize(self.image_size[:2], Image.Resampling.LANCZOS)
            image = np.array(image)

            # Apply mirroring
            if self.mirroring:
                if np.random.rand() > 0.5:
                    image = np.fliplr(image)
                if np.random.rand() > 0.5:
                    image = np.flipud(image)

             # Apply rotation
            if self.rotation & self.index == 0:
                angle = np.random.choice([90, 180, 270])
                image = np.array(Image.fromarray(image).rotate(angle))

            batch_images.append(image)
            batch_labels.append(self.labels[image_key])
            self.index += 1

        #return images, labels
        return np.array(batch_images), np.array(batch_labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
            
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, label):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict.get(self.labels[label], "Unknown")
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        # call next method
        batch_images, batch_labels = self.next() 

        # get the number of images
        num = len(batch_images)

        # create the subplots
        fig, axes = plt.subplots(1, num, figsize = (15, 5))

        # plot i-th image with its label
        for i in range(num):
            axes[i].imshow(batch_images[i])
            axes[i].set_title(self.class_name(batch_labels[i]))
            axes[i].axis('off')
        
        plt.show()

        pass

