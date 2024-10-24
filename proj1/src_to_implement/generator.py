import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

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
        # Load labels
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, f))]
        if self.shuffle:
            np.random.shuffle(self.image_files)
        self.index = 0
        self.epoch = 0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        batch_images = []
        batch_labels = []

        for _ in range(self.batch_size):
            if self.index >= len(self.image_files):
                self.index = 0
                self.epoch += 1
                if self.shuffle:
                    np.random.shuffle(self.image_files)

            image_file = self.image_files[self.index]
            image_path = os.path.join(self.file_path, image_file)
            image = scipy.misc.imread(image_path)
            image = scipy.misc.imresize(image, self.image_size[:2])

            if self.rotation:
                image = np.rot90(image)
            if self.mirroring:
                image = np.fliplr(image)
                batch_images.append(image)
            batch_labels.append(self.labels[image_file])

            self.index += 1

        # Ensure the batch has the correct size
        while len(batch_images) < self.batch_size:
            image_file = self.image_files[self.index]
            image_path = os.path.join(self.file_path, image_file)
            image = scipy.misc.imread(image_path)
            image = scipy.misc.imresize(image, self.image_size[:2])

            if self.rotation:
                image = np.rot90(image)
            if self.mirroring:
                image = np.fliplr(image)

            batch_images.append(image)
            batch_labels.append(self.labels[image_file])

            self.index += 1
            if self.index >= len(self.image_files):
                self.index = 0
                if self.shuffle:
                    np.random.shuffle(self.image_files)
        #return images, labels
        return np.array(batch_images), np.array(batch_labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        # Apply mirroring
        if self.mirroring and np.random.rand() > 0.5:
            img = np.fliplr(img)

        # Apply rotation
        if self.rotation:
            angle = np.random.choice([90, 180, 270])
            img = scipy.misc.imrotate(img, angle)
            
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, image):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict.get(self.labels[image], "Unknown")
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        pass

