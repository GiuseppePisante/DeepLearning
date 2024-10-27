from pattern import Circle
import numpy as np  
import matplotlib.pyplot as plt

def main():
    c = Circle(1024, 200,(512,256))
    res=c.draw()
    c.show()
    reference_img = np.load('reference_arrays/circle.npy')
    
    res[:]=0
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 res, c.output, "draw() did not return a copy!")
    
    plt.imshow(reference_img, cmap="gray")
    plt.title("Circle Pattern")
    plt.show(block=True)

if __name__ == "__main__":
    main()