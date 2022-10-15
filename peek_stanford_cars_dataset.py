import torchvision
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('TkAgg')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols)+1, cols, i+1)
        plt.imshow(img[0])


if __name__ == "__main__":
    data = torchvision.datasets.StanfordCars(root=".", download=True)
    show_images(data)
    plt.show()
