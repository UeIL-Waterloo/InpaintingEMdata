import sys

import matplotlib.pyplot as plt

from algorithmInpainting import *
from biharmonicInpainting import *
import os
from skimage import color, io
import tkinter as tk
from tkinter import filedialog as fd
import cv2


def mse(imageA, imageB, mask=None):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images
    if mask == 'spiral':
        imageA = circleMaskImage(imageA)
        imageB = circleMaskImage(imageB)

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar" the two images are
    return err

def pass2D(img):
    if len(img.shape) > 2:
        return color.rgb2gray(img)
    else:
        return img

class Inpaint:
    def __init__(self, fullFilePath: str, resize: float = 0):
        self.image = cv2.imread(fullFilePath)
        if resize:
            self.image = cv2.resize(self.image , (int(self.image .shape[0] * resize), int(self.image .shape[1] * resize)))
        self.fileName = os.path.splitext(os.path.basename(fullFilePath))[0]
        self.filePath = os.path.dirname(fullFilePath)

    def inpaint(self, percent: int, iType: str, mask: str, show=False):
        if iType == 'algorithm':
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            imageClass = Algorithm(image, percentInpaint=percent)
        elif iType == 'biharmonic':
            imageClass = Biharmonic(self.image, percentInpaint=percent)
        else:
            sys.exit("type given is not a valid option.")

        if mask == 'random':
            inpainted_img = imageClass.randomInpaint(show=show)
        elif mask == 'spiral':
            inpainted_img = imageClass.spiralInpaint(show=show)
        else:
            sys.exit("mask given is not a valid option.")

        self.image = pass2D(self.image)
        inpainted_img = pass2D(inpainted_img)

        meanSquardError = mse(self.image, inpainted_img, mask=mask)
        # print('percent:', percent, iType, mask, 'mse:', meanSquardError)
        return meanSquardError

class GUI:
    @staticmethod
    def select_files():
        root = tk.Tk()
        paths = fd.askopenfilenames(filetypes=[('all files', '.*'),
                                               ('text files', '.txt'),
                                               ('image files', '.png'),
                                               ('image files', '.jpg')], title="Choose file(s).")
        paths = root.tk.splitlist(paths)
        root.destroy()
        return paths

if __name__ == '__main__':
    # Import file with tkinter selection.
    fullFilePaths = GUI.select_files()

    for file in fullFilePaths:
        percent = 20
        resize = 0.5
        resize_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 1]
        #resize_list = [0.1]
        algo_rand = []
        algo_spir = []
        biha_rand = []
        biha_spir = []
        for i in resize_list:
            print(i)
            algo_rand.append(Inpaint(file, resize=i).inpaint(percent, 'algorithm', 'random', show=False))
            algo_spir.append(Inpaint(file, resize=i).inpaint(percent, 'algorithm', 'spiral', show=False))
            biha_rand.append(Inpaint(file, resize=i).inpaint(percent, 'biharmonic', 'random', show=False))
            biha_spir.append(Inpaint(file, resize=i).inpaint(percent, 'biharmonic', 'spiral', show=False))

        plt.plot(resize_list, algo_rand, label = 'algo_rand')
        plt.plot(resize_list, algo_spir, label = 'algo_spir')
        plt.plot(resize_list, biha_rand, label = 'biha_rand')
        plt.plot(resize_list, biha_spir, label = 'biha_spir')
        plt.ylabel("Mean Squared Error")
        plt.xlabel("Resize value")
        plt.legend()
        plt.show()

