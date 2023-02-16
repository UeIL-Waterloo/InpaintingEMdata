import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dictlearn as dl


inpainter = dl.Inpaint('Images/test_image.tif', 'Images/test_mask_50.png')
inpainted_image = inpainter.train().inpaint()

plt.subplot(121)
plt.imshow(inpainter.patches.image)
plt.title('Original')

plt.subplot(122)
plt.imshow(inpainted_image)
plt.title('Inpainted')

plt.show()