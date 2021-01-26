from skimage import io
import matplotlib.pyplot as plt

origImage = io.imread('./images/gigi.jpg')
(height, width, channels) = origImage.shape
startCropX = width // 2
startCropY = height // 2
croppedImage = origImage[startCropY:, startCropX:]

plt.imshow(croppedImage)
plt.show()