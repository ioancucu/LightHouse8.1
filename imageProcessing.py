import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image
image = cv2.imread("C:\\Users\\cucui\\Downloads\\archive (3)\\trainingSet\\trainingSet\\4\\img_122.jpg", cv2.IMREAD_GRAYSCALE)
image_float = image.astype(np.float32)

normalized_image = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float))
plt.imshow(normalized_image)
plt.axis("off")  # Remove axis ticks and labels
plt.show()

print(normalized_image.flatten())

# Iterate over each pixel
for row in range(image.shape[0]):
    for col in range(image.shape[1]):
        # Get the pixel values
        pixel = image[row, col]