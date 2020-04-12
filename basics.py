import cv2
from matplotlib import pyplot as plt

image = cv2.imread('dog.png')
print(image)
#cv2.imwrite("test.jpg", image)
#plt.imshow(image)
#plt.show()
#cv2.imshow("dog", image)
#cv2.waitKey(0)