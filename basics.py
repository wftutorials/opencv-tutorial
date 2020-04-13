import cv2

image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(gray_image, (300, 300))
blurimg = cv2.GaussianBlur(small_image, (7, 7), 0)
cv2.imshow("dog", small_image)
cv2.imshow("blurred dog", blurimg)
cv2.waitKey(0)