import cv2

img1 = cv2.imread("images.jpeg")
print(img1.shape)

cv2.imshow("The original image", img1)
cv2.waitKey(0)