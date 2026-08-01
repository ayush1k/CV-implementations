import cv2

img1 = cv2.imread("images.jpeg")
print(img1.shape)

# cv2.imshow("The original image", img1)
# cv2.waitKey(0)

cv2.imwrite("saved_original_image.jpeg", img1)

resize = cv2.resize(img1, (640, 480))
grey = cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
blurring = cv2.GaussianBlur(img1, (5,5), 0)
edges = cv2.Canny(img1, 100, 200)

cv2.imshow("Resized image", resize)
cv2.imshow("Greyish image", grey)
cv2.imshow("Blur image", blurring)
cv2.imshow("edgy image", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()