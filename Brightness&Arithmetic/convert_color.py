import cv2 as cv

img1 = cv.imread("images/img.png", cv.IMREAD_COLOR)
if img1 is None:
    print("Image load failed!")
    exit()

img2 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

cv.imshow("img1", img1)
cv.imshow("img2", img2)
cv.waitKey()
cv.destroyAllWindows()
