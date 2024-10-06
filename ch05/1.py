import cv2 as cv

img1 = cv.imread("images/img.png", cv.IMREAD_COLOR)
if img1 is None:
    print("Image load failed!")
    exit()

img2 = cv.cvtColor(img1)

cv.imshow("img", img1)
cv.waitKey()
cv.destroyAllWindows()
