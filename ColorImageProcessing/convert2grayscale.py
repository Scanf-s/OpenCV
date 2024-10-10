import cv2

src = cv2.imread('images/img.png', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed!')
    exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow('src', src)
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()