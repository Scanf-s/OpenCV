import cv2

src = cv2.imread('images/mnms.png', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed!')
    exit()

bgr_planes = cv2.split(src)

cv2.imshow('src', src)
cv2.imshow('b', bgr_planes[0])
cv2.imshow('g', bgr_planes[1])
cv2.imshow('r', bgr_planes[2])
cv2.waitKey()
cv2.destroyAllWindows()