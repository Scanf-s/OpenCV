import cv2

src = cv2.imread("images/img.png", cv2.IMREAD_GRAYSCALE)
if src is None:
    exit()

canny_1 = cv2.Canny(src, 50, 100) # 두 임계값을 50, 100으로 설정
canny_2 = cv2.Canny(src, 50, 150) # 두 임계값을 50, 150으로 설정

cv2.imshow("src", src)
cv2.imshow("canny_1", canny_1)
cv2.imshow("canny_2", canny_2)
cv2.waitKey()
cv2.destroyAllWindows()