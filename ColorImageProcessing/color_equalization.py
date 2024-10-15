import cv2

src = cv2.imread("images/mnms.png", cv2.IMREAD_COLOR)

if src is None:
    print("Image load failed!")
    exit()

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

y, cr, cb = cv2.split(src_ycrcb)

y = cv2.equalizeHist(y) # Y 항목 (밝기 정보)에 대해서만 히스토그램 평활화

src_ycrcb = cv2.merge([y, cr, cb]) # 다시 결과를 원본에 합쳐준다.

dst = cv2.cvtColor(src_ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()