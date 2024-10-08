import cv2

src = cv2.imread("images/img.png", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    exit()

for sigma in range(1, 6):
    blurred = cv2.GaussianBlur(src, (0, 0), sigma) # unsharp 필터링을 할 때 일단 Blurring된 영상이 필요하니까

    alpha = 1.0
    # h(x, y) = (1 + alpha) * src(x, y) - alpha * blurred(x, y)
    dst = cv2.addWeighted(src, 1 + alpha, blurred, -alpha, 0)

    cv2.putText(dst, f"sigma = {sigma}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.imshow("dst", dst)
    cv2.waitKey()

cv2.destroyAllWindows()