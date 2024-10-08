import cv2
import matplotlib.pyplot as plt

src1 = cv2.imread('images/img.png', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('images/mask.png', cv2.IMREAD_GRAYSCALE)
src2 = cv2.resize(src2, (src1.shape[1], src1.shape[0]))

if src1 is None or src2 is None:
    print('Image load failed!')
    exit()

dst1 = cv2.add(src1, src2)
dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0)
dst3 = cv2.subtract(src1, src2)
dst4 = cv2.absdiff(src1, src2)

plt.subplot(231), plt.axis('off'), plt.imshow(src1, cmap='gray'), plt.title('src1')
plt.subplot(232), plt.axis('off'), plt.imshow(src2, cmap='gray'), plt.title('src2')
plt.subplot(233), plt.axis('off'), plt.imshow(dst1, cmap='gray'), plt.title('dst1')
plt.subplot(234), plt.axis('off'), plt.imshow(dst2, cmap='gray'), plt.title('dst2')
plt.subplot(235), plt.axis('off'), plt.imshow(dst3, cmap='gray'), plt.title('dst3')
plt.subplot(236), plt.axis('off'), plt.imshow(dst4, cmap='gray'), plt.title('dst4')
plt.show()