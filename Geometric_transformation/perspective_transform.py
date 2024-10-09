import cv2
import numpy as np

def on_mouse(event, x, y, flags, param):
    """
    마우스 처리 콜백 함수
    """
    global cnt, src_pts
    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽 클릭했을 때
        if cnt < 4: # 총 4개의 포인트 지정
            src_pts[cnt, :] = np.array([x, y], np.float32)
            """
            4개의 점을 저장하는 src_pts 배열 (x = 4)
            2차원 좌표이므로 (y = 2)
            여기서 src_pts의 콜론은 모든 열을 선택한다는 의미로, cnt번째 행의 모든 열을 선택한다는 것
            따라서 cnt번째 점의 좌표가 저장된 배열을 가져온다는 뜻
            """
            cnt += 1

            cv2.circle(src, (x, y), 1, (255, 0, 0), -1) # 지름 1짜리 파란색 원으로 찍은곳 표시
            cv2.imshow('src', src)

    if cnt == 4:
        w = 200 # 결과 이미지 가로 크기
        h = 300 # 결과 이미지 세로 크기

        dst_pts = np.array( # 변환 후 이미지 매핑 좌표 (사각형 네개의 꼭짓점으로 지정)
            [[0, 0],
             [w - 1, 0],
             [w -1, h - 1],
             [0, h - 1]],
            dtype=np.float32
        )

        perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts) # 투시 행렬 구하기
        dst = cv2.warpPerspective(src, perspective_matrix, (w, h)) # 투시 변환 연산
        cv2.imshow("dst", dst)

cnt = 0
src_pts = np.zeros((4, 2), np.float32)
src = cv2.imread('images/cards.png', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed!')
    exit()

cv2.namedWindow("src")
cv2.setMouseCallback("src", on_mouse)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()