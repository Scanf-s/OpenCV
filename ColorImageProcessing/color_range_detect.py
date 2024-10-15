import cv2


def on_hue_changed(_=None):
    lower_h = cv2.getTrackbarPos('H_lower', 'mask')
    upper_h = cv2.getTrackbarPos('H_upper', 'mask')

    lower_b = (lower_h, 100, 0)
    upper_b = (upper_h, 255, 255)
    mask = cv2.inRange(src_hsv, lower_b, upper_b)

    cv2.imshow('mask', mask)

def main():

    global src_hsv

    src = cv2.imread("images/mnms.png", cv2.IMREAD_COLOR)
    if src is None:
        exit()

    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    cv2.imshow("src", src)

    cv2.namedWindow("mask")
    cv2.createTrackbar('H_lower', 'mask', 40, 179, on_hue_changed)
    cv2.createTrackbar('H_upper', 'mask', 80, 179, on_hue_changed)
    on_hue_changed(0)

    cv2.waitKey()
    cv2.destroyAllWindows()

main()