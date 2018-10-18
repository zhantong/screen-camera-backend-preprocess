import cv2
import numpy as np
import os.path


def video_to_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    count = 0
    dir_name = os.path.join(os.path.dirname(video_path), os.path.splitext(video_path)[0])
    os.makedirs(dir_name, exist_ok=True)
    while success:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, gray3 = cv2.threshold(gray, 200, 255, 0)
        image, contours, hierarchy = cv2.findContours(gray3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        dst = cv2.bitwise_and(gray, gray, mask=mask)

        dst[dst == 0] = 255

        cv2.imwrite(os.path.join(dir_name, "frame%04d.png" % count), dst)
        success, image = vidcap.read()
        count += 1
