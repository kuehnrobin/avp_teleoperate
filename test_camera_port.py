import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        print(f"Camera found at /dev/video{i}")
        cap.release()
