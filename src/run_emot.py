import sys
import cv2
import numpy as np

# ...

frame_bytes = sys.stdin.read()
frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

cv2.imshow("Frame", frame)
cv2.waitKey(1)
