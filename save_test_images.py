import cv2
import numpy as np

camera=cv2.VideoCapture(0)
kernel = np.ones((14,14),np.uint8)
image_name="3.3"

def threshDisplay(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    x, y, w, h = (380, 10, 240, 280)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cut = frame[y:y + h, x:x + w]
    gray = threshDisplay(cut)

    cv2.imshow("Gray",gray)
    cv2.imshow("Frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite("Test_Images/"+image_name+".jpg",gray)

camera.release()
cv2.destroyAllWindows()