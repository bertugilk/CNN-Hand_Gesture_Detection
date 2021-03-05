import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image

threshold = 0.65
num = 0
camera = cv2.VideoCapture(0)
model=load_model('Model/hand_gestures2.h5')

def digits(img):
    global num
    classIndex = int(model.predict_classes(img))

    if classIndex == 0:
        num = 5
    elif classIndex == 1:
        num = 4
    elif classIndex == 3:
        num = 3
    elif classIndex == 4:
        num = 2
    elif classIndex == 2:
        num = 1
    else:
        num=None
    return num

def preProcessing(img):
    img=image.img_to_array(img)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)
    img = np.reshape(img,(1,64,64,1))
    return img

def threshDisplay(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def main():
    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)

        x, y, w, h = (380, 10, 240, 280)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cut = frame[y:y + h, x:x + w]

        gray=threshDisplay(cut)
        img = preProcessing(gray)

        #cv2.imshow("image",gray)

        digit = digits(img)
        predictions = model.predict(img)
        probVal= np.amax(predictions)

        print("Digit: ",digit)
        print("ProbVal: ",probVal)

        if probVal > threshold:
            cv2.putText(frame,"Digit: "+str(digit),(380,320),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)
            cv2.putText(frame,"Accuracy: "+str(probVal), (380, 350), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Original Image",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()