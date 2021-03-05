from keras.models import load_model
import numpy as np
from keras.preprocessing import image

model=load_model('Model/hand_gestures.h5')

#test_img= image.load_img('Dataset/fingers/four/75.png',target_size=(64,64))
test_img= image.load_img('Test_Images/1.2.jpg', target_size=(64, 64))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)

classIndex = int(model.predict_classes(test_img))
print("Index: ",classIndex)

if classIndex==0:
    print("Five")
elif classIndex==1:
    print("Four")
elif classIndex==3:
    print("Three")
elif classIndex==4:
    print("Two")
elif classIndex==2:
    print("One")
# According to the folder order in the data set:
# 5 -> 0.index
# 4 -> 1.index
# 3 -> 3.index
# 2 -> 4.index
# 1 -> 2.index