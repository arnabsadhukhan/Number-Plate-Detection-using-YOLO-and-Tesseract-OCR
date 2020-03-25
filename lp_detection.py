import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pytesseract
import re



net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))





licence_cascade = cv2.CascadeClassifier('indian_license_plate.xml')





def cut_lp(image):
    
    mask = cv2.inRange(image,(150,150,150),(256,256,256))
    x_axis = np.sum(mask,axis=0)
    x_axis = x_axis/np.max(x_axis)
    
    x_axis = x_axis>0.2
    temp = np.where(x_axis==1)[0]
   
    x1 = temp[0]
    x2 = temp[-1]
    x_axis = np.sum(mask,axis=1)
    x_axis = x_axis/np.max(x_axis)
    x_axis = x_axis>0.2
    temp = np.where(x_axis==1)[0]
    y1 = temp[0]
    y2 =temp[-1]
    return image[y1:y2,x1:x2]




cap = cv2.VideoCapture("video_test.mp4")


pytesseract.pytesseract.tesseract_cmd ='Tesseract-OCR/tesseract.exe'

window_name = 'Image'
  
  
#fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 255, 255) 
  
# Line thickness of 2 px 
thickness = 2




done = False
cv2.destroyAllWindows()
#cap = cv2.VideoCapture(1)
count = 0
while not done:
    #req = requests.get(url)
    #img_arr = np.array(bytearray(req.content),dtype = np.uint8)
    #frame = cv2.imdecode(img_arr,-1)
    ret,frame = cap.read()
    frame = cv2.resize(frame,(960,540))
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    img = frame
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices =  cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.4)
    no_of_cars =0
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        if no_of_cars>=1:
            break
        if class_ids[i]==2:
            no_of_cars+=1
            #x ,y = x-10,y-10
            #w,h = w+20,h+20
            cv2.rectangle(img, (x, y), (x+w , y+h), (255,0,255), 2)
            x,y,w,h = np.abs([x,y,w,h])*2.5
            #try:
            if True:
                car = frame[int(y):int(y+h+50),int(x):int(x+w)]
                cv2.imshow('cars',car)
                gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
                plates = licence_cascade.detectMultiScale(frame,scaleFactor = 1.3, minNeighbors = 7)
                
                for (x,y,w,h) in plates:
                    
                    plate = cv2.cvtColor(cut_lp(frame[int(y-10):int(y+h+10),int(x-10):int(x+w+10)]), cv2.COLOR_BGR2GRAY)
                    cv2.imshow('plate',plate)
                    plate = cv2.blur(plate,(5,5))
                    result = pytesseract.image_to_string(plate)
                    detected_lp = "".join(re.split('[^0-9A-Z]',result))
                    if result !='' and len(result)>5:
                        print('Detected LP:',detected_lp)
                    
                    
    cv2.imshow('frame',cv2.resize(img,(600,300)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        done = True
    if count==54:
        cv2.destroyAllWindows()
        done = True      
    
cap.release()

cv2.destroyAllWindows()




