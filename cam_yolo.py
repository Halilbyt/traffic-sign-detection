import numpy as np
import cv2 as cv
import time

# Kod taslak adımları:

# 1-) Kameradan görüntünün okunması
# 2-) yolo v3 network'ün yüklenmesi (cfg ve ağırlıklar)
# 3-) frame'in flob değerlerinin alınması
# 4-) flobların ağ çıktılarının hesaplanması (ileri yönde geçiş)
# 5-) Dikdörtgen parametrelerinin alınması (objenin merkez kordinatları ve genişlik ile uzunluk değerleri)
# 6-) Bir obje için birden fazla tahmini değer var ise en büyüğünün seçilmesi
# 7-) Dikdörtgenin çizilmesi, text değerlerinin atanması ve sonuçşarın gözlemlenmesi (son)

cap = cv.VideoCapture(0)
w,h  = None,None

# Sınıf isimlerinin alınması
path_names = r'YOLO-3-OpenCV\yolo-coco-data\coco.names'
with open(path_names) as f:
    labels = [lines.strip() for lines in f]

# yolo v3 ağının yüklenmesi config ve ağırlık dosyaları

config_path = r'YOLO-3-OpenCV\yolo-coco-data\yolov3.cfg'
weight_path = r'YOLO-3-OpenCV\yolo-coco-data\yolov3.weights'

network = cv.dnn.readNetFromDarknet(config_path,weight_path)

# çıkış katmanlarının isimlerinin alınması 3 tane

layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i-1] for i in network.getUnconnectedOutLayers()]

''' ['yolo_82', 'yolo_94', 'yolo_106'] '''

# min olasılık sınırı
probability_min = 0.5
# karelerin üst üste gelmesini onlemek üçün alt sınır
threshold = 0.3
colours = np.random.randint(0,255,size = (len(labels),3),dtype = 'uint8')

while True:
    ret,frame = cap.read()
    if not ret:
        break
    # getting special dimensions of the frame : 
    if w is None or h is None:
        h,w = frame.shape[:2]
    #getting blob from current frame
    blob = cv.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB = True, crop = False)
    #Implemeting forward pass with our blob and its only through output layers : 
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
   
    #getting bounding boxes : 
    bounding_boxes = []
    confidence = []
    class_number = []

    for results in output_from_network:
        for detected_objects in results:
            scores = detected_objects[5:] # getting 80 class probability results
            class_current = np.argmax(scores) #getting index of class that max probability
            confidence_current = scores[class_current] # getting probablity result the class that have max 
            if confidence_current > probability_min: 
                # first two values is center point of the objects and 3. and 4. is width and height
                box_current = detected_objects[0:4]*np.array([w,h,w,h])
                x_center,y_center,box_width,box_height = box_current
                #finding top left cordinat values : 
                x_min = int(x_center - (box_width)/2)
                y_min = int(y_center - (box_height)/2)

                bounding_boxes.append([x_min,y_min,int(box_width),int(box_height)])
                confidence.append(float(confidence_current))
                class_number.append(class_current)
    
    results = cv.dnn.NMSBoxes(bounding_boxes,confidence,probability_min,threshold)

    if len(results) > 0:
        for i in results.flatten():
            x_min,y_min = bounding_boxes[i][0],bounding_boxes[i][1]
            box_width,box_height = bounding_boxes[i][2],bounding_boxes[i][3]
            colour_box_current = colours[class_number[i]].tolist()

            cv.rectangle(frame,(x_min,y_min),(box_width+x_min,box_height+y_min),colour_box_current,2)
            text_box_current = '{} : {:.4f}'.format(labels[int(class_number[i])],confidence[i])
            cv.putText(frame,text_box_current,(x_min,y_min-5),cv.FONT_HERSHEY_COMPLEX,0.5,colour_box_current,2)
    
    cv.imshow('Detect Object',frame)
    if cv.waitKey(20) & 0xFF == 27:
        break