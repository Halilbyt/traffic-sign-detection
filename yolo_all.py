import cv2 as cv
import numpy as np
import time

from sklearn.utils import resample

'''
Kamera Uygulamasi
'''
# Camera objesi oluşturuldu
cap = cv.VideoCapture(0)
w,h = None,None
# video için 
'''
path = 'buses-to-test.mp4'
path2 = 'traffic-sign-to-test.mp4'
cap = cv.VideoCapture(path)
w,h = None,None
'''
path_names2 = ''
path_names = 'classes.names'
with open(path_names) as f:
    etiketler = [lines.strip() for lines in f]
#print(etiketler)
'''
 >coco datasets etiketleri<
80 tane etiket listesinin hepsi:  
['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 
'bus', 'train', 'truck', 'boat', 'traffic light', 
'fire hydrant', 'stop sign', 'parking meter',
 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
  'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 
  'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
   'remote', 'keyboard', 'cell phone',
 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

'''
#
# Modelin yüklenmesi coco network ağırlıklar ve config dosyası ki bu dosyoda ağ parametreleri yer alıyor (katmanlar aktivasyon fonksiyonları vs)
#
path_agirlik = 'yolov3_ts.weights'
path_cfg = 'yolov3_ts_train.cfg'

network = cv.dnn.readNetFromDarknet(path_cfg,path_agirlik)
katman_isimleri = network.getLayerNames() # => ağ isimlerinin alınması bunu yapmamızın sebebi bu ağ isimlerinden sadece çıkış katmanları bize gerekiyor
cikis_katman_isimleri = [katman_isimleri[i - 1] for i in network.getUnconnectedOutLayers()]
#print(cikis_katman_isimleri) => ['yolo_82', 'yolo_94', 'yolo_106'] üç tane çıkış katmanı mevcut bunlardan gelen çıkışlar gerekli bize

min_olasilik = 0.5
alt_deger = 0.3
renkler = np.random.randint(0,255,size = (len(etiketler),3), dtype = 'uint8')
#print(renkler.shape) => (80,3) 80 tane etiket değerine karşılık 80 tane renk
   

while True:
    ret,frame = cap.read()
    frame = cv.flip(frame,1)
    if not ret:
        break
    # frame'in boy ve genişlik değerlerinin atanması:: 
    if w is None or h is None:
        h,w = frame.shape[:2]
    #okunan ilk frame değerinin blob formatına dönüştürlmesi
    blob = cv.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB = True, crop = False)
    # ön eğitimli ağa blob değerinin gönderilmesi 
    network.setInput(blob)
    
    ag_ciktisi = network.forward(cikis_katman_isimleri)
    
   
    cerceve_siniri = []
    olasilik_degeri = []
    sinif_numarasi = []

    for sonuclar in ag_ciktisi: # ağ çıktısı için üc tane cıkış katman sonuclarını teker teker dolasışr
        for saptanan_nesneler in sonuclar: # çıkış katmanlarından birisi için nesne tespit sonuçlarını alır..
            skorlar = saptanan_nesneler[5:] # ilk dört eleman x-y kordinatları ve uzunluk ve genişlik değerlerini verir
            simdiki_sinif = np.argmax(skorlar) # en yüksek olasılık değerini veren sınıfın indexini bulur
            simdiki_olasilik = skorlar[simdiki_sinif] # bu index sayesinde max skorun kendisini elde ederiz
            if simdiki_olasilik > min_olasilik: # eğer olasılık değeri bizim sınır olarak belirttiğimiz değerden yüksek ise::
                simdiki_cerceve = saptanan_nesneler[0:4]*np.array([w,h,w,h]) # olasılık değeri tatmin edici ise bu değerin ilk 4 ünü al
                x_merk,y_merk,cerceve_gen,cerceve_yuk = simdiki_cerceve # parametreleri ayrı değişkenlere ata
                x_min,y_min = int(x_merk - (cerceve_gen/2)), int(y_merk - (cerceve_yuk/2)) # çizilecek dikdörtgenin sol üst köse kordinatlarını hesapla
                cerceve_siniri.append([x_min,y_min,int(cerceve_gen),int(cerceve_yuk)]) # dikdörtgen çizmek için gerekli parametreleri ceceve sinir listesine ekle
                olasilik_degeri.append(float(simdiki_olasilik)) # tespit edilen nesnelerin olasılık değerlerini  olasılık değeri listesine ekle
                sinif_numarasi.append(simdiki_sinif) # hangi sinıfa ait olduğunun bilgisini de sınıf numarası listesine ekle
    
    sonuclar = cv.dnn.NMSBoxes(cerceve_siniri,olasilik_degeri,min_olasilik,alt_deger) # sonuclar olarak cercevelerin birden fazlasının üst üste gelmemesi için bu fonksiyon
                                                                                    # kullanılır mantığı ise bir kare de birden fazla tespit için üst üste binen kareler çizil,
                                                                                    # mesinin önüne geçmek için en yüksek olasılık değerini vereni alır.
    sayac = 1
    if len(sonuclar) > 0:
        for i in sonuclar.flatten():
            sayac += 1
            x_min,y_min =  cerceve_siniri[i][0],cerceve_siniri[i][1]
            cerceve_gen,cerceve_yuk = cerceve_siniri[i][2],cerceve_siniri[i][3]
            renkler_simdiki = renkler[sinif_numarasi[i]].tolist()
            cv.rectangle(frame,(x_min,y_min),(cerceve_gen+x_min,cerceve_yuk+y_min),renkler_simdiki,3)
            text_simdiki = '{} : {: .4f}'.format(etiketler[int(sinif_numarasi[i])],olasilik_degeri[i])
            cv.putText(frame,text_simdiki,(x_min,y_min-5),cv.FONT_HERSHEY_COMPLEX,0.7,renkler_simdiki,2)
        
    #cv.imshow('4 Sinif icin Tabela Tespiti',frame)
    print(sonuclar)
    print(type(sonuclar))
    if cv.waitKey(20) & 0xFF == 27:
        break