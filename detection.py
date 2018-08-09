# -*- coding: utf-8 -*-
from imageai.Detection import ObjectDetection
#ImageAI nesne algılama sınıfını import ettik 
import os

main_path = os.getcwd()
print(main_path)
#anadizinin yolunu bulup main_path değişkenine atadık. 

detector = ObjectDetection()
#nesne tespit sınıfımızı ilk satırda tanımladık
detector.setModelTypeAsRetinaNet()
#model tipini RetinaNet'e ayarladık
detector.setModelPath( os.path.join(main_path , "resnet50_coco_best_v2.0.1.h5"))
#modelin yolunu belirledik.
detector.loadModel()
#model yüklendi.
detections = detector.detectObjectsFromImage(input_image=os.path.join(main_path ,"test_images", "trafic.jpg"), output_image_path=os.path.join(main_path ,"detected_images", "trafic.jpg"))
#input_image kısmında giriş resmimizi tanımladık.
#output_image_path kısmında ise detect edildikten sonra kaydedilecek resmin yolunu tanımladık. 

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"]  )
    #  eachObject["box_points"] eklenebilir.
#görüntüde algılanan her nesne üzerindeki modelin ismini ve yüzde olasılığını yazdırırız.