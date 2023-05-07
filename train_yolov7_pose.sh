# Download YOLOv7 repository and make necessary changes
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
git checkout pose
sed -i 's/onnxruntime/# onnxruntime/g' requirements.txt
cd ..

#### COCO KEYPOINT ANNOTATION DOWNLOAD CODE #####
test -e coco && rm -rf coco || echo "coco directory doesn't exist"
wget -P coco/ https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-keypoints.zip
unzip coco/coco2017labels-keypoints.zip -d coco/
rm -rf coco/coco2017labels-keypoints.zip

#### COCO VALID 2017 DOWNLOAD #####
wget -P coco/ http://images.cocodataset.org/zips/val2017.zip
unzip coco/val2017.zip -d coco/images
rm -rf coco/val2017.zip

#### COCO TRAIN 2017 DOWNLOAD #####
wget -P coco/ http://images.cocodataset.org/zips/train2017.zip
unzip coco/train2017.zip -d coco/images
rm -rf coco/train2017.zip

#### DOWNLOAD THE PRE TRAINED MODEL ####
wget -P yolov7/weights/ https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt

#### MAKE NECESSARY CHANGES TO THE YAML FILE REQUIRED FOR TRAINING ####
echo "train: coco/train2017.txt" > yolov7/data/coco2017_kpts.yaml
echo "val: coco/val2017.txt" >> yolov7/data/coco2017_kpts.yaml
echo "nc: 1" >> yolov7/data/coco2017_kpts.yaml
echo "names: ['person']" >> yolov7/data/coco2017_kpts.yaml


# Start Training
# docker-compose run posture pip install -r yolov7/requirements.txt && python  yolov7/train.py --data yolov7/data/coco2017_kpts.yaml --cfg yolov7/cfg/yolov7-w6-pose.yaml --weights yolov7/weights/yolov7-w6-pose.pt --batch-size 128 --img 960 --kpt-label --sync-bn --device 0 --name yolov7-w6-pose --hyp yolov7/data/hyp.pose.yaml