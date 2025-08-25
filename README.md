YOLOv9
Implementation of paper - YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information

arxiv.org Hugging Face Spaces Hugging Face Spaces Colab OpenCV


Performance
MS COCO

Model	Test Size	APval	AP50val	AP75val	Param.	FLOPs
YOLOv9-T	640	38.3%	53.1%	41.3%	2.0M	7.7G
YOLOv9-S	640	46.8%	63.4%	50.7%	7.1M	26.4G
YOLOv9-M	640	51.4%	68.1%	56.1%	20.0M	76.3G
YOLOv9-C	640	53.0%	70.2%	57.8%	25.3M	102.1G
YOLOv9-E	640	55.6%	72.8%	60.6%	57.3M	189.0G
Useful Links
Expand
Installation
Docker environment (recommended)

Expand
Evaluation
yolov9-s-converted.pt yolov9-m-converted.pt yolov9-c-converted.pt yolov9-e-converted.pt yolov9-s.pt yolov9-m.pt yolov9-c.pt yolov9-e.pt gelan-s.pt gelan-m.pt gelan-c.pt gelan-e.pt

# evaluate converted yolov9 models
python val.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights './yolov9-c-converted.pt' --save-json --name yolov9_c_c_640_val

# evaluate yolov9 models
# python val_dual.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights './yolov9-c.pt' --save-json --name yolov9_c_640_val

# evaluate gelan models
# python val.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights './gelan-c.pt' --save-json --name gelan_c_640_val
You will get the results:

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.702
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.578
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.585
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.693
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.652
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.702
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.541
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.760
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.844
Training
Data preparation

bash scripts/get_coco.sh
Download MS COCO dataset images (train, val, test) and labels. If you have previously used a different version of YOLO, we strongly recommend that you delete train2017.cache and val2017.cache files, and redownload labels
Single GPU training

# train yolov9 models
python train_dual.py --workers 8 --device 0 --batch 16 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

# train gelan models
# python train.py --workers 8 --device 0 --batch 32 --data data/coco.yaml --img 640 --cfg models/detect/gelan-c.yaml --weights '' --name gelan-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15
Multiple GPU training

# train yolov9 models
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

# train gelan models
# python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/gelan-c.yaml --weights '' --name gelan-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15
Re-parameterization
See reparameterization.ipynb.

Inference

# inference converted yolov9 models
python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-c-converted.pt' --name yolov9_c_c_640_detect

# inference yolov9 models
# python detect_dual.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-c.pt' --name yolov9_c_640_detect

# inference gelan models
# python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './gelan-c.pt' --name gelan_c_c_640_detect
Citation
@article{wang2024yolov9,
  title={{ YOLOv9 }},
  author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
  booktitle={arXiv preprint arXiv:2402.13616},
  year={2024}
}
@article{chang2023yolor,
  title={{ YOLOR }},
  author={Chang, Hung-Shuo and Wang, Chien-Yao and Wang, Richard Robert and Chou, Gene and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2309.16921},
  year={2023}
}
Teaser
Parts of code of YOLOR-Based Multi-Task Learning are released in the repository.


Object Detection
gelan-c-det.pt

object detection

# coco/labels/{split}/*.txt
# bbox or polygon (1 instance 1 line)
python train.py --workers 8 --device 0 --batch 32 --data data/coco.yaml --img 640 --cfg models/detect/gelan-c.yaml --weights '' --name gelan-c-det --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 10
Model	Test Size	Param.	FLOPs	APbox
GELAN-C-DET	640	25.3M	102.1G	52.3%
YOLOv9-C-DET	640	25.3M	102.1G	53.0%
Instance Segmentation
gelan-c-seg.pt

object detection instance segmentation

# coco/labels/{split}/*.txt
# polygon (1 instance 1 line)
python segment/train.py --workers 8 --device 0 --batch 32  --data coco.yaml --img 640 --cfg models/segment/gelan-c-seg.yaml --weights '' --name gelan-c-seg --hyp hyp.scratch-high.yaml --no-overlap --epochs 300 --close-mosaic 10
Model	Test Size	Param.	FLOPs	APbox	APmask
GELAN-C-SEG	640	27.4M	144.6G	52.3%	42.4%
YOLOv9-C-SEG	640	27.4M	145.5G	53.3%	43.5%
Panoptic Segmentation
gelan-c-pan.pt

object detection instance segmentation semantic segmentation stuff segmentation panoptic segmentation

# coco/labels/{split}/*.txt
# polygon (1 instance 1 line)
# coco/stuff/{split}/*.txt
# polygon (1 semantic 1 line)
python panoptic/train.py --workers 8 --device 0 --batch 32  --data coco.yaml --img 640 --cfg models/panoptic/gelan-c-pan.yaml --weights '' --name gelan-c-pan --hyp hyp.scratch-high.yaml --no-overlap --epochs 300 --close-mosaic 10
Model	Test Size	Param.	FLOPs	APbox	APmask	mIoU164k/10ksemantic	mIoUstuff	PQpanoptic
GELAN-C-PAN	640	27.6M	146.7G	52.6%	42.5%	39.0%/48.3%	52.7%	39.4%
YOLOv9-C-PAN	640	28.8M	187.0G	52.7%	43.0%	39.8%/-	52.2%	40.5%
Image Captioning (not yet released)
object detection instance segmentation semantic segmentation stuff segmentation panoptic segmentation image captioning

# coco/labels/{split}/*.txt
# polygon (1 instance 1 line)
# coco/stuff/{split}/*.txt
# polygon (1 semantic 1 line)
# coco/annotations/*.json
# json (1 split 1 file)
python caption/train.py --workers 8 --device 0 --batch 32  --data coco.yaml --img 640 --cfg models/caption/gelan-c-cap.yaml --weights '' --name gelan-c-cap --hyp hyp.scratch-high.yaml --no-overlap --epochs 300 --close-mosaic 10
Model	Test Size	Param.	FLOPs	APbox	APmask	mIoU164k/10ksemantic	mIoUstuff	PQpanoptic	BLEU@4caption	CIDErcaption
GELAN-C-CAP	640	47.5M	-	51.9%	42.6%	42.5%/-	56.5%	41.7%	38.8	122.3
YOLOv9-C-CAP	640	47.5M	-	52.1%	42.6%	43.0%/-	56.4%	42.1%	39.1	122.0
Acknowledgements
Expand
