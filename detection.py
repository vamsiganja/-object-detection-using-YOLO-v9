#Import All the Required Libraries
import argparse
import os
import platform
import sys
from pathlib import Path
########################

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#######################
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (non_max_suppression, xyxy2xywh, strip_optimizer, cv2, increment_path, check_requirements, check_imshow,
check_file, check_img_size,Profile, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode
import math
import time
opt = {
    "weights" : "weights/yolov9-c.pt",
    "imgsz" : (640, 640),
    "conf_thres": 0.25,
    "iou_thres" : 0.45,
    "classes": None,
    "device": 'cpu',
    "half": False,
    "vid_stride": 1, #video frame-ratge stride
    "augment" : False,
    "visualize" : False,
    "agnostic_nms" : False,
    "max_det" : 1000,
    "name": 'exp',
    "exist_ok": False,
    "save_txt": False,
}
def classNames():
    cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush","door","tree"
                  ]
    return cocoClassNames

def colorLabels(classid):
    if classid == 0: #person
        color = (85, 45, 255)
    elif classid == 2: #car
        color = (222, 82, 175)
    elif classid == 3: #Motorbike
        color = (0, 204, 255)
    elif classid == 5: #Bus
        color = (0,149,255)
    else:
        color = (200, 100,0)
    return tuple(color)

@smart_inference_mode()
def objectDetection(source):
    frameCount = 0
    ptime = 0
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_file and is_url:
        source = check_file(source)
    #Directories
    save_dir = increment_path(Path(ROOT / 'runs/detect') / opt["name"], exist_ok=opt["exist_ok"])  # increment run
    (save_dir / 'labels' if opt["save_txt"] else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    #Load Model
    device = select_device(opt["device"])
    #model = DetectMultiBackend(opt['weights'], device = opt["device"], dnn = False, data = ROOT / 'data/coco.yaml',  fp16= opt['half'])
    model = DetectMultiBackend(opt['weights'], device=device, dnn=False, data=ROOT / 'data/coco.yaml', fp16=opt['half'])

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(opt["imgsz"], s = stride)
    #Dataloader
    bs = 1 #batchsize
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=opt["vid_stride"])
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, stride = stride, auto=pt, vid_stride= opt["vid_stride"])
    vid_path, vid_writer = [None] * bs, [None] * bs

    #Run Inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        frameCount += 1
        print("Frame No :", frameCount)
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float() #uint8 to fp16/32
            im /= 255 #0 - 255 to 0.0  -  1.0
            if len(im.shape) == 3:
                im = im[None] #expand for batch dim
        # Inference
        with dt[1]:
            visualize = opt["visualize"]
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=opt["augment"], visualize=visualize)
            pred = pred[0][1]
            #print("Done 1")
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, opt["conf_thres"], opt["iou_thres"], opt["classes"], opt["agnostic_nms"], max_det=opt["max_det"])
            #print("Done 2 and predictions",pred)
        totalDetections = 0
        frameRate = 0
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #save_crop = opt["save_crop"]
            #imc = im0.copy() if save_crop else im0  # for save_crop
            #print("Length of Detections", len(det))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    totalDetections += int(n)
                    #print("Total Detections", totalDetections)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print("x1, y1, x2, y2", x1, y1, x2, y2)
                    cat = int(cls)
                    color = colorLabels(cat)
                    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 3)
                    className = classNames()
                    name = className[cat]
                    conf = math.ceil((conf * 100))/100
                    label = f'{name}{conf}'
                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(im0, (x1, y1), c2, color, -1)
                    cv2.putText(im0, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                ctime = time.time()
                fps = 1/(ctime - ptime)
                ptime = ctime
                frameRate += fps
                #cv2.putText(im0, str(int(fps)), (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 100, 100), 3)
                #cv2.imshow("Image", im0)
                #cv2.waitKey(1)
                if dataset.mode  == 'image':
                    cv2.imwrite(save_path, im0)
                else: #video or stream
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release() #release previous video writer
                        if vid_cap: #video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            #frameRate += fps
                        else: #stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4')) #force .mp4 suffix on result videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        #print("Total Detections", totalDetections)
        #print("Frame Rate", frameRate)
        yield im0, frameRate, im0.shape, totalDetections

objectDetection("resource/image1.jpg")