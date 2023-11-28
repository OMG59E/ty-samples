import os
import json
import cv2
import traceback


def convert(detections):
    _detections = []
    for detection in detections:
        _detections.append((detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2, detection.conf, detection.cls))
    return _detections
    
def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    
    
def detections2txt(detections, filepath):
    with open(filepath, "w") as f:
        for det in detections:
            (x1, y1, x2, y2), conf, cls = det[0:4], det[4], det[5]
            text = "{} {} {} {} {} {}\n".format(conf, cls, x1, y1, x2, y2)
            f.write(text)
 

def detections_mask2json(detections, contours_lists: list, filepath):
    with open(filepath, "w") as f:
        if not contours_lists:
            return
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        image_id = int(name)
        pred_lists = list()
        for idx, det in enumerate(detections):
            (x1, y1, x2, y2), conf, cls = det[0:4], det[4], det[5]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            conf = float(conf)
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cls = int(cls)
            category_id = coco80_to_coco91_class()[cls]
            contours = contours_lists[idx]
            new_contours = list()
            area = 0
            for _, contour in enumerate(contours):
                if contour.shape[0] <= 2:
                    continue
                area += cv2.contourArea(contour)
                new_contour = contour.flatten().tolist()
                if len(new_contour) == 4:
                    new_contour.append(new_contour[-1])
                new_contours.append(new_contour)
            if len(new_contours) == 0:
                continue
            pred_lists.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "score": conf,
                "segmentation": new_contours,
                "area": area,
                "iscrowd": 0
            })
        f.write(json.dumps(pred_lists))
        

def detections_kpt2json(outputs, filepath):
    with open(filepath, "w") as f:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        image_id = int(name)
        pred_lists = list()
        for output in outputs:
            cls = int(output[5])
            conf = float(output[4])
            x1, y1, x2, y2 = output[0:4].tolist()
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            kpts = output[6:].tolist()
            for k in range(len(kpts)):
                if (k + 1) % 3 == 0:
                    kpts[k] = 1
            category_id = coco80_to_coco91_class()[cls]
            pred_lists.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "score": conf,
                "keypoints": kpts,
                "iscrowd": 0
            })
        f.write(json.dumps(pred_lists))
        

def merge_json(save_results, pred_json):
    label_files = os.listdir(save_results)
    results = list()
    for filename in label_files:
        name, ext = os.path.splitext(filename)
        if ext != ".json":
            continue
        with open(os.path.join(save_results, filename), "r") as f:
            line = f.read().strip()
            if len(line) == 0:
                continue
            detections = json.loads(line)
            if not detections:
                continue
            results.extend(detections)
    with open(pred_json, "w") as f:
        json.dump(results, f)


def detection_txt2json(save_results, pred_json, to_coco91=True):
    """将检测的txt结果转为coco json
    JSON format [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...]
    :param save_results:
    :param pred_json:
    :param to_coco91:
    :return:
    """
    label_files = os.listdir(save_results)
    pred_list = list()
    for filename in label_files:
        name, ext = os.path.splitext(filename)
        if ext != ".txt":
            continue
        image_id = int(name)
        with open(os.path.join(save_results, filename), "r") as f:
            lines = f.readlines()
            for line in lines:
                conf, cls, x1, y1, x2, y2 = line.strip().split()
                conf = float(conf)
                cls = int(float(cls))
                x1 = int(float(x1))
                y1 = int(float(y1))
                x2 = int(float(x2))
                y2 = int(float(y2))
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                category_id = coco80_to_coco91_class()[cls] if to_coco91 else cls
                pred_list.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "score": conf
                })
    with open(pred_json, "w") as f:
        json.dump(pred_list, f)
    print("Write pred results to json file -> {}".format(pred_json))
    

def coco_eval(pred_json, anno_json, image_ids, iou_type="bbox"):
    """ coco 评估方法
    :param pred_json:
    :param anno_json:
    :param image_ids:
    :param iou_type:
    :return:
    """
    print("Evaluating pycocotools mAP... saving {}...".format(pred_json))
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        cocoGt = COCO(anno_json)  # init annotations api
        pred = cocoGt.loadRes(pred_json)  # init predictions api
        eval = COCOeval(cocoGt, pred, iou_type)
        eval.params.imgIds = cocoGt.getImgIds()  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        _map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        return _map, map50
    except Exception as e:
        print("pycocotools unable to run: {}\n{}".format(e, traceback.format_exc()))
        exit(-1)


           