import os
import sys
import shutil
import cv2
import tqdm
import time
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import python.pymz as mz
from python.datasets.coco import COCO2017Val
from python.utils import detections2txt, detection_txt2json, coco_eval, convert


def evaluate(m, dataset_path):
    bs = 1
    dataset = COCO2017Val(dataset_path)
    img_paths = dataset.get_datas(0)

    save_results = "results"
    if os.path.exists(save_results):
        shutil.rmtree(save_results)
    os.makedirs(save_results)

    duration = 0
    cnt = 0
    for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
        basename = os.path.basename(img_path)
        filename, ext = os.path.splitext(basename)
        label_path = os.path.join(save_results, "{}.txt".format(filename))
        cv_image = cv2.imread(img_path)
        if cv_image is None:
            print("Failed to decode img by opencv -> {}".format(img_path))
            continue
        t0 = time.time()
        detections = m.inference(cv_image)
        duration += time.time() - t0
        detections = convert(detections)
        cnt += 1
        detections2txt(detections, label_path)

    pred_json = "pred.json"
    detection_txt2json(save_results, pred_json, to_coco91=True)
    _map, map50 = coco_eval(pred_json, dataset.annotations_file, dataset.image_ids)
    return {
        "input_size": "{}x{}x{}x{}".format(bs, 3, 640, 640),
        "dataset": dataset.dataset_name,
        "num": len(img_paths),
        "map": "{:.6f}".format(_map),
        "map50": "{:.6f}".format(map50),
        "latency": "{:.6f}".format(duration * 1000.0 / cnt)
    }
        

def demo(m, args):
    cv_image = cv2.imread(args.img_path)
    if cv_image is None:
        print("Failed to load img -> {}".format(args.img_path))
        exit(-1)

    detections = m.inference(cv_image)

    for detection in detections:
        cv2.rectangle(cv_image, (detection.box.x1, detection.box.y1),
                  (detection.box.x2, detection.box.y2), (0, 0, 255), 2, 8)

    cv2.imwrite(args.save_path, cv_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python model")
    parser.add_argument("type", type=str, choices=("demo", "eval"),
                        help="Please specify a operator")
    parser.add_argument("--cfg", "-c", type=str, default="/config/sdk.cfg", required=False, 
                        help="Please specify sdk.cfg path")
    parser.add_argument("--model_path", "-m", type=str, required=True, 
                        help="Please specify a model")
    parser.add_argument("--img_path", "-i", type=str, required=("demo" in sys.argv),
                        help="Please specify a img path, work in demo mode")
    parser.add_argument("--save_path", "-s", type=str, required=("demo" in sys.argv),
                        help="Please specify a save path, work in demo mode")
    parser.add_argument("--dataset_path", "-d", type=str, required=("eval" in sys.argv), 
                        help="Please specify coco2017 dataset path, work in eval mode")
    args = parser.parse_args()

    mz.dcl_init(args.cfg)

    m = mz.YOLOv8()
    if m.load(args.model_path) != 0:
        print("Failed to load model -> {}".format(args.model_path))
        exit(-1)

    if args.type == "demo":
        m.set_iou_threshold(0.45)
        m.set_conf_threshold(0.25)
        demo(m, args)
    elif args.type == "eval":
        m.set_iou_threshold(0.65)
        m.set_conf_threshold(0.01)
        print(evaluate(m, args.dataset_path))
    else:
        print("Not support type")

    if m.unload() != 0:
        print("Failed to unload")
        exit(-1)

    mz.dcl_finalize()


