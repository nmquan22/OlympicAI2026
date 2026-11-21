"""Official COCO bbox evaluator for original COCO JSON and exported predictions."""
import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations")
    parser.add_argument("detections")
    args = parser.parse_args()
    truth = COCO(args.annotations)
    with open(args.detections, encoding="utf-8") as stream:
        predictions = json.load(stream)
    if predictions:
        detected = truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = {**truth.dataset, "annotations": []}
        detected.createIndex()
    metric = COCOeval(truth, detected, "bbox")
    metric.evaluate()
    metric.accumulate()
    metric.summarize()


if __name__ == "__main__":
    main()
