import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import json 
from PIL import Image, ImageDraw
import torch
import pandas as pd
from pathlib import Path

from strhub.data.module import SceneTextDataModule
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg


# constants
WINDOW_NAME = "PARseq OCR"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
      "--csv",
      help="The output bbox + label",
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )
            poly = predictions['instances'].polygons.cpu().numpy()
            poly_bbox= []
            for itr in poly:
                _x = itr[::2]
                _y = itr[1::2]
                poly_bbox.append([_x,_y])
            bbox = []
            for itr in poly_bbox:
                x_min = min(itr[0])
                x_max = max(itr[0])
                y_min = min(itr[1])
                y_max = max(itr[1])
                bbox.append([x_min,y_min,x_max,y_max])
            pil_img = Image.fromarray(img)
            img_draw = ImageDraw.Draw(pil_img)
            text = []
            for box in bbox:
                img_draw.rectangle(box,outline = 'red')
                cropped_img = pil_img.crop(box)
                img_transformed = img_transform(cropped_img).unsqueeze(0)
                logits = parseq(img_transformed)
                pred = logits.softmax(-1)
                label, confidence = parseq.tokenizer.decode(pred)
                text.append(label[0])
                
            output_dict = {
              'bbox': bbox,
              'text': text
            }
            df = pd.DataFrame(output_dict)
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                pil_img.save(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
            if args.csv:
                if os.path.isdir(args.csv):
                    assert os.path.isdir(args.csv), args.csv
                    out_filename = os.path.join(args.csv, Path(path).stem + '.csv')
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.csv"
                    out_filename = args.csv
                df.to_csv(out_filename)