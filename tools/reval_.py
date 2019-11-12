#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Xingyi Zhou
# --------------------------------------------------------

# Reval = re-eval. Re-evaluate saved detections.
import sys
import os.path as osp

# sys.path.insert(0, osp.join(osp.dirname(__file__), 'voc_eval_lib'))

# from model.test import apply_nms
from .evaluate import evaluate_detection
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('detection_file', type=str)
    parser.add_argument('--output_dir', help='results directory', type=str)
    parser.add_argument('--output_dir', help='results directory', type=str)
    parser.add_argument('--nms', dest='apply_nms', help='apply nms',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def from_dets(detection_file):

    evaluate_detection(detection_file, image_path_list, all_classes, iou_threshold)


if __name__ == '__main__':
    args = parse_args()

    imdb_name = args.imdb_name
    from_dets(imdb_name, args.detection_file, args)
