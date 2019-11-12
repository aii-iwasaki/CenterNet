import os
import sys
import cv2
import numpy as np
import pandas as pd
import json
import pickle
from xml.etree import ElementTree


def load_annotation(xml_file_path, all_classes=None, width_height=True):
    annotation = []
    root = ElementTree.parse(xml_file_path).getroot()
    for o in root.findall('object'):
        written_dom = o.find('written')
        written = written_dom.text == '1' if written_dom is not None else False
        xmin, xmax = int(o.find('bndbox').find('xmin').text), int(o.find('bndbox').find('xmax').text)
        ymin, ymax = int(o.find('bndbox').find('ymin').text), int(o.find('bndbox').find('ymax').text)
        class_name = o.find('name').text
        if all_classes is None or class_name in all_classes:
            annotation.append({
                'class': class_name,
                'position': [xmin, ymin, xmax - xmin, ymax - ymin] if width_height else [xmin, ymin, xmax, ymax],
                'written': written
            })
    return annotation


def dict2list(d):
    target_list = []
    for value in d.values():
        target_list += value
    return target_list


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_overlap(bbs, bbgt):
    # compute overlaps
    # intersection
    overlaps = []
    for bb in bbs:
        ixmin = np.maximum(bbgt[:, 0], bb[0])
        iymin = np.maximum(bbgt[:, 1], bb[1])
        ixmax = np.minimum(bbgt[:, 2], bb[2])
        iymax = np.minimum(bbgt[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (bbgt[:, 2] - bbgt[:, 0] + 1.) *
               (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

        overlaps.append(inters / uni)
    return np.array(overlaps)


def reform_result(detection_result):
    det_res = {}
    for d in detection_result:
        c_id = d['category_id']
        i_id = d['image_id']
        if i_id not in det_res:
            det_res[i_id] = {}

        if c_id not in det_res[i_id]:
            det_res[i_id][c_id] = []

        det_res[i_id][c_id].append(np.array(list(d['bbox']) + [d['score']]))
    return det_res


def evaluate_detection(detection_file, image_path_list, all_classes, iou_threshold, output_path=None, white_list=None,
                       verbose=1):
    if white_list is None:
        white_list = all_classes

    with open(os.path.join(detection_file), 'rb') as f:
        if 'json' in detection_file:
            detection_result = json.load(f)
        else:
            detection_result = pickle.load(f, encoding='latin1')

    detection_result = reform_result(detection_result)
    recall_dict, precision_dict, average_precisions = {}, {}, {}
    for label in range(len(all_classes)):
        if white_list is not None and not (all_classes[label] in white_list):
            continue
        if all_classes[label] == '__background__':
            continue
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0
        num_true_positive, num_false_positive = 0, 0

        for image_path in image_path_list:
            if not os.path.isfile(os.path.splitext(image_path)[0] + '.xml'):
                return
            annotation = load_annotation(os.path.splitext(image_path)[0] + '.xml',
                                         all_classes=[all_classes[label]], width_height=False)
            detections = np.zeros((0, 5))
            annotations = np.zeros((0, 4))

            for d in detection_result[image_path][label]:
                detections = np.vstack((detections, d))

            # for b, s, l in zip(box_list, score_list, label_list):
            #     if l == label:
            #         detections = np.vstack((detections, np.array(list(b) + [s])))
            for a in annotation:
                if a['class'] == all_classes[label]:
                    annotations = np.vstack((annotations, np.array(a['position'])))

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if len(annotations) == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                    num_true_positive += 1
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    num_false_positive += 1

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            recall_dict[label] = 0, 0
            precision_dict[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        recall_dict[label] = num_true_positive / num_annotations, num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        num_precision = 1 if num_true_positive + num_false_positive == 0 else num_true_positive / (num_true_positive + num_false_positive)
        precision_dict[label] = num_precision, num_annotations

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    # print evaluation
    columns = ['class', 'sample', 'precision', 'recall', 'f1', 'ap']
    result_table = pd.DataFrame(columns=columns)
    precision_list, recall_list, avg_precisions = [], [], []
    for label in precision_dict.keys():
        precision, num_annotations = precision_dict[label]
        recall, _ = recall_dict[label]
        average_precision, _ = average_precisions[label]
        if 0 < num_annotations:
            precision_list = precision_list + [precision] * int(num_annotations)
            recall_list = recall_list + [recall] * int(num_annotations)
            avg_precisions = avg_precisions + [average_precision] * int(num_annotations)

            f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            row = [all_classes[label], int(num_annotations), round(precision, 4), round(recall, 4), round(f1, 4), round(average_precision, 4)]
            result_table = result_table.append(pd.DataFrame([row], columns=columns))

    acc = sum(avg_precisions) / len(avg_precisions)
    precision = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    row = ['summary', len(avg_precisions), round(precision, 4), round(recall, 4), round(f1, 4), round(acc, 4)]
    result_table = result_table.append(pd.DataFrame([row], columns=columns))
    if verbose:
        print('-------- detection result --------')
        result_table.to_csv(sys.stdout, sep='\t', index=False)

    if output_path:
        result_table.to_csv(output_path, sep='\t', index=False)

    return acc, precision, recall, f1

