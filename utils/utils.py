import os
import re
import glob
from xml.etree import ElementTree


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


def img_glob(path):
    ext = '.*\.(jpg|jpeg|png|bmp|gif)$'
    path = os.path.join(path, '*')
    files = glob.glob(path)
    files = [f for f in files if re.search(ext, f, re.IGNORECASE)]
    return files


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
