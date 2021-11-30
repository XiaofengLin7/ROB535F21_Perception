#! /usr/bin/python3
import numpy as np
from glob import glob
import csv
import skimage
import skimage.io as io

import sys
sys.path.append('../')
#import utils
from utils.utils import rotate3d, get_bbox


classes = np.loadtxt('classes.csv', skiprows=1, dtype=str, delimiter=',')
labels = classes[:, 2].astype(np.uint8)


def write_labels(path):
    files = glob('{}/*/*_bbox.bin'.format(path))
    files.sort()
    name = '{}/trainval_labels.csv'.format(path)
    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image', 'label'])

        for file in files:
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_bbox.bin', '')

            bbox = np.fromfile(file, dtype=np.float32)
            bbox = bbox.reshape([-1, 11])
            #ipdb.set_trace()
            found_valid = False
            for b in bbox:
                # ignore_in_eval
                if bool(b[-1]):
                    continue
                found_valid = True
                class_id = b[9].astype(np.uint8)
                label = labels[class_id]
            if not found_valid:
                label = 0

            writer.writerow(['{}/{}'.format(guid, idx), label])

    print('Wrote report file `{}`'.format(name))


def write_centroids(path):
    files = glob('{}/*/*_bbox.bin'.format(path))
    files.sort()
    name = '{}/trainval_centroids.csv'.format(path)

    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image/axis', 'value'])

        for file in files:
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_bbox.bin', '')

            bbox = np.fromfile(file, dtype=np.float32)
            bbox = bbox.reshape([-1, 11])
            for b in bbox:
                # ignore_in_eval
                if bool(b[-1]):
                    continue
                xyz = b[3:6]
                for a, v in zip(['x', 'y', 'z'], xyz):
                    writer.writerow(['{}/{}/{}'.format(guid, idx, a), v])

    print('Wrote report file `{}`'.format(name))

def write_bboxes(path):
    files = glob('{}/*/*_bbox.bin'.format(path))
    files.sort()
    name = '{}/trainval_bboxes.csv'.format(path)

    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image', 'x', 'y', 'w', 'h'])

        for file in files:
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_bbox.bin', '')
            proj = np.fromfile(file.replace('_bbox.bin', '_proj.bin'), dtype=np.float32)
            img = file.replace('_bbox.bin', '_image.jpg')
            img = skimage.img_as_float64(io.imread(img))
            img_h, img_w, _ = img.shape
            proj.resize([3, 4])
            bboxes = np.fromfile(file, dtype=np.float32)
            bboxes = bboxes.reshape([-1, 11])
            for bbox in bboxes:
                # ignore_in_eval
                if bool(bbox[-1]):
                    continue
                #xyz = b[3:6]

                bbox = bbox.reshape([-1, 11])
                b = bbox[0]
                R = rotate3d(b[0:3])
                t = b[3:6]
                sz = b[6:9]
                vert_3D, edges = get_bbox(-sz / 2, sz / 2)
                vert_3D = R @ vert_3D + t[:, np.newaxis]

                vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
                vert_2D = vert_2D / vert_2D[2, :]
                #ipdb.set_trace()
                min_x = int(np.min(vert_2D[0]))
                max_x = int(np.max(vert_2D[0]))
                min_y = int(np.min(vert_2D[1]))
                max_y = int(np.max(vert_2D[1]))
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_y = min(img_h, max_y)
                max_x = min(img_w, max_x)
                x = (min_x + max_x) / 2
                y = (min_y + max_y) / 2

                w = max_x - min_x
                h = max_y - min_y
                xyz = np.array([ x, y, w, h])
                writer.writerow(['{}/{}'.format(guid, idx), x, y, w, h])

                #for a, v in zip(['x', 'y', 'w', 'h'], xyz):
                    #writer.writerow(['{}/{}/{}'.format(guid, idx, a), v])

    print('Wrote report file `{}`'.format(name))

if __name__ == '__main__':
    np.random.seed(0)
    for path in ['data/trainval']:
        write_labels(path)
        write_centroids(path)
        write_bboxes(path)
