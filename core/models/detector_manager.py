from __future__ import print_function

import time
from .mtcnn import MTCNN


class DetectorManager(object):
    def __init__(self, path, use_gpu, verbose_level):
        self._model = None
        self.use_gpu = use_gpu
        self.verbose = verbose_level
        self._init_model(path)

    def _init_model(self, path):
        start = time.time()
        self._model = MTCNN(path, self.use_gpu)
        end = time.time()
        print("*" * 60)
        print("*" * 60)
        if self.use_gpu:
            print('\tDetector GPU mode')
        print('\tDetector loading time:', (end - start))
        print("*" * 60)
        print("*" * 60)

    def extract_bb(self, img):
        start = time.time()
        bb_faces, points = self._model.detect_face(img)
        end = time.time()
        detection_time = end - start
        if self.verbose > 0:
            print('\tDetect :', detection_time, 's,', len(bb_faces), 'faces')
        return bb_faces, detection_time

