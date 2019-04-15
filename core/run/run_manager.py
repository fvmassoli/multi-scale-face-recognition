from __future__ import print_function

import os
import cv2
import sys
import csv
import time
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import open_stream, draw_bounding_boxes, print_times, show


class RunManager(object):
    def __init__(self, models_manager, classifier_manager, features_file_path, show_video, verbose_level):
        self._models_manager = models_manager
        self._classifier_manager = classifier_manager
        self._features_file_path = features_file_path
        self._show_video = show_video
        self._verbose_level = verbose_level

    def _read_stream(self, stream):
        start = time.time()
        ret, img = stream.read()
        end = time.time()
        capture_time = end - start
        return ret, img, capture_time

    def _extract(self, img):
        bounding_boxes, detection_time = self._models_manager.extract_bounding_boxes(img)
        descriptors, bb, extraction_time = self._models_manager.extract_features(img=img,
                                                                                 bounding_boxes=bounding_boxes)
        return bb, descriptors, detection_time, extraction_time

    # def _recognize(self):
    #     # Start stream
    #     stream, ret = self._get_stream()
    #     # Loop over the stream
    #     while ret:
    #         matches = []
    #         # Read stream
    #         ret, img, capture_time = self._read_stream(stream)
    #         # Extract bounding boxes and features
    #         bb, descriptors, detection_time, extraction_time = self._extract(img=img)
    #         if len(descriptors) != 0:
    #             start = time.time()
    #             # Classify faces
    #             person_id_metric, confidence_metric = self._classifier_manager.classify_faces(descriptors)
    #             for i in range(len(confidence_metric)):
    #                 if confidence_metric[i] < self.metric_detection_confidence:
    #                     match_metric = "{} ==> Conf = {:.2f}".format('X', confidence_metric[i])
    #                 else:
    #                     name_metric = self._names_data_frame[self._names_data_frame['label'] == person_id_metric[i]]['name']
    #                     match_metric = "{}-metric={:.2f}".format(name_metric.iloc[0], confidence_metric[i])
    #                 matches.append(match_metric)
    #             end = time.time()
    #             match_time = end - start
    #         # Draw bounding boxes
    #         if bb is not None and img is not None:
    #             self._draw_bounding_boxes(bb, img, matches)
    #         self._print_times(capture_time, detection_time, extraction_time, match_time)
    #         self._show(img=img)
    #     stream.release()
    #     cv2.destroyAllWindows()

    def extract_features_from_video(self, url_or_webcam_idx, number_of_images):
        stream, ret = open_stream(url_or_webcam_idx)
        descriptors = []
        n = 0
        while ret and n < number_of_images:
            # Read stream
            ret, img, capture_time = self._read_stream(stream)
            # Extract bounding boxes and features
            bb, descriptor, detection_time, extraction_time = self._extract(img=img)
            # If founds more than one face skip the image
            if len(bb) != 1:
                continue
            n += 1
            descriptors.append(descriptor)
            if bb is not None and len(bb) != 0:
                for i in range(len(bb)):
                    cv2.rectangle(img, bb[i][:2], bb[i][2:4], (0, 255, 0), 2)
            if self._verbose_level > 1:
                print_times(capture_time, detection_time, extraction_time, 0)
            if self._show_video:
                show(img=img)
            # Freeze 1 sec in order to get different face orientation
            time.sleep(1)
        stream.release()
        cv2.destroyAllWindows()
        print(len(descriptors))
        # self._save_features([descriptor])

    # def extract_features_from_images(self):
    #     if not os.path.isdir(self._training_image_folder):
    #         raise IOError('Root folder not found')
    #     subfolders = os.listdir(self._training_image_folder)
    #     if len(subfolders) == 0:
    #         raise IOError('Root folder is empty')
    #     images_dict = {k: os.listdir(os.path.join(self._training_image_folder, k)) for k in subfolders}
    #     for k in images_dict:
    #         features = []
    #         for img in tqdm(images_dict[k], desc='Extracting features from folder ' + k):
    #             img = cv2.imread(os.path.join(self._training_image_folder, k, img))
    #             bb, descriptor, detection_time, extraction_time = self._extract(img=img)
    #             features.append(descriptor)
    #             if bb is not None and len(bb) != 0:
    #                 for i in range(len(bb)):
    #                     cv2.rectangle(img, bb[i][:2], bb[i][2:4], (0, 255, 0), 2)
    #             self._show(img=img)
    #         self._save_features(features)
    #     cv2.destroyAllWindows()
    #
    # def _save_features(self, features):
    #     if features is None:
    #         raise Exception('Something during training went wrong! If training from image folders check thath all'
    #                         'identities have the same number of valid images (i.e. images with a detected face)')
    #     if sys.version_info[0] >= 3:
    #         person_name = input("Insert person id: ")
    #     else:
    #         person_name = raw_input("Insert person id: ")
    #     features = np.asarray(features)
    #     df = pd.read_csv(self.names_file)
    #     label = 0
    #     if len(df) != 0:
    #         label = df.index.values[-1] + 1
    #     df = df[df['name']==person_name]
    #     if len(df) != 0:
    #         label = df['label'].iloc[0]
    #     with open(self.names_file, 'a') as names:
    #         writer = csv.writer(names)
    #         writer.writerow([person_name, label])
    #     with open(self.labels_file, 'a') as lbs:
    #         np.savetxt(lbs, np.repeat(np.asarray(label, dtype=np.int), features.shape[0]), delimiter=' ')
    #     with open(self.features_file, 'a') as fts:
    #         np.savetxt(fts, np.concatenate(features), delimiter=' ')
