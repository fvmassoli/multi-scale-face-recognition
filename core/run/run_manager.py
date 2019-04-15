from __future__ import print_function

import os
import sys
import cv2
import csv
import time
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


class RunManager(object):
    def __init__(self, models_manager, classifier_manager, training_mode, extract_from_folder, extract_from_video,
                 extract_path, names_file, video_url, webcam_idx, show_video, verbose_level,
                 number_training_images,training_image_folder):
        self._models_manager = models_manager
        self._classifier_manager = classifier_manager
        self._training_mode = training_mode
        self._extract_from_folder = extract_from_folder
        self._extract_from_video = extract_from_video
        self._extract_path = extract_path
        self._training_image_folder = training_image_folder
        self._video_url = video_url
        self._webcam_idx = webcam_idx
        self._show_video = show_video
        self._verbose_level = verbose_level
        self._number_training_images = number_training_images
        self._names_data_frame = pd.read_csv(names_file)

    def _get_stream(self):
        if self._video_url is not None:
            stream = cv2.VideoCapture(self._video_url)
        else:
            stream = cv2.VideoCapture(self._webcam_idx)
        if not stream.isOpened():
            raise IOError('Cannot read webcam or video')
        return stream, True

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

    def _draw_bounding_boxes(self, bb, img, matches):
        for i in range(len(bb)):
            cv2.rectangle(img, bb[i][:2], bb[i][2:4], (0, 255, 0), 2)
            rwidth = bb[i][2] - bb[i][0]
            rheight = bb[i][3] - bb[i][1]
            reccolor = (0, 255, 0)
            if rwidth < 65 or rheight < 65:
                reccolor = (0, 0, 255)
            cv2.rectangle(img, bb[i][:2], bb[i][2:4], reccolor, 2)
            if i + 1 <= len(matches):
                (text_width, text_height) = cv2.getTextSize(matches[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness=1)[0]
                box_coords = ((int(bb[i][0]) - 4, int(bb[i][1]) - 3),
                              (int(bb[i][0]) - 4 + text_width - 2, int(bb[i][1]) - 3 - text_height - 2))
                cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
                cv2.putText(img, matches[i], (int(bb[i][0]) - 4, int(bb[i][1]) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, reccolor, 2)
            if self._show_video:
                cv2.imshow('frame1', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def _print_times(self, capture_time, detection_time, extraction_time, match_time):
        if self._verbose_level > 0:
            print("\t Capture time:    {}\n"
                  "\t Detection time:  {}\n"
                  "\t Extraction time: {}\n"
                  "\t Match time:      {}".format(capture_time, detection_time, extraction_time, match_time))

    def _show(self, img):
        if self._show_video:
            cv2.imshow('frame1', img)

    def _recognize(self):
        # Start stream
        stream, ret = self._get_stream()
        # Loop over the stream
        while ret:
            matches = []
            # Read stream
            ret, img, capture_time = self._read_stream(stream)
            # Extract bounding boxes and features
            bb, descriptors, detection_time, extraction_time = self._extract(img=img)
            if len(descriptors) != 0:
                start = time.time()
                # Classify faces
                person_id_metric, confidence_metric = self._classifier_manager.classify_faces(descriptors)
                for i in range(len(confidence_metric)):
                    if confidence_metric[i] < self.metric_detection_confidence:
                        match_metric = "{} ==> Conf = {:.2f}".format('X', confidence_metric[i])
                    else:
                        name_metric = self._names_data_frame[self._names_data_frame['label'] == person_id_metric[i]]['name']
                        match_metric = "{}-metric={:.2f}".format(name_metric.iloc[0], confidence_metric[i])
                    matches.append(match_metric)
                end = time.time()
                match_time = end - start
            # Draw bounding boxes
            if bb is not None and img is not None:
                self._draw_bounding_boxes(bb, img, matches)
            self._print_times(capture_time, detection_time, extraction_time, match_time)
            self._show(img=img)
        stream.release()
        cv2.destroyAllWindows()

    def _train_from_video(self):
        stream, ret = self._get_stream()
        n = 0
        while ret and n < self._number_training_images:
            # Read stream
            ret, img, capture_time = self._read_stream(stream)
            # Extract bounding boxes and features
            bb, descriptor, detection_time, extraction_time = self._extract(img=img)
            # If founds more than one face skip the image
            if len(bb) != 1:
                continue
            n += 1
            if bb is not None and len(bb) != 0:
                for i in range(len(bb)):
                    cv2.rectangle(img, bb[i][:2], bb[i][2:4], (0, 255, 0), 2)
            self._print_times(capture_time, detection_time, extraction_time, 0)
            self._show(img=img)
            # Freeze 1 sec in order to get different face orientation
            time.sleep(1)
        stream.release()
        cv2.destroyAllWindows()
        self._save_features([descriptor])

    def _train_from_image_folder(self):
        if not os.path.isdir(self._training_image_folder):
            raise IOError('Root folder not found')
        subfolders = os.listdir(self._training_image_folder)
        if len(subfolders) == 0:
            raise IOError('Root folder is empty')
        images_dict = {k: os.listdir(os.path.join(self._training_image_folder, k)) for k in subfolders}
        for k in images_dict:
            features = []
            for img in tqdm(images_dict[k], desc='Extracting features from folder ' + k):
                img = cv2.imread(os.path.join(self._training_image_folder, k, img))
                bb, descriptor, detection_time, extraction_time = self._extract(img=img)
                features.append(descriptor)
                if bb is not None and len(bb) != 0:
                    for i in range(len(bb)):
                        cv2.rectangle(img, bb[i][:2], bb[i][2:4], (0, 255, 0), 2)
                self._show(img=img)
            self._save_features(features)
        cv2.destroyAllWindows()

    def _save_features(self, features):
        if features is None:
            raise Exception('Something during training went wrong! If training from image folders check thath all'
                            'identities have the same number of valid images (i.e. images with a detected face)')
        if sys.version_info[0] >= 3:
            person_name = input("Insert person id: ")
        else:
            person_name = raw_input("Insert person id: ")
        features = np.asarray(features)
        df = pd.read_csv(self.names_file)
        label = 0
        if len(df) != 0:
            label = df.index.values[-1] + 1
        df = df[df['name']==person_name]
        if len(df) != 0:
            label = df['label'].iloc[0]
        with open(self.names_file, 'a') as names:
            writer = csv.writer(names)
            writer.writerow([person_name, label])
        with open(self.labels_file, 'a') as lbs:
            np.savetxt(lbs, np.repeat(np.asarray(label, dtype=np.int), features.shape[0]), delimiter=' ')
        with open(self.features_file, 'a') as fts:
            np.savetxt(fts, np.concatenate(features), delimiter=' ')

    def run(self):
        if self._training_mode and self._extract_from_folder:
            self._train_from_image_folder()
        elif self._training_mode and self._extract_from_video():
            self._train_from_video()
        else:
            self._recognize()
