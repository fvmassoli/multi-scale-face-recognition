from __future__ import print_function

import torch
import torchvision.transforms as t

import time
from sklearn.preprocessing import normalize


class ExtractorManager(object):
    def __init__(self, path, use_gpu, training_mode, verbose_level):
        self.use_gpu = use_gpu
        self.verbose = verbose_level
        self._training_mode = training_mode
        self._model = self._init_model(path)
        self._transforms = self._init_transforms()

    def _init_model(self, path):
        start = time.time()
        model = torch.load('senet50_ft_pytorch.pth')
        ckp = torch.load(path, map_location='cpu')
        for n, p in model.named_parameters():
            p.data = ckp['model_state_dict'][n]
        model.eval()
        end = time.time()
        print("*" * 60)
        print("*" * 60)
        if self.use_gpu:
            model.cuda()
            print('\tExtractor GPU mode')
        print('\tExtractor loading time:', (end - start))
        print("*" * 60)
        print("*" * 60)
        return model

    def _subtract_mean(self, x):
        mean_vector = [91.4953, 103.8827, 131.0912]
        x *= 255.
        x[0] -= mean_vector[0]
        x[1] -= mean_vector[1]
        x[2] -= mean_vector[2]
        return x

    def _init_transforms(self):
        return t.Compose([
            t.Resize(),
            t.CenterCrop(),
            t.ToTensor(),
            t.Lambda(lambda x: self._subtract_mean(x))
        ])

    def extract_feat(self, img, bounding_boxes):
        bb = []
        descriptors = []
        extraction_time = 0
        h, w, _ = img.shape

        start = time.time()

        for face in bounding_boxes:

            # while extracting features to fill the database,
            # only a single face per image is allowed
            if len(bounding_boxes) != 1 and self._training_mode:
                break

            (startX, startY, endX, endY) = face
            da = (endX-startX)*0.3*0.5
            db = (endY-startY)*0.3*0.5
            startX = int(max(1, startX-da))
            endX = int(min(w, endX+da))
            startY = int(max(1, startY-db))
            endY = int(min(h, endY+db))

            bb.append((startX, startY, endX, endY))
            face = img[startY:endY, startX:endX]

            if face.size == 0:
                return None, None

            if face.size != 0:

                face_ = self._transforms(face)
                if self.use_gpu:
                    face_ = face_.cuda(non_blocking=True)
                descriptor = self.model(face_)

                if self.verbose > 0:
                    print('\tExtract:', extraction_time, 's')
                descriptor = normalize(descriptor.reshape(1, -1))

                descriptors.append(descriptor)

        end = time.time()
        extraction_time = start+end

        return descriptors, bb, extraction_time
