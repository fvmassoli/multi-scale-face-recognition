from core.models.detector_manager import DetectorManager
from core.models.extractor_manager import ExtractorManager


class ModelsManager(object):
    def __init__(self, detector_path, extractor_path, use_gpu, training_mode, verbose_level):
        self._detector = self._load_detector(detector_path, use_gpu, verbose_level)
        self._extractor = self._load_extractor(extractor_path, use_gpu, training_mode, verbose_level)

    def _load_detector(self, detector_path, use_gpu, verbose_level):
        return DetectorManager(path=detector_path,
                               use_gpu=use_gpu,
                               verbose_level=verbose_level)

    def _load_extractor(self, extractor_path, use_gpu, training_mode, verbose_level):
        return ExtractorManager(path=extractor_path,
                                use_gpu=use_gpu,
                                training_mode=training_mode,
                                verbose_level=verbose_level)

    def extract_bounding_boxes(self, img):
        assert self._detector is not None, "Detector not initialized"
        bb_faces, detection_time = self._detector.extract_bb(img=img)
        return bb_faces, detection_time

    def extract_features(self, img, bounding_boxes):
        assert self._extractor is not None, "Extractor not initialized"
        descriptors, bb, extraction_time = self._extractor.extract_feat(img=img, bounding_boxes=bounding_boxes)
        return descriptors, bb, extraction_time
