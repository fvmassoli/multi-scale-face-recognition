from __future__ import print_function

from utils import get_args, check_files
from core.run.run_manager import RunManager
from core.models_manager.models_manager import ModelsManager
from core.classifier.classifier_manager import ClassifierManager

import torch


def main(args):

    use_gpu = torch.cuda.is_available() and args.useGPU

    check_files(training_mode=args.trainingMode,
                features_file=args.featuresFilePath,
                labels_file=args.labelsFilePath,
                names_file=args.namesFilePath)

    classifier_manager = ClassifierManager(features_file=args.featuresFilePath,
                                           labels_file=args.labelsFilePath,
                                           train_classifier=args.trainClassifier,
                                           classifier_path=args.classifierPath,
                                           verbose_level=args.verboseLevel)

    models_manager = ModelsManager(detector_path=args.detectorPath,
                                   extractor_path=args.extractorPath,
                                   use_gpu=use_gpu,
                                   training_mode=args.trainingMode,
                                   verbose_level=args.verboseLevel)

    run_manager = RunManager(models_manager=models_manager,
                             classifier_manager=classifier_manager,
                             training_mode=args.trainingMode,
                             extract_from_folder=args.extractFromFolder,
                             extract_from_video=args.extractFromVideo,
                             extract_path=args.extractPath,
                             names_file=args.namesFilePath,
                             video_url=args.videoUrl,
                             webcam_idx=args.webcamIdx,
                             show_video=args.showVideo,
                             verbose_level=args.verboseLevel,
                             training_image_folder=args.trainingImageFolder)
    run_manager.run()


if __name__ == '__main__':
    args = get_args()
    main(args=args)
