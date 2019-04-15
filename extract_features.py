import torch

from utils import get_args

from core.run.run_manager import RunManager
from core.models_manager.models_manager import ModelsManager


def main(args):

    models_manager = ModelsManager(detector_path=args.detectorPath,
                                   extractor_path=args.extractorPath,
                                   use_gpu=torch.cuda.is_available() and args.useGPU,
                                   training_mode=True,
                                   verbose_level=args.verboseLevel)

    run_manager = RunManager(models_manager=models_manager,
                             classifier_manager=None,
                             features_file_path=args.featuresFilePath,
                             show_video=args.showVideo,
                             verbose_level=args.verboseLevel)

    if args.extractFromFolder:
        run_manager.extract_features_from_images(args.extractPath)
    elif args.extractFromVideo:
        run_manager.extract_features_from_video(args.urlOrWebcamIdx, args.numberOfImages)
    else:
        print('Nothing selected')


if __name__ == '__main__':
    args = get_args()
    main(args=args)
