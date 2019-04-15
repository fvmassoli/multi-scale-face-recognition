from utils import get_args
from core.classifier.classifier_manager import ClassifierManager


def main(args):

    classifier_manager = ClassifierManager(features_file=args.featuresFilePath,
                                           classifier_path=args.classifierPath,
                                           verbose_level=args.verboseLevel)
    classifier_manager.train()


if __name__ == '__main__':
    args = get_args()
    main(args=args)
