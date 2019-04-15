import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Deep Face Verifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-detp', '--detectorPath', default='./core/models_weights/detector_models',
                        help='Path to detector_models directory (default: ./core/models_weights/detector_model)')
    parser.add_argument('-extp', '--extractorPath', default='./core/models_weights/extractor_model/senet50_ft_pytorch.pth',
                        help='Path to extractor directory '
                             '(default: ./core/models_weights/extractor_model/senet50_ft_pytorch.pth)')
    parser.add_argument('-ef', '--extractFromFolder', action='store_true', help='Extract from folder (default: False)')
    parser.add_argument('-ev', '--extractFromVideo', action='store_true', help='Extract from video (default: False)')
    parser.add_argument('-n', '--numberOfImages', type=int, default=10, help='NUmber of images (default: 10)')
    parser.add_argument('-ep', '--extractPath', help='Path to images from which extract features (default: False)')
    parser.add_argument('-url', '--urlOrWebcamIdx', default=None,
                        help='Url of streaming, path to a video or index of webcam (default: None)')
    parser.add_argument('-f', '--featuresFilePath', help='Features file path')
    # parser.add_argument('-clfp', '--classifierCheckpointPath', default='./classifier_ckt',
    #                     help='Path to store/load classifier checkpoint (default: ./classifier_ckt)')
    #
    #
    # # parser.add_argument('-', '--', action=, default=, help=' (default: )')
    #
    # parser.add_argument('-tc', '--trainClassifier', action='store_true', help='Train classifier (default: False)')
    # parser.add_argument('-clfp', '--classifierPath', help='Path to classifier')
    # parser.add_argument('-t', '--trainingMode', action='store_true', help='Training mode (default: False)')
    #
    # parser.add_argument('-l', '--labelsFilePath', help='Labels file path')
    # parser.add_argument('-n', '--namesFilePath', help='Names file path')
    # parser.add_argument('-w', '--webcamIdx', default=0, type=int, help='Activate the webcam at the specified index (default: 0)')
    # parser.add_argument('-d', '--trainingImageFolder', help='Path to the root directory containing images for training.'
    #                                                         ' The folder structure has to be: '
    #                                                         'root/identity_one/images... '
    #                                                         'All the identities have to have 10 images.')

    parser.add_argument('-gpu', '--useGPU', action='store_true', help='Use GPU (default: False)')
    parser.add_argument('-sv', '--showVideo', action='store_true', help='Show the video stream (default: false)')
    parser.add_argument('-v', '--verboseLevel', default=0, type=int, choices=[0, 1],
                        help='Set verbosity level (default: 0)')
    return parser.parse_args()


def check_files(training_mode, features_file, labels_file, names_file):
    ff = os.path.exists(features_file)
    lf = os.path.exists(labels_file)
    nf = os.path.exists(names_file)
    # In recognition mode all files have to bu supplied
    if not training_mode:
        assert ff, "Features file not found"
        assert lf, "Labels file not found"
        assert nf, "Names file not found"
    else:
        if not ff:
            os.makedirs(ff)
            print("Created features file: {}".format(ff))
        if not ff:
            os.makedirs(lf)
            print("Created labels file: {}".format(lf))
        if not ff:
            os.makedirs(nf)
            print("Created names file: {}".format(nf))


# def create_log_file(current_time, base='../', train=False):
#
#     if not train:
#         dir = os.path.join(base, "recognition_logs")
#         img_dir = os.path.join(base, 'recognition_log_imgs_' + current_time)
#         file_name = os.path.join(base, 'recognition_log_' + current_time + '.txt')
#         NOFACE_LOG_FILE_DIR = os.path.join(base, 'norecognition_logs')
#         if not os.path.isdir(NOFACE_LOG_FILE_DIR):
#             os.makedirs(NOFACE_LOG_FILE_DIR)
#     else:
#         dir = os.path.join(base, 'training_logs')
#         img_dir = os.path.join(base, 'training_log_imgs_' + current_time)
#         file_name = os.path.join(base, 'training_log_' + current_time + '.txt')
#
#     wd = os.path.dirname(os.path.realpath(__file__))
#     # Creates logs run folder
#     log_dir = os.path.join(wd, dir)
#     if not os.path.isdir(log_dir):
#         os.mkdir(log_dir)
#     # Creates logs/<current_time> folder
#     current_time_dir = os.path.join(log_dir, current_time)
#     if not os.path.isdir(current_time_dir):
#         os.mkdir(current_time_dir)
#     # Creates logs/<current_time>/img/<current_date_imgs_folder> folder
#     img_dir = os.path.join(current_time_dir, img_dir)
#     if not os.path.isdir(img_dir):
#         os.mkdir(img_dir)
#     # Creates logs/<current_time>/txt/<current_date_txt> file
#     log_file = os.path.join(current_time_dir, file_name)
#     if not os.path.exists(log_file):
#         f = open(log_file, 'w')
#         f.close
#     return log_file, img_dir
