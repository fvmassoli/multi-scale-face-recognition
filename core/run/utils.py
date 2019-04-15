import cv2


def open_stream(video_url):
    stream = cv2.VideoCapture(video_url)
    if not stream.isOpened():
        raise IOError('Cannot read webcam or video')
    return stream, True


def draw_bounding_boxes(bb, img, matches):
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


def print_times(capture_time, detection_time, extraction_time, match_time):
    print("\t Capture time:    {}\n"
          "\t Detection time:  {}\n"
          "\t Extraction time: {}\n"
          "\t Match time:      {}".format(capture_time, detection_time, extraction_time, match_time))


def show(img):
    cv2.imshow('frame1', img)

