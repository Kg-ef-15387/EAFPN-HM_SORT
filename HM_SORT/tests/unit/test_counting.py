import os
import cv2
import numpy as np


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_bbox_with_label(image, bbox, id, confidence, font_scale=2, thickness=2, show_con=False):
    x_min, y_min, x_max, y_max = bbox
    if show_con:
        label_text = f"ID{id}:{confidence:.2f}"
    else:
        label_text = '%d' % id

    color = compute_color_for_labels(id)
    t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)
    cv2.rectangle(image, (x_min, y_min), (x_min + t_size[0] + 3, y_min + t_size[1] + 4), color, -1)
    cv2.putText(image, label_text, (x_min, y_min + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return image


def Counting(model, track, video_path, out_path, match_hit=5):
    capture = cv2.VideoCapture(video_path)
    video_writer = cv2.VideoWriter(os.path.join(out_path, 'result.mp4'), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))

    output = []
    current_id_list = []
    caomei = []
    hei = []
    qianxi = []
    pubu = []
    tomato_number = len(current_id_list)
    cls_number = [caomei, hei, qianxi, pubu]

    while True:
        _, im = capture.read()
        if im is None:
            break

        detection_result = np.array(model(im)[0].boxes.data.to('cpu'))
        result = track.update(dets=detection_result, img=im)

        text = "Tomato Number: {}".format(tomato_number)

        output.append(text + '\n')

        org = (10, 50)  # (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 3

        cv2.putText(im, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        for t in track.active_tracks:
            track_id = t.id
            track_cls = int(t.cls)

            if track_id not in current_id_list and t.hit_streak >= match_hit:
                current_id_list.append(track_id)
                cls_number[track_cls].append(track_cls)

        tomato_number = len(current_id_list)

        for i in result:
            bbox = i[0:4].tolist()
            bbox = [int(x) for x in bbox]
            ID = int(i[4:5][0])
            con = i[5:6][0]

            result_image = draw_bbox_with_label(im, bbox, ID, con)

        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        print(result.shape)
        if result.shape[0] != 0:
            cv2.imshow("Object Detection", result_image)
            video_writer.write(result_image)
        else:
            cv2.imshow("Object Detection", im)
            video_writer.write(im)
        cv2.waitKey(1)

    with open(os.path.join(out_path, 'output.txt'), 'w', encoding='utf-8') as outfile:
        for line in output:
            outfile.write(line + '\n')
