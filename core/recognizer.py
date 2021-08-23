# -*- coding: utf-8 -*-
# @Time    : 2017/9/2 13:40
# @Author  : 郑梓斌

import json
import os

import requests
import numpy as np
import face_recognition

FACE_POINTS = list(range(0, 50))
JAW_POINTS = list(range(0, 17))
LEFT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(17, 22))
# MOUTH_POINTS = list(range(48, 72))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(42, 48))
RIGHT_BROW_POINTS = list(range(22, 27))

LEFT_FACE = list(range(0, 8)) + list(range(27, 28))
RIGHT_FACE = list(range(8, 17)) + list(range(25, 36))

JAW_END = 17
FACE_START = 0
FACE_END = 50

OVERLAY_POINTS = [
    LEFT_FACE,
    JAW_POINTS,
    RIGHT_FACE

]


def face_points(image):
    # points = []
    # txt = image + '.txt'
    #
    # if os.path.isfile(txt):
    #     with open(txt) as file:
    #         for line in file:
    #             points = line
    # elif os.path.isfile(image):
    #     # points = landmarks_by_face__(image)
    #     with open(txt, 'w') as file:
    #         file.write(str(points))

    # faces = json.loads(points)['faces']
    # print(points)
    image2 = face_recognition.load_image_file(image)
    landmark = face_recognition.face_landmarks(image2)
    matrix_list = np.matrix(matrix_marks(landmark[0]))
    point_list = []
    for p in matrix_list.tolist():
        point_list.append((int(p[0]), int(p[1])))

    return matrix_list, point_list, 200


# def landmarks_by_face__(image):
#     url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
#     params = {
#         'api_key': 'UVqp3_Es7_Gr504eGd0HQR4EWtvNC7u5',
#         'api_secret': 'PB5hI_ytmOkRIYXxEQEIwmbH0bFdqrL6',
#         'return_landmark': 1,
#     }
#     file = {'image_file': open(image, 'rb')}
#
#     r = requests.post(url=url, files=file, data=params)
#
#     if r.status_code == requests.codes.ok:
#         return r.content.decode('utf-8')
#     else:
#         return r.content


def matrix_rectangle(left, top, width, height):
    pointer = [
        (left, top),
        (left + width / 2, top),
        (left + width - 1, top),
        (left + width - 1, top + height / 2),
        (left, top + height / 2),
        (left, top + height - 1),
        (left + width / 2, top + height - 1),
        (left + width - 1, top + height - 1)
    ]

    return pointer


def matrix_marks(res):
    pointer = []

    for i in res:
        print(i, len(res.get(i)))
        if i=="top_lip" or i=="bottom_lip":
            continue
        for p in res.get(i):

            pointer.append(list(p))
    #     [res['contour_left1']['x'], res['contour_left1']['y']],
    #     [res['contour_left2']['x'], res['contour_left2']['y']],
    #     [res['contour_left3']['x'], res['contour_left3']['y']],
    #     [res['contour_left4']['x'], res['contour_left4']['y']],
    #     [res['contour_left5']['x'], res['contour_left5']['y']],
    #     [res['contour_left6']['x'], res['contour_left6']['y']],
    #     [res['contour_left7']['x'], res['contour_left7']['y']],
    #     [res['contour_left8']['x'], res['contour_left8']['y']],
    #     [res['contour_left9']['x'], res['contour_left9']['y']],
    #     [res['contour_chin']['x'], res['contour_chin']['y']],
    #     [res['contour_right9']['x'], res['contour_right9']['y']],
    #     [res['contour_right8']['x'], res['contour_right8']['y']],
    #     [res['contour_right7']['x'], res['contour_right7']['y']],
    #     [res['contour_right6']['x'], res['contour_right6']['y']],
    #     [res['contour_right5']['x'], res['contour_right5']['y']],
    #     [res['contour_right4']['x'], res['contour_right4']['y']],
    #     [res['contour_right3']['x'], res['contour_right3']['y']],
    #     [res['contour_right2']['x'], res['contour_right2']['y']],
    #     [res['contour_right1']['x'], res['contour_right1']['y']],
    #
    #     [res['left_eye_bottom']['x'], res['left_eye_bottom']['y']],
    #     [res['left_eye_center']['x'], res['left_eye_center']['y']],
    #     [res['left_eye_left_corner']['x'], res['left_eye_left_corner']['y']],
    #     [res['left_eye_lower_left_quarter']['x'], res['left_eye_lower_left_quarter']['y']],
    #     [res['left_eye_lower_right_quarter']['x'], res['left_eye_lower_right_quarter']['y']],
    #     [res['left_eye_pupil']['x'], res['left_eye_pupil']['y']],
    #     [res['left_eye_right_corner']['x'], res['left_eye_right_corner']['y']],
    #     [res['left_eye_top']['x'], res['left_eye_top']['y']],
    #     [res['left_eye_upper_left_quarter']['x'], res['left_eye_upper_left_quarter']['y']],
    #     [res['left_eye_upper_right_quarter']['x'], res['left_eye_upper_right_quarter']['y']],
    #
    #     [res['left_eyebrow_left_corner']['x'], res['left_eyebrow_left_corner']['y']],
    #     [res['left_eyebrow_upper_left_quarter']['x'], res['left_eyebrow_upper_left_quarter']['y']],
    #     [res['left_eyebrow_upper_middle']['x'], res['left_eyebrow_upper_middle']['y']],
    #     [res['left_eyebrow_upper_right_quarter']['x'], res['left_eyebrow_upper_right_quarter']['y']],
    #     [res['left_eyebrow_right_corner']['x'], res['left_eyebrow_right_corner']['y']],
    #     [res['left_eyebrow_lower_left_quarter']['x'], res['left_eyebrow_lower_left_quarter']['y']],
    #     [res['left_eyebrow_lower_middle']['x'], res['left_eyebrow_lower_middle']['y']],
    #     [res['left_eyebrow_lower_right_quarter']['x'], res['left_eyebrow_lower_right_quarter']['y']],
    #
    #     [res['mouth_left_corner']['x'], res['mouth_left_corner']['y']],
    #     [res['mouth_lower_lip_bottom']['x'], res['mouth_lower_lip_bottom']['y']],
    #     [res['mouth_lower_lip_left_contour1']['x'], res['mouth_lower_lip_left_contour1']['y']],
    #     [res['mouth_lower_lip_left_contour2']['x'], res['mouth_lower_lip_left_contour2']['y']],
    #     [res['mouth_lower_lip_left_contour3']['x'], res['mouth_lower_lip_left_contour3']['y']],
    #     [res['mouth_lower_lip_right_contour1']['x'], res['mouth_lower_lip_right_contour1']['y']],
    #     [res['mouth_lower_lip_right_contour2']['x'], res['mouth_lower_lip_right_contour2']['y']],
    #     [res['mouth_lower_lip_right_contour3']['x'], res['mouth_lower_lip_right_contour3']['y']],
    #     [res['mouth_lower_lip_top']['x'], res['mouth_lower_lip_top']['y']],
    #     [res['mouth_right_corner']['x'], res['mouth_right_corner']['y']],
    #     [res['mouth_upper_lip_bottom']['x'], res['mouth_upper_lip_bottom']['y']],
    #     [res['mouth_upper_lip_left_contour1']['x'], res['mouth_upper_lip_left_contour1']['y']],
    #     [res['mouth_upper_lip_left_contour2']['x'], res['mouth_upper_lip_left_contour2']['y']],
    #     [res['mouth_upper_lip_left_contour3']['x'], res['mouth_upper_lip_left_contour3']['y']],
    #     [res['mouth_upper_lip_right_contour1']['x'], res['mouth_upper_lip_right_contour1']['y']],
    #     [res['mouth_upper_lip_right_contour2']['x'], res['mouth_upper_lip_right_contour2']['y']],
    #     [res['mouth_upper_lip_right_contour3']['x'], res['mouth_upper_lip_right_contour3']['y']],
    #     [res['mouth_upper_lip_top']['x'], res['mouth_upper_lip_top']['y']],
    #
    #     [res['nose_contour_left1']['x'], res['nose_contour_left1']['y']],
    #     [res['nose_contour_left2']['x'], res['nose_contour_left2']['y']],
    #     [res['nose_contour_left3']['x'], res['nose_contour_left3']['y']],
    #     [res['nose_contour_lower_middle']['x'], res['nose_contour_lower_middle']['y']],
    #     [res['nose_contour_right1']['x'], res['nose_contour_right1']['y']],
    #     [res['nose_contour_right2']['x'], res['nose_contour_right2']['y']],
    #     [res['nose_contour_right3']['x'], res['nose_contour_right3']['y']],
    #     [res['nose_left']['x'], res['nose_left']['y']],
    #     [res['nose_right']['x'], res['nose_right']['y']],
    #     [res['nose_tip']['x'], res['nose_tip']['y']],
    #
    #     [res['right_eye_bottom']['x'], res['right_eye_bottom']['y']],
    #     [res['right_eye_center']['x'], res['right_eye_center']['y']],
    #     [res['right_eye_left_corner']['x'], res['right_eye_left_corner']['y']],
    #     [res['right_eye_lower_left_quarter']['x'], res['right_eye_lower_left_quarter']['y']],
    #     [res['right_eye_lower_right_quarter']['x'], res['right_eye_lower_right_quarter']['y']],
    #     [res['right_eye_pupil']['x'], res['right_eye_pupil']['y']],
    #     [res['right_eye_right_corner']['x'], res['right_eye_right_corner']['y']],
    #     [res['right_eye_top']['x'], res['right_eye_top']['y']],
    #     [res['right_eye_upper_left_quarter']['x'], res['right_eye_upper_left_quarter']['y']],
    #     [res['right_eye_upper_right_quarter']['x'], res['right_eye_upper_right_quarter']['y']],
    #
    #     [res['right_eyebrow_left_corner']['x'], res['right_eyebrow_left_corner']['y']],
    #     [res['right_eyebrow_upper_left_quarter']['x'], res['right_eyebrow_upper_left_quarter']['y']],
    #     [res['right_eyebrow_upper_middle']['x'], res['right_eyebrow_upper_middle']['y']],
    #     [res['right_eyebrow_upper_right_quarter']['x'], res['right_eyebrow_upper_right_quarter']['y']],
    #     [res['right_eyebrow_right_corner']['x'], res['right_eyebrow_right_corner']['y']],
    #     [res['right_eyebrow_lower_left_quarter']['x'], res['right_eyebrow_lower_left_quarter']['y']],
    #     [res['right_eyebrow_lower_middle']['x'], res['right_eyebrow_lower_middle']['y']],
    #     [res['right_eyebrow_lower_right_quarter']['x'], res['right_eyebrow_lower_right_quarter']['y']],
    # ]
    pointer.append(pointer[0])
    pointer.append(pointer[17])

    print("landmark", len(pointer), pointer)
    return pointer
