# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python

import numpy as np
import imageio.v3 as iio

import cv2 as cv

import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision


# ここにあるモデルファイルが必要
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
MODEL_PATH = 'models/hand_landmarker.task'


def cv_draw_landmarks(image: np.ndarray, detected_hands):
    """ Draw landmarks on the image
    """
    h, w, _ = image.shape

    for k, lm in enumerate(detected_hands.hand_landmarks):
        for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            pt1 = (int(lm[c.start].x * w), int(lm[c.start].y * h))
            pt2 = (int(lm[c.end].x * w), int(lm[c.end].y * h))
            assert 0 <= pt1[0] < w and 0 <= pt1[1] < h
            assert 0 <= pt2[0] < w and 0 <= pt2[1] < h

            cv.line(image, pt1, pt2, color=(255, 0, 0), thickness=1)


def main():
    # Create a hand landmarker instance with the image mode:
    base_options = tasks.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,  # 必須
        running_mode=vision.RunningMode.IMAGE,
        num_hands=5,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        # 与える画像は mp.Image である必要がある
        # GPU メモリーをサポートしている
        # https://developers.google.com/mediapipe/api/solutions/python/mp/Image

        # numpy 配列から作る場合
        img = iio.imread('assets/woman_hands.jpg')
        img_mp = mp.Image(data=img, image_format=mp.ImageFormat.SRGB)

        # ファイルから作る場合
        # (その画像をほかで加工するなら、上記の方法のほうが良い？)
        # img_mp = mp.Image.create_from_file('assets/woman_hands.jpg')
        # img = img_mp.numpy_view().copy()

        # mp.Image は readonly の numpy view が作れる
        assert img.shape == img_mp.numpy_view().shape

        result = landmarker.detect(img_mp)

        # 結果 HandLandmarkerResult の中身
        # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/vision/HandLandmarkerResult
        # 3つのフィールドを持ち、その型は以下のページにある。
        # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/components/containers
        # モジュールが深いので assert は省略

        # 検出した手の分だけリストになっている
        assert type(result.hand_landmarks) is list
        assert type(result.hand_world_landmarks) is list
        assert type(result.handedness) is list
        assert len(result.hand_landmarks) == len(result.hand_world_landmarks) == len(result.handedness)

        assert type(result.hand_landmarks[0]) is list
        # hand_landmarks[idx_hand][idx_lm] の型は NormalizedLandmark
        #   x, y, z, visibility, presence を持っている

        assert type(result.hand_world_landmarks[0]) is list
        # hand_world_landmarks[idx_hand][idx_lm] の型は Landmark
        #   x, y, z, visibility, presence を持っている

        assert type(result.handedness[0]) is list
        # handedness[idx_hand] はなぜか 1 要素の list
        # handedness[idx_hand][0] の型は Category
        #   index, score, display_name, category_name を持っている

        # print('hand_landmarks', type(result.hand_landmarks), len(result.hand_landmarks), result.hand_landmarks)
        # print('hand_world_landmarks', type(result.hand_world_landmarks), len(result.hand_world_landmarks), result.hand_world_landmarks)
        # print('handedness', type(result.handedness), len(result.handedness), result.handedness)

        cv_draw_landmarks(img, result)

        cv.imshow("result", img[..., ::-1])
        cv.waitKey()


if __name__ == '__main__':
    main()
