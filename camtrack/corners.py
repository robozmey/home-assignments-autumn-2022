#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
import matplotlib.pyplot as plt

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli,
    filter_frame_corners
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _detect_corners(img):
    points = cv2.goodFeaturesToTrack(img, 50, 0.01, 10)
    corners = FrameCorners(
        np.array(range(points.shape[0])),
        points,
        np.ones(points.shape[0]) * 20
    )
    return corners


def draw_masks(img, corners, radius=3,
               y_first=False):
    for pt in corners:
        if y_first:
            pt_tuple = tuple(pt[::-1])
        else:
            pt_tuple = tuple(pt)
        # print(pt_tuple)
        cv2.circle(img, (int(pt_tuple[0]), int(pt_tuple[1])), radius, 0, -1)
    return img



# python3 corners.py "/home/vladimir/env/dataset/fox_head_short/rgb/*.jpg" --show
# python3 testrunner.py "/home/vladimir/env/dataset/dataset_ha1.yml" .

def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    maxCorners=200
    minDistance=20
    qualityLevel = 0.01
    blockSize=7
    size_scale = 20
    size_offset = 10


    def calc_corners():
        eigen_vals = cv2.cornerMinEigenVal(np.uint8(image_1 * 255), blockSize=blockSize, ksize=3)

    feature_params = dict(maxCorners=maxCorners,
                          qualityLevel=qualityLevel,
                          minDistance=minDistance,
                          blockSize=blockSize)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))
    image_0 = frame_sequence[0]

    p0 = cv2.goodFeaturesToTrack(image_0, mask=None, **feature_params)
    eigen_vals = cv2.cornerMinEigenVal(np.uint8(image_0 * 255), blockSize=blockSize, ksize=3)
    f_max_corner = eigen_vals.max()
    quality = eigen_vals / f_max_corner

    corners = FrameCorners(
        np.array(range(p0.shape[0])),
        p0,
        np.apply_along_axis(
            lambda c: eigen_vals[int(c[1])][int(c[0])], 1, p0.reshape(-1, 2)) / f_max_corner * size_scale + size_offset
    )
    max_id = p0.shape[0]

    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        print(f"Calculating corners frame: {frame} / {len(frame_sequence[1:])}")

        p1, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255), np.uint8(image_1 * 255), p0, None, **lk_params)

        corners = FrameCorners(
            corners.ids,
            p1,
            corners.sizes
        )

        eigen_vals = cv2.cornerMinEigenVal(np.uint8(image_1 * 255), blockSize=blockSize, ksize=3)
        max_corner = eigen_vals.max()
        # print(max_corner)
        quality = eigen_vals / max_corner

        corners = filter_frame_corners(corners, (st==1).reshape(st.shape[0]))
        corners = filter_frame_corners(corners, np.apply_along_axis(
            lambda c: c[0] < image_1.shape[1] and c[1] < image_1.shape[0] and c[0] >= 0 and c[1] >= 0, 1, corners.points))

        corners = filter_frame_corners(corners, np.apply_along_axis(lambda c: eigen_vals[int(c[1])][int(c[0])] > qualityLevel * max_corner, 1, corners.points))

        new_count = maxCorners - corners.ids.shape[0]
        if new_count > 0:
            feature_params = dict(maxCorners=new_count,
                                  qualityLevel=qualityLevel,
                                  minDistance=minDistance,
                                  blockSize=7)

            mask = draw_masks(np.uint8(image_1 * 0 + 255), corners.points, radius=minDistance)

            # plt.imshow(draw_masks(image_0, corners.points, radius=minDistance))
            # plt.show()

            ap = cv2.goodFeaturesToTrack(image_1, mask=mask, **feature_params)

            if ap is not None:
                # print(ap.reshape(-1, 2))
                corners = FrameCorners(
                    np.concatenate([corners.ids.flatten(), np.array(range(max_id, max_id + ap.shape[0]))]),
                    np.concatenate([corners.points.reshape(-1, 2), ap.reshape(-1, 2)]),
                    np.concatenate([corners.sizes.flatten(),
                                    np.apply_along_axis(
                                        lambda c: eigen_vals[int(c[1])][int(c[0])], 1, ap.reshape(-1, 2)).flatten()
                                    / f_max_corner * size_scale + size_offset])
                )
                max_id = max_id + ap.shape[0]

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1
        p0 = corners.points

        # corners = _calculate_corners(image_0)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
