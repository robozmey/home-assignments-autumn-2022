#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import click
import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,

    Correspondences,
    TriangulationParameters,
    triangulate_correspondences,
    build_correspondences,
    project_points,
    compute_reprojection_errors,
    rodrigues_and_translation_to_view_mat3x4,
)

def calc_init_frames(corner_storage: CornerStorage, intrinsic_mat):
    frame_count = len(corner_storage)

    known_view_2 = None
    max_counts = -1
    known_view_1 = None
    step = 5

    for frame1 in range(0, frame_count, step):
        for frame2 in range(frame1+step, frame_count, step):

            _, ids1, ids2 = np.intersect1d(corner_storage[frame1].ids.flatten(), corner_storage[frame2].ids.flatten(),
                                           return_indices=True)
            pts1 = corner_storage[frame1].points[ids1]
            pts2 = corner_storage[frame2].points[ids2]

            if len(pts1) < 50:
                break

            E, inliers = cv2.findEssentialMat(pts1, pts2, intrinsic_mat, cv2.RANSAC)

            _, hom_inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)

            if hom_inliers.sum() >= inliers.sum() * 0.7:
                continue

            counts, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, intrinsic_mat, cv2.RANSAC)

            # print(counts)

            if max_counts < counts:
                max_counts = counts
                known_view_2 = (frame2, Pose(R_est, t_est.reshape(-1)))
                known_view_1 = (frame1, Pose(np.eye(3), np.zeros(3)))

    return known_view_1, known_view_2


def intersect_3d_2d(ids3, points3d, ids2, points2d):
    _, i_ids3, i_ids2 = np.intersect1d(ids3, ids2, return_indices=True)
    return points3d[i_ids3], points2d[i_ids2]


def intersect_indexes(ids_1, ids_2):
    _, i_ids_1, i_ids_2 = np.intersect1d(ids_1, ids_2, return_indices=True)
    return i_ids_1, i_ids_2


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    frame_count = len(corner_storage)
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    triangulation_parameters = TriangulationParameters(1, 0, 0)

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = calc_init_frames(corner_storage, intrinsic_mat)

        previous_view = np.hstack((
            known_view_2[1].r_mat,
            known_view_2[1].t_vec.reshape(-1, 1)
        ))
    else:
        previous_view = pose_to_view_mat3x4(known_view_2[1])

    preprevious_view = pose_to_view_mat3x4(known_view_1[1])

    init_id_0 = known_view_1[0]
    init_id_1 = known_view_2[0]

    view_mats = np.full(shape=frame_count, fill_value=None)

    view_mats[init_id_0] = preprevious_view
    view_mats[init_id_1] = previous_view

    ###

    # print('frames: ', known_view_1[0], known_view_2[0])
    # print('1r', known_view_1[1].r_mat)
    # print('2r', known_view_2[1].r_mat @ np.linalg.inv(known_view_1[1].r_mat))
    # print('1t', known_view_1[1].t_vec)
    # print('2t', (known_view_2[1].t_vec - known_view_1[1].t_vec) / np.linalg.norm(known_view_2[1].t_vec - known_view_1[1].t_vec))

    corners_0 = corner_storage[init_id_0]
    corners_1 = corner_storage[init_id_1]

    previous_corners = corners_1

    correspondences = build_correspondences(corners_0, corners_1)

    tr_points3d, tr_ids, tr_cos = triangulate_correspondences(
        correspondences,
        view_mats[init_id_0],
        view_mats[init_id_1],
        intrinsic_mat,
        triangulation_parameters)

    #########

    frame_order = np.array(range(0, frame_count))

    np.random.shuffle(frame_order)


    with click.progressbar(frame_order,
                           label='Calculating view_mats',
                           length=len(frame_order)) as progress_bar:
        for current_frame in progress_bar:
            if current_frame in [init_id_0, init_id_1]:
                continue

            current_id = current_frame

            current_corners = corner_storage[current_id]

            i_3d, i_2d = intersect_3d_2d(tr_ids, tr_points3d, current_corners.ids, current_corners.points)

            dist_coeffs = np.zeros((4, 1))
            success, rvec, tvec, inliers = cv2.solvePnPRansac(i_3d, i_2d, intrinsic_mat, distCoeffs=dist_coeffs)

            if inliers is not None:
                i_id_1, i_id_2 = intersect_indexes(tr_ids, current_corners.ids)

                outliers_ids_1 = np.delete(tr_ids, i_id_1[inliers])
                outliers_ids_2 = np.delete(current_corners.ids, i_id_2[inliers])

                outliers_ids = np.union1d(outliers_ids_1, outliers_ids_2)
            else:
                outliers_ids = None

            current_view = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

            # add new 3D points
            correspondences = build_correspondences(previous_corners, current_corners, outliers_ids)
            new_tr_points3d, new_tr_ids, new_tr_cos = triangulate_correspondences(
                correspondences,
                previous_view,
                current_view,
                intrinsic_mat,
                triangulation_parameters)

            is_new = np.logical_not(np.isin(new_tr_ids, tr_ids))

            tr_ids = np.concatenate([tr_ids, new_tr_ids[is_new]])
            tr_points3d = np.concatenate([tr_points3d, new_tr_points3d[is_new]])

            # end of iteration
            view_mats[current_id] = current_view

            previous_view = current_view
            previous_corners = current_corners

    ###

    point_cloud_builder = PointCloudBuilder(tr_ids,
                                            tr_points3d)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
