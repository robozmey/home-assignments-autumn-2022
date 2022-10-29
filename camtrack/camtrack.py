#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

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

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    init_id_0 = known_view_1[0]
    init_id_1 = known_view_2[0]

    corners_0 = corner_storage[init_id_0]
    corners_1 = corner_storage[init_id_1]

    previous_view = known_view_2[1]
    previous_corners = corners_1

    ###

    correspondences = build_correspondences(corners_0, corners_1)

    triangulation_parameters = TriangulationParameters(1, 0, 0)

    tr_points3d, tr_ids, tr_cos = triangulate_correspondences(
        correspondences,
        pose_to_view_mat3x4(known_view_1[1]),
        pose_to_view_mat3x4(known_view_2[1]),
        intrinsic_mat,
        triangulation_parameters)

    def intersect_3d_2d(ids3, points3d, ids2, points2d):
        _, i_ids3, i_ids2 = np.intersect1d(ids3, ids2, return_indices=True)
        return points3d[i_ids3], points2d[i_ids2]

    def intersect_indexes(ids_1, ids_2):
        _, i_ids_1, i_ids_2 = np.intersect1d(ids_1, ids_2, return_indices=True)
        return i_ids_1, i_ids_2

    #########

    frame_count = len(corner_storage)
    frame_order = np.array(range(0, frame_count))

    np.random.shuffle(frame_order)

    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count

    view_mats[init_id_0] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[init_id_1] = pose_to_view_mat3x4(known_view_2[1])

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

            print(outliers_ids, inliers)

            current_view = view_mat3x4_to_pose(rodrigues_and_translation_to_view_mat3x4(rvec, tvec))

            # add new 3D points
            correspondences = build_correspondences(previous_corners, current_corners, outliers_ids)
            new_tr_points3d, new_tr_ids, new_tr_cos = triangulate_correspondences(
                correspondences,
                pose_to_view_mat3x4(previous_view),
                pose_to_view_mat3x4(current_view),
                intrinsic_mat,
                triangulation_parameters)

            is_new = np.logical_not(np.isin(new_tr_ids, tr_ids))

            tr_ids = np.concatenate([tr_ids, new_tr_ids[is_new]])
            tr_points3d = np.concatenate([tr_points3d, new_tr_points3d[is_new]])

            # end of iteration
            view_mats[current_id] = pose_to_view_mat3x4(current_view)

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
