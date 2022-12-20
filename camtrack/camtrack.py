#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import math
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import click
import cv2
from scipy.spatial.transform import Rotation

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

def intersect_3d_2d(ids3, points3d, ids2, points2d):
    _, i_ids3, i_ids2 = np.intersect1d(ids3, ids2, return_indices=True)
    return points3d[i_ids3], points2d[i_ids2]


def intersect_indexes(ids_1, ids_2):
    _, i_ids_1, i_ids_2 = np.intersect1d(ids_1, ids_2, return_indices=True)
    return i_ids_1, i_ids_2


def triangulate_3d_point(view_mats, points_2d, intrinsic_mat):
    equations = np.empty(shape=(2 * len(points_2d), 4), dtype=float)

    for (i, (view_mat, point_2d)) in enumerate(zip(view_mats, points_2d)):
        proj = intrinsic_mat @ view_mat
        equations[i * 2] = proj[2] * point_2d[0] - proj[0]
        equations[i * 2 + 1] = proj[2] * point_2d[1] - proj[1]

    res = np.linalg.lstsq(equations[:, :3], -equations[:, 3], rcond=None)
    return res[0]


triangulation_parameters = TriangulationParameters(8.0, 0.5, 0.2)
dist_coeffs = np.zeros((4, 1))


def calc_view(frame_queue, corner_storage, known_ids, known_points, intrinsic_mat, global_inliers):

    current_frame = list(frame_queue)[0]
    max_corners = -1

    frames = list(frame_queue)
    np.random.shuffle(frames)

    for frame in frames:
        corners = corner_storage[frame]

        inliers_count = (np.isin(corners.ids, known_ids)).sum()

        if inliers_count > max_corners:
            max_corners = inliers_count
            current_frame = frame

    # print(max_corners)

    current_corners = corner_storage[current_frame]

    i_3d, i_2d = intersect_3d_2d(known_ids, known_points, current_corners.ids, current_corners.points)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(i_3d, i_2d, intrinsic_mat, None, reprojectionError=8.0, confidence=0.99,
                                      iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)

    if inliers is not None:
        global_inliers.update(inliers.flatten())

    current_view = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

    print(f"    Chosen {current_frame} frame")
    return current_frame, current_view


# def calc_view_and_remove_outliers(frame_queue, corner_storage, known_ids, known_points, intrinsic_mat, global_inliers):
#     frames = list(frame_queue)
#     np.random.shuffle(frames)
#
#     min_error = 1e10
#     res_frame = None
#     res_view = None
#
#     for frame in frames:
#         flag = np.isin(corner_storage[frame].ids.flatten(), known_ids)
#
#         corners = corner_storage[frame]
#         ids = corners.ids[flag]
#
#         i_3d, i_2d = intersect_3d_2d(known_ids, known_points, corners.ids, corners.points)
#         success, rvec, tvec, inliers = cv2.solvePnPRansac(i_3d, i_2d, intrinsic_mat, None, reprojectionError=8.0,
#                                                           confidence=0.99,
#                                                           iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)
#         view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
#
#         error = compute_reprojection_errors(i_3d, i_2d, intrinsic_mat @ view_mat).mean()
#         outliers = np.delete(np.arange(len(i_3d)), inliers)
#         outliers_ids = ids[outliers]
#         outlier_flag = np.isin(known_ids, outliers_ids)
#         known_ids = known_ids[np.logical_not(outlier_flag)]
#         known_points = known_points[np.logical_not(outlier_flag)]
#
#         if error > 10:
#             continue
#
#         global_inliers.update(ids[inliers].flatten())
#
#         if error < min_error:
#             min_error = error
#             res_frame = frame
#             res_view = view_mat
#
#     if min_error < 10.0:
#         print(f"    Chosen {res_frame} frame ...")
#         return res_frame, res_view
#     else:
#         return calc_view(frame_queue, corner_storage, known_ids, known_points, intrinsic_mat, global_inliers)



def triangulate(known_frames, known_ids, known_view_mats, corner_storage, intrinsic_mat):
    current_frame = known_frames[-1]
    current_view = known_view_mats[current_frame]
    current_corners = corner_storage[current_frame]

    tr_points3d, tr_ids, tr_cos = (None, None, None)

    angle_threshold = 1

    for frame in known_frames:
        corners = corner_storage[frame]
        view = known_view_mats[frame]

        correspondences = build_correspondences(corners, current_corners, known_ids)

        if len(correspondences.ids) < 1:
            continue

        new_tr_points3d, new_tr_ids, new_tr_cos = triangulate_correspondences(
            correspondences,
            view,
            current_view,
            intrinsic_mat,
            triangulation_parameters)

        new_tr_cos = min(1, new_tr_cos)
        angle = math.acos(new_tr_cos) * 180 / math.pi
        if tr_points3d is not None and angle < angle_threshold:
            continue

        if tr_points3d is None or len(new_tr_points3d) > len(tr_points3d):
            tr_points3d = new_tr_points3d
            tr_ids = new_tr_ids
            tr_cos = new_tr_cos

    return tr_points3d, tr_ids, tr_cos


def retriangulate(known_frames, known_ids, known_view_mats, known_points, corner_storage, intrinsic_mat):

    for (index, id) in enumerate(known_ids):
        points_2d = []
        view_mats = []

        for frame in known_frames:
            corners = corner_storage[frame]
            if id in corners.ids.flatten():
                view_mats.append(known_view_mats[frame])
                point_2d = corners.points[(corners.ids.flatten() == id)][0]
                points_2d.append(point_2d)

        if len(points_2d) >= 5:
            new_point_3d = triangulate_3d_point(view_mats, points_2d, intrinsic_mat)
            known_points[index] = new_point_3d


def recalc_views(known_frames, known_ids, known_view_mats, known_points, corner_storage, intrinsic_mat, global_inliers):
    for frame in list(known_frames):
        corners = corner_storage[frame]
        # current_ids = current_corners.ids.flatten()
        #
        # flag = np.isin(current_ids, list(global_inliers))
        # flag2 = np.isin(known_ids, current_ids[flag])
        #
        # i_3d = known_points[flag2]
        #
        # if len(i_3d) < 4:
        #     continue

        i_3d, i_2d = intersect_3d_2d(known_ids, known_points, corners.ids, corners.points)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(i_3d, i_2d,
                                    intrinsic_mat, None, reprojectionError=8.0, confidence=0.99,
                                    iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            continue

        new_view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

        known_view_mats[frame] = new_view_mat


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

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = calc_init_frames(corner_storage, intrinsic_mat)

        previous_view = np.hstack((
            known_view_2[1].r_mat,
            known_view_2[1].t_vec.reshape(-1, 1)
        ))
    else:
        previous_view = pose_to_view_mat3x4(known_view_2[1])

    preprevious_view = pose_to_view_mat3x4(known_view_1[1])

    init_id_1 = known_view_1[0]
    init_id_2 = known_view_2[0]
    print(f"\nInit frames: {init_id_1}, {init_id_2}\n")

    known_view_mats = np.full(shape=frame_count, fill_value=None)

    known_view_mats[init_id_1] = preprevious_view
    known_view_mats[init_id_2] = previous_view

    known_frames = [init_id_1, init_id_2]
    known_ids = np.empty(0, dtype=int)
    known_points = np.empty((0, 3))
    global_inliers = set()

    ############

    frame_queue = set(range(frame_count))
    frame_queue.remove(init_id_1)
    frame_queue.remove(init_id_2)

    for i in range(frame_count - 2):
        print(f"Calculating track frame: {i+1} / {frame_count-2}")

        new_tr_points3d, new_tr_ids, new_tr_cos = triangulate(known_frames, known_ids, known_view_mats, corner_storage, intrinsic_mat)

        if new_tr_points3d is not None:
            is_new = np.logical_not(np.isin(new_tr_ids, known_ids))
            known_ids = np.concatenate([known_ids, new_tr_ids[is_new]])
            known_points = np.concatenate([known_points, new_tr_points3d[is_new]])

        print(f"    Point cloud size: {len(known_ids)}")

        current_frame, current_view = calc_view(frame_queue, corner_storage, known_ids, known_points, intrinsic_mat, global_inliers)

        frame_queue.remove(current_frame)
        known_view_mats[current_frame] = current_view
        known_frames.append(current_frame)

        if i % 10 == 0:
            retriangulate(known_frames, known_ids, known_view_mats, known_points, corner_storage, intrinsic_mat)
            recalc_views(known_frames, known_ids, known_view_mats, known_points, corner_storage, intrinsic_mat, global_inliers)


    ###

    point_cloud_builder = PointCloudBuilder(known_ids,
                                            known_points)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        known_view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, known_view_mats))
    return poses, point_cloud


def calc_init_frames(corner_storage: CornerStorage, intrinsic_mat):
    frame_count = len(corner_storage)

    known_view_2 = None
    max_counts = -1
    known_view_1 = None
    step = 5
    max_angle = -1

    for count_threshold in [100, 60, 10, 5]:

        frames_found = 0

        for frame1 in range(0, frame_count, step):
            for frame2 in range(frame1+step, frame_count, step):

                _, ids1, ids2 = np.intersect1d(corner_storage[frame1].ids.flatten(), corner_storage[frame2].ids.flatten(),
                                               return_indices=True)
                pts1 = corner_storage[frame1].points[ids1]
                pts2 = corner_storage[frame2].points[ids2]

                if len(pts1) < 30:
                    break

                E, inliers = cv2.findEssentialMat(pts1, pts2, intrinsic_mat, cv2.RANSAC)

                _, hom_inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)

                if hom_inliers.sum() >= inliers.sum() * 0.7:
                    continue

                counts, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, intrinsic_mat, cv2.RANSAC)

                vec_ones = np.array([1, 0, 0])
                acos = np.dot(vec_ones, R_est @ vec_ones)
                angle = abs(np.arccos(np.clip(acos, -1.0, 1.0)) / math.pi * 180)

                if counts < count_threshold:
                    continue


                if max_angle < angle:
                    frames_found += 1
                    max_angle = angle
                    known_view_2 = (frame2, Pose(R_est, t_est.reshape(-1)))
                    known_view_1 = (frame1, Pose(np.eye(3), np.zeros(3)))
                    max_counts = counts

                # if max_counts < counts:
                #     max_counts = counts
                #     known_view_2 = (frame2, Pose(R_est, t_est.reshape(-1)))
                #     known_view_1 = (frame1, Pose(np.eye(3), np.zeros(3)))

                print(f"    at {frame1}, {frame2} angle: {angle}, max angle: {max_angle}, max_counts: {max_counts}")

        if frames_found > 0:
            break

    return known_view_1, known_view_2


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
