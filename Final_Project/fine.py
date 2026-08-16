from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass
class FineResult:
    transform: np.ndarray       # 4x4
    fitness: float              # fraction of points with a correspondence
    inlier_rmse: float          # RMSE over those correspondences
    n_iter: int                 # iterations actually performed
    elapsed: float


def icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    max_correspondence_distance: float,
    estimation: str = "point_to_point",
    max_iteration: int = 100,
    relative_fitness: float = 1e-8,
    relative_rmse: float = 1e-8,
) -> FineResult:
    # Standard ICP from Open3D, either point_to_point or point_to_plane.
    t0 = time.perf_counter()

    if estimation == "point_to_plane":
        # point-to-plane converges in far fewer iterations but needs normals
        if not target.has_normals():
            raise ValueError("point-to-plane ICP needs target normals; run preprocess first")
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif estimation == "point_to_point":
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    else:
        raise ValueError(f"unknown ICP type: {estimation}")

    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance,
        np.asarray(T_init, dtype=float), est,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=relative_fitness,
            relative_rmse=relative_rmse,
            max_iteration=max_iteration),
    )

    return FineResult(
        transform=np.asarray(result.transformation),
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        n_iter=max_iteration,
        elapsed=time.perf_counter() - t0,
    )


def iicp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    max_correspondence_distance: float,
    normal_angle_deg: float = 30.0,
    max_iteration: int = 100,
    tolerance: float = 1e-8,
) -> FineResult:
    # Improved ICP that keeps a pair only if it passes both distance and normal angle.
    t0 = time.perf_counter()

    if not source.has_normals() or not target.has_normals():
        raise ValueError("IICP needs normals on both clouds; run preprocess first")

    src = np.asarray(source.points, dtype=float)
    src_n = np.asarray(source.normals, dtype=float)
    tgt = np.asarray(target.points, dtype=float)
    tgt_n = np.asarray(target.normals, dtype=float)

    tree = cKDTree(tgt)
    cos_thresh = np.cos(np.deg2rad(normal_angle_deg))

    T = np.asarray(T_init, dtype=float).copy()
    prev_rmse = float("inf")
    fitness = 0.0
    inlier_rmse = float("nan")
    n_done = 0

    for it in range(max_iteration):
        n_done = it + 1
        moved = (np.hstack([src, np.ones((len(src), 1))]) @ T.T)[:, :3]
        moved_n = src_n @ T[:3, :3].T          # normals rotate but do not translate

        dist, idx = tree.query(moved, k=1, workers=-1)
        mask = dist < max_correspondence_distance

        # absolute value, since estimated normals may point either way along a surface
        cos = np.abs(np.einsum("ij,ij->i", moved_n, tgt_n[idx]))
        mask &= cos > cos_thresh

        n_pair = int(mask.sum())
        if n_pair < 3:                         # three pairs minimum for a rigid fit
            break

        P = moved[mask]
        Q = tgt[idx[mask]]

        # Kabsch: centre both sets, take the SVD, guard against a reflection
        pc, qc = P.mean(axis=0), Q.mean(axis=0)
        H = (P - pc).T @ (Q - qc)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

        T_step = np.eye(4)
        T_step[:3, :3] = R
        T_step[:3, 3] = qc - R @ pc
        T = T_step @ T

        inlier_rmse = float(np.sqrt(np.mean(dist[mask] ** 2)))
        fitness = n_pair / len(src)

        if abs(prev_rmse - inlier_rmse) < tolerance:
            break
        prev_rmse = inlier_rmse

    return FineResult(
        transform=T,
        fitness=fitness,
        inlier_rmse=inlier_rmse,
        n_iter=n_done,
        elapsed=time.perf_counter() - t0,
    )


def refine(
    method: str,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    max_correspondence_distance: float,
    max_iteration: int = 100,
    normal_angle_deg: float = 30.0,
) -> FineResult:
    # Entry point for fine registration: none, icp, icp_plane or iicp.
    method = method.lower()

    if method == "none":
        return FineResult(np.asarray(T_init), float("nan"), float("nan"), 0, 0.0)

    if method == "iicp":
        return iicp(source, target, T_init, max_correspondence_distance,
                    normal_angle_deg=normal_angle_deg, max_iteration=max_iteration)

    if method in ("icp", "icp_point", "point_to_point"):
        return icp(source, target, T_init, max_correspondence_distance,
                   "point_to_point", max_iteration)

    if method in ("icp_plane", "point_to_plane"):
        return icp(source, target, T_init, max_correspondence_distance,
                   "point_to_plane", max_iteration)

    raise ValueError(f"unknown fine registration method: {method}")