from __future__ import annotations

import copy
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ==================================================
# Loading and preprocessing
# ==================================================
def load_point_cloud(path: str | None = None) -> o3d.geometry.PointCloud:
    # Load a point cloud, or Open3D's built-in Stanford Bunny when path is None.
    if path is None:
        mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
        pcd = o3d.geometry.PointCloud(mesh.vertices)
    else:
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:      # some .ply files are meshes, so points is empty
            mesh = o3d.io.read_triangle_mesh(path)
            pcd = o3d.geometry.PointCloud(mesh.vertices)

    if len(pcd.points) == 0:
        raise ValueError(f"no points loaded from {path}")
    return pcd


def preprocess(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius_factor: float = 2.0,
    remove_outlier: bool = False,
) -> o3d.geometry.PointCloud:
    # Downsample for speed and estimate the normals that FPFH and ICP need.
    down = pcd.voxel_down_sample(voxel_size)

    if remove_outlier and len(down.points) > 20:
        down, _ = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * normal_radius_factor, max_nn=30))
    return down


# ==================================================
# Pose encoding
# ==================================================
def params_to_matrix(params: np.ndarray) -> np.ndarray:
    # Turn tx, ty, tz, rx, ry, rz into a 4x4 matrix using Euler xyz in radians.
    params = np.asarray(params, dtype=float).ravel()
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", params[3:6]).as_matrix()
    T[:3, 3] = params[0:3]
    return T


def matrix_to_params(T: np.ndarray) -> np.ndarray:
    # Inverse of params_to_matrix.
    t = np.asarray(T)[:3, 3]
    r = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_euler("xyz")
    return np.concatenate([t, r])


def random_transform(
    max_rot_deg: float = 60.0,
    max_trans: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    # Random transform with an exact rotation magnitude about a random 3D axis.
    rng = np.random.default_rng(seed)

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = Rotation.from_rotvec(axis * np.deg2rad(max_rot_deg)).as_matrix()

    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = direction * max_trans
    return T


def make_pair(
    pcd: o3d.geometry.PointCloud,
    T_gt: np.ndarray,
    noise_sigma: float = 0.0,
    seed: int | None = None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    # Build a source and a target where target is T_gt applied to source.
    source = copy.deepcopy(pcd)
    target = copy.deepcopy(pcd).transform(T_gt)

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        pts = np.asarray(target.points)
        target.points = o3d.utility.Vector3dVector(
            pts + rng.normal(scale=noise_sigma, size=pts.shape))
    return source, target


def crop_partial(
    pcd: o3d.geometry.PointCloud, keep_ratio: float = 0.7, seed: int | None = None
) -> o3d.geometry.PointCloud:
    # Cut along a random direction to simulate partial overlap.
    rng = np.random.default_rng(seed)
    pts = np.asarray(pcd.points)

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    n_keep = max(10, int(len(pts) * keep_ratio))
    idx = np.argsort(pts @ axis)[-n_keep:]
    return pcd.select_by_index(idx.tolist())


def voxel_for_target_count(
    pcd: o3d.geometry.PointCloud,
    target_count: int,
    tol: float = 0.15,
    max_step: int = 25,
) -> float:
    # Binary search a voxel size giving roughly target_count points.
    diag = bbox_diagonal(pcd)
    lo, hi = diag * 1e-4, diag * 0.5
    best = diag * 0.02

    for _ in range(max_step):
        # geometric mean, since count falls roughly as the cube of voxel size
        mid = np.sqrt(lo * hi)
        n = len(pcd.voxel_down_sample(mid).points)
        best = mid
        if abs(n - target_count) <= tol * target_count:
            break
        if n > target_count:
            lo = mid
        else:
            hi = mid
    return best


# ==================================================
# Shared objective function
# ==================================================
class RegistrationFitness:
    # Trimmed mean nearest neighbour distance, shared by every optimizer.

    def __init__(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        trim_ratio: float = 0.8,
    ):
        self.source = np.ascontiguousarray(source_points, dtype=float)
        self.source_h = np.hstack([self.source, np.ones((len(self.source), 1))])
        self.tree = cKDTree(np.asarray(target_points, dtype=float))   # built once
        self.trim_ratio = float(trim_ratio)
        # keeping the closest fraction limits outliers and non-overlapping regions
        self.k = max(1, int(len(self.source) * self.trim_ratio))
        # compare algorithms on this, not on epochs, since cost per epoch differs
        self.eval_count = 0

    def evaluate_matrix(self, T: np.ndarray) -> float:
        moved = (self.source_h @ np.asarray(T).T)[:, :3]
        dist, _ = self.tree.query(moved, k=1, workers=-1)
        self.eval_count += 1
        # mean rather than sum, so point count does not change the scale
        return float(np.mean(np.partition(dist, self.k - 1)[: self.k]))

    def __call__(self, params: np.ndarray) -> float:
        return self.evaluate_matrix(params_to_matrix(params))

    def reset_counter(self) -> None:
        self.eval_count = 0


# ==================================================
# Metrics
# ==================================================
def bbox_diagonal(pcd: o3d.geometry.PointCloud) -> float:
    # Bounding box diagonal, used to scale size-dependent parameters.
    return float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))


def rotation_error_deg(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    # Relative rotation error in degrees.
    R_err = np.asarray(T_est)[:3, :3].T @ np.asarray(T_gt)[:3, :3]
    cos = (np.trace(R_err) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def translation_error(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    # Relative translation error, in the same units as the cloud.
    return float(np.linalg.norm(np.asarray(T_est)[:3, 3] - np.asarray(T_gt)[:3, 3]))


def gt_rmse(T_est: np.ndarray, T_gt: np.ndarray, source_points: np.ndarray) -> float:
    # RMSE over the exact correspondence, unlike the inlier RMSE ICP reports.
    src = np.asarray(source_points, dtype=float)
    src_h = np.hstack([src, np.ones((len(src), 1))])
    a = (src_h @ np.asarray(T_est).T)[:, :3]
    b = (src_h @ np.asarray(T_gt).T)[:, :3]
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def evaluate(
    T_est: np.ndarray, T_gt: np.ndarray, source_points: np.ndarray
) -> dict[str, float]:
    return {
        "RRE_deg": rotation_error_deg(T_est, T_gt),
        "RTE": translation_error(T_est, T_gt),
        "RMSE": gt_rmse(T_est, T_gt, source_points),
    }


def is_success(metrics: dict, rre_thresh: float = 5.0, rte_thresh: float = 0.01) -> bool:
    # Success test, with thresholds scaled to the size of the cloud.
    return metrics["RRE_deg"] < rre_thresh and metrics["RTE"] < rte_thresh


def chamfer_distance(
    source_points: np.ndarray,
    target_points: np.ndarray,
    T: np.ndarray | None = None,
    trim_ratio: float = 1.0,
) -> float:
    # Symmetric mean nearest neighbour distance, for when no ground truth exists.
    src = np.asarray(source_points, dtype=float)
    tgt = np.asarray(target_points, dtype=float)
    if T is not None:
        src = (np.hstack([src, np.ones((len(src), 1))]) @ np.asarray(T).T)[:, :3]

    # partial overlap adds a constant floor, so only relative comparisons hold
    def one_way(a, b):
        d, _ = cKDTree(b).query(a, k=1, workers=-1)
        k = max(1, int(len(d) * trim_ratio))
        return float(np.mean(np.partition(d, k - 1)[:k]))

    return 0.5 * (one_way(src, tgt) + one_way(tgt, src))


# ==================================================
# Stanford .conf parsing
# ==================================================
def parse_conf(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Read bmesh lines from a Stanford .conf into a filename to pose mapping.
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            parts = line.split()
            if len(parts) >= 9 and parts[0] == "bmesh":
                poses[os.path.basename(parts[1])] = (
                    np.array(parts[2:5], dtype=float),
                    np.array(parts[5:9], dtype=float))
    return poses


def _pose_matrix(t: np.ndarray, q: np.ndarray, wxyz: bool, conj: bool) -> np.ndarray:
    # Assemble a 4x4 pose under one quaternion convention.
    q = np.asarray(q, dtype=float)
    # wxyz True means the file stores w first; scipy expects x, y, z, w
    xyzw = np.array([q[1], q[2], q[3], q[0]]) if wxyz else q
    n = np.linalg.norm(xyzw)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    xyzw = xyzw / n
    if conj:
        xyzw = np.array([-xyzw[0], -xyzw[1], -xyzw[2], xyzw[3]])

    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
    T[:3, 3] = t
    return T


def gt_candidates(
    poses: dict, src_name: str, tgt_name: str
) -> list[tuple[str, np.ndarray]]:
    # All eight candidates across quaternion order, conjugation and direction.
    src_name, tgt_name = os.path.basename(src_name), os.path.basename(tgt_name)
    missing = [n for n in (src_name, tgt_name) if n not in poses]
    if missing:
        raise KeyError(f"not present in the .conf file: {missing}")

    out = []
    for wxyz in (False, True):
        for conj in (False, True):
            P_s = _pose_matrix(*poses[src_name], wxyz=wxyz, conj=conj)
            P_t = _pose_matrix(*poses[tgt_name], wxyz=wxyz, conj=conj)
            tag = f"{'wxyz' if wxyz else 'xyzw'}/{'conj' if conj else 'plain'}"
            out.append((f"{tag}/inv(Pt)@Ps", np.linalg.inv(P_t) @ P_s))
            out.append((f"{tag}/Pt@inv(Ps)", P_t @ np.linalg.inv(P_s)))
    return out


def resolve_gt(
    poses: dict,
    src_name: str,
    tgt_name: str,
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    trim_ratio: float = 0.5,
) -> tuple[np.ndarray, str, float]:
    # Pick the candidate that actually aligns the clouds, by chamfer distance.
    # a large returned chamfer means the file layout differs from what is assumed
    best = None
    for tag, T in gt_candidates(poses, src_name, tgt_name):
        ch = chamfer_distance(src_points, tgt_points, T, trim_ratio=trim_ratio)
        if best is None or ch < best[2]:
            best = (T, tag, ch)
    return best


def compose_extra_transform(
    source: o3d.geometry.PointCloud,
    T_gt: np.ndarray | None,
    angle_deg: float,
    trans: float = 0.0,
    seed: int | None = 0,
) -> tuple[o3d.geometry.PointCloud, np.ndarray | None]:
    # Apply an extra random transform to the source and update the reference.
    M = random_transform(angle_deg, trans, seed=seed)

    # rotate about the centre so the rotation itself adds no displacement
    c = source.get_center()
    T_c, T_b = np.eye(4), np.eye(4)
    T_c[:3, 3], T_b[:3, 3] = -c, c
    M = T_b @ M @ T_c

    source_new = copy.deepcopy(source).transform(M)
    # if target is T_gt times source, replacing source by M times source gives this
    T_gt_new = None if T_gt is None else T_gt @ np.linalg.inv(M)
    return source_new, T_gt_new


# ==================================================
# Output helpers
# ==================================================
SRC_COLOR = [1.00, 0.706, 0.000]   # orange, the source
TGT_COLOR = [0.00, 0.651, 0.929]   # blue, the target


def _colored(pcd, color, T=None):
    p = copy.deepcopy(pcd).paint_uniform_color(color)
    return p.transform(T) if T is not None else p


def box_table(header_lines: list[str], rows: list[str], indent: str = "") -> list[str]:
    # Wrap already aligned table lines in a closed ASCII border.
    width = max((len(x) for x in header_lines + rows), default=0)
    edge = indent + "+" + "-" * (width + 2) + "+"
    rule = indent + "|" + "-" * (width + 2) + "|"

    out = [edge]
    out += [f"{indent}| {h:<{width}} |" for h in header_lines]
    out.append(rule)
    out += [f"{indent}| {r:<{width}} |" for r in rows]
    out.append(edge)
    return out


def can_display() -> bool:
    # Whether a window can be opened, since Open3D only prints when headless.
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def show(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    T: np.ndarray | None = None,
    window_name: str = "registration",
    show_axes: bool = False,
) -> bool:
    # Show source in orange over target in blue, returning whether a window opened.
    if not can_display():
        return False

    # with T as None the clouds keep their own coordinates, showing the initial pose
    geoms = [_colored(source, SRC_COLOR, T), _colored(target, TGT_COLOR)]

    if show_axes:
        size = max(bbox_diagonal(target), 1e-6) * 0.3
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=target.get_axis_aligned_bounding_box().min_bound))

    try:
        o3d.visualization.draw_geometries(geoms, window_name=window_name)
        return True
    except Exception as e:
        print(f"  could not open a window ({type(e).__name__}: {e})", flush=True)
        return False


def save_pair_ply(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    T: np.ndarray | None,
    path: str,
) -> None:
    # Write the overlay to a .ply for CloudCompare or MeshLab.
    o3d.io.write_point_cloud(
        path, _colored(source, SRC_COLOR, T) + _colored(target, TGT_COLOR))