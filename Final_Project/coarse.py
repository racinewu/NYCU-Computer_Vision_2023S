from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import open3d as o3d

from utils import RegistrationFitness, params_to_matrix


# ==================================================
# Common types and helpers
# ==================================================
@dataclass
class CoarseResult:
    transform: np.ndarray                               # 4x4
    fitness: float = float("nan")                       # objective value, lower is better
    n_eval: int = 0                                     # fitness evaluations used
    elapsed: float = 0.0                                # seconds
    history: list[float] = field(default_factory=list)  # best value per epoch


def centroid_align(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    # Translate the source centroid onto the target centroid.
    # This frees the search to spend its budget on rotation, the hard part
    T = np.eye(4)
    T[:3, 3] = np.mean(target_points, axis=0) - np.mean(source_points, axis=0)
    return T


def default_bounds(scale: float = 0.2) -> np.ndarray:
    # Search bounds of shape (6, 2), with all rotation axes spanning a full turn.
    # scale can be small once centroid alignment has removed most of the offset
    return np.array([[-scale, scale]] * 3 + [[-np.pi, np.pi]] * 3)


def _init_population(bounds: np.ndarray, pop_size: int, rng) -> np.ndarray:
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, len(bounds)))


# ==================================================
# Grey Wolf Optimizer
# ==================================================
def gwo(
    fitness: RegistrationFitness,
    bounds: np.ndarray,
    pop_size: int = 30,
    max_iter: int = 200,
    seed: int | None = None,
) -> CoarseResult:
    # Grey wolf optimizer, steered by the average of the alpha, beta and delta leaders.
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    fitness.reset_counter()

    dim = len(bounds)
    lo, hi = bounds[:, 0], bounds[:, 1]

    pos = _init_population(bounds, pop_size, rng)
    score = np.array([fitness(p) for p in pos])

    order = np.argsort(score)
    alpha, beta, delta = pos[order[0]].copy(), pos[order[1]].copy(), pos[order[2]].copy()
    alpha_score = score[order[0]]
    history = [alpha_score]

    for it in range(max_iter):
        # a falls linearly from two to zero, trading exploration for exploitation
        a = 2.0 - 2.0 * it / max_iter

        for i in range(pop_size):
            new = np.zeros(dim)
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(dim), rng.random(dim)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                new += leader - A * np.abs(C * leader - pos[i])
            pos[i] = np.clip(new / 3.0, lo, hi)

        score = np.array([fitness(p) for p in pos])
        order = np.argsort(score)

        if score[order[0]] < alpha_score:       # elitism, replace only on improvement
            alpha_score = score[order[0]]
            alpha = pos[order[0]].copy()
        beta = pos[order[1]].copy()
        delta = pos[order[2]].copy()
        history.append(alpha_score)

    return CoarseResult(
        transform=params_to_matrix(alpha),
        fitness=float(alpha_score),
        n_eval=fitness.eval_count,
        elapsed=time.perf_counter() - t0,
        history=history,
    )


# ==================================================
# Whale Optimization Algorithm
# ==================================================
def woa(
    fitness: RegistrationFitness,
    bounds: np.ndarray,
    pop_size: int = 30,
    max_iter: int = 200,
    seed: int | None = None,
    b: float = 1.0,
) -> CoarseResult:
    # Whale optimization algorithm with spiral, encircling and random search moves.
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    fitness.reset_counter()

    dim = len(bounds)
    lo, hi = bounds[:, 0], bounds[:, 1]

    pos = _init_population(bounds, pop_size, rng)
    score = np.array([fitness(p) for p in pos])

    best_idx = int(np.argmin(score))
    best, best_score = pos[best_idx].copy(), float(score[best_idx])
    history = [best_score]

    for it in range(max_iter):
        a = 2.0 - 2.0 * it / max_iter          # two down to zero
        a2 = -1.0 - it / max_iter              # minus one down to minus two, for l

        for i in range(pop_size):
            r1, r2 = rng.random(dim), rng.random(dim)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = rng.random()
            l = (a2 - 1.0) * rng.random() + 1.0

            if p < 0.5:
                if np.mean(np.abs(A)) < 1.0:
                    new = best - A * np.abs(C * best - pos[i])          # encircle
                else:
                    rand = pos[rng.integers(pop_size)]                  # explore
                    new = rand - A * np.abs(C * rand - pos[i])
            else:
                new = (np.abs(best - pos[i]) * np.exp(b * l)            # spiral
                       * np.cos(2.0 * np.pi * l) + best)
            pos[i] = np.clip(new, lo, hi)

        score = np.array([fitness(p) for p in pos])
        idx = int(np.argmin(score))
        if score[idx] < best_score:
            best_score = float(score[idx])
            best = pos[idx].copy()
        history.append(best_score)

    return CoarseResult(
        transform=params_to_matrix(best),
        fitness=best_score,
        n_eval=fitness.eval_count,
        elapsed=time.perf_counter() - t0,
        history=history,
    )


# ==================================================
# Improved Whale Optimization Algorithm
# ==================================================
def _tent_chaos(n: int, dim: int, rng, mu: float = 0.7) -> np.ndarray:
    # Tent chaotic map, spreading over zero to one more evenly than uniform sampling.
    seq = np.empty((n, dim))
    x = rng.uniform(0.05, 0.95, size=dim)
    for i in range(n):
        x = np.where(x < mu, x / mu, (1.0 - x) / (1.0 - mu))
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        seq[i] = x
    return seq


def iwoa(
    fitness: RegistrationFitness,
    bounds: np.ndarray,
    pop_size: int = 30,
    max_iter: int = 200,
    seed: int | None = None,
    b: float = 1.0,
    F: float = 0.5,
    CR: float = 0.5,
) -> CoarseResult:
    # WOA with chaotic init, a non-linear convergence factor and a DE operator.
    # all three changes target the same weakness, namely limited exploration
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    fitness.reset_counter()

    dim = len(bounds)
    lo, hi = bounds[:, 0], bounds[:, 1]

    # change one, chaotic candidates plus their opposites, keeping the better half
    cand = lo + _tent_chaos(pop_size, dim, rng) * (hi - lo)
    allc = np.vstack([cand, lo + hi - cand])
    alls = np.array([fitness(p) for p in allc])
    keep = np.argsort(alls)[:pop_size]
    pos, score = allc[keep].copy(), alls[keep].copy()

    best_idx = int(np.argmin(score))
    best, best_score = pos[best_idx].copy(), float(score[best_idx])
    history = [best_score]

    for it in range(max_iter):
        # change two, a falls slowly early and quickly late, prolonging exploration
        a = 2.0 * np.cos(np.pi / 2.0 * it / max_iter)
        a2 = -1.0 - it / max_iter

        for i in range(pop_size):               # the WOA body itself is unchanged
            r1, r2 = rng.random(dim), rng.random(dim)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = rng.random()
            l = (a2 - 1.0) * rng.random() + 1.0

            if p < 0.5:
                if np.mean(np.abs(A)) < 1.0:
                    new = best - A * np.abs(C * best - pos[i])
                else:
                    rand = pos[rng.integers(pop_size)]
                    new = rand - A * np.abs(C * rand - pos[i])
            else:
                new = (np.abs(best - pos[i]) * np.exp(b * l)
                       * np.cos(2.0 * np.pi * l) + best)
            pos[i] = np.clip(new, lo, hi)

        score = np.array([fitness(p) for p in pos])

        # change three, a DE trial vector gives a route out of a local optimum
        for i in range(pop_size):
            if rng.random() >= CR:
                continue
            r = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)
            mutant = pos[r[0]] + F * (pos[r[1]] - pos[r[2]])
            cross = rng.random(dim) < CR
            if not cross.any():
                cross[rng.integers(dim)] = True        # inherit at least one dimension
            trial = np.clip(np.where(cross, mutant, pos[i]), lo, hi)

            trial_score = fitness(trial)
            if trial_score < score[i]:                 # greedy selection
                pos[i], score[i] = trial, trial_score

        idx = int(np.argmin(score))
        if score[idx] < best_score:
            best_score = float(score[idx])
            best = pos[idx].copy()
        history.append(best_score)

    return CoarseResult(
        transform=params_to_matrix(best),
        fitness=best_score,
        n_eval=fitness.eval_count,
        elapsed=time.perf_counter() - t0,
        history=history,
    )


# ==================================================
# FPFH with RANSAC, the classical feature based baseline
# ==================================================
def compute_fpfh(
    pcd: o3d.geometry.PointCloud, voxel_size: float, radius_factor: float = 5.0
) -> o3d.pipelines.registration.Feature:
    # FPFH descriptors, with the radius conventionally five times the voxel size.
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * radius_factor, max_nn=100))


def fpfh_ransac(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    voxel_size: float,
    seed: int | None = None,
    max_iteration: int = 100_000,
    confidence: float = 0.999,
) -> CoarseResult:
    # FPFH feature matching followed by RANSAC.
    # note this uses normals and descriptors, more information than a pure search
    t0 = time.perf_counter()

    if seed is not None:
        o3d.utility.random.seed(int(seed))

    src_fpfh = compute_fpfh(source_down, voxel_size)
    tgt_fpfh = compute_fpfh(target_down, voxel_size)
    dist_thresh = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration, confidence),
    )

    return CoarseResult(
        transform=np.asarray(result.transformation),
        fitness=float(result.inlier_rmse),
        # RANSAC scores internally, so zero means not applicable, not free
        n_eval=0,
        elapsed=time.perf_counter() - t0,
    )


# ==================================================
# Entry point
# ==================================================
OPTIMIZERS = {"gwo": gwo, "woa": woa, "iwoa": iwoa}

# hybrid methods: RANSAC finds a coarse solution, then a search refines nearby
HYBRID = {"ransac_gwo": "gwo", "ransac_woa": "woa", "ransac_iwoa": "iwoa"}


def iters_for_budget(method: str, pop_size: int, budget: int, CR: float = 0.5) -> int:
    # Epochs that keep a method inside a fitness evaluation budget.
    # IWOA also evaluates opposites and trial vectors, so equal epochs is not equal cost
    base = HYBRID.get(method.lower(), method.lower())
    if base == "iwoa":
        return max(1, int((budget - 2 * pop_size) / (pop_size * (1.0 + CR))))
    if base in ("gwo", "woa"):
        return max(1, int((budget - pop_size) / pop_size))
    return 0                      # not a search method, so a budget does not apply


def _search_frame(src: np.ndarray, tgt: np.ndarray, center_align: bool):
    # Move the clouds into the frame the search runs in.
    # params_to_matrix rotates about the origin, so the centroid must sit there
    if center_align:
        T_pre = centroid_align(src, tgt)
        c = np.mean(tgt, axis=0)              # centroids coincide after alignment
        return (src + T_pre[:3, 3]) - c, tgt - c, T_pre, c
    return src, tgt, np.eye(4), np.zeros(3)


def _compose(T_search: np.ndarray, T_pre: np.ndarray, c: np.ndarray) -> np.ndarray:
    # Map a solution found in the search frame back to world coordinates.
    T_to_origin, T_back = np.eye(4), np.eye(4)
    T_to_origin[:3, 3] = -c
    T_back[:3, 3] = c
    return T_back @ T_search @ T_to_origin @ T_pre


def _run_metaheuristic(
    name, src, tgt, seed, pop_size, max_iter, trim_ratio, bounds, center_align
) -> CoarseResult:
    src_s, tgt_s, T_pre, c = _search_frame(src, tgt, center_align)
    fitness = RegistrationFitness(src_s, tgt_s, trim_ratio=trim_ratio)
    res = OPTIMIZERS[name](fitness, bounds, pop_size=pop_size,
                           max_iter=max_iter, seed=seed)
    res.transform = _compose(res.transform, T_pre, c)
    return res


def hybrid_register(
    opt_name: str,
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    voxel_size: float,
    seed: int | None = None,
    pop_size: int = 30,
    max_iter: int = 200,
    trim_ratio: float = 0.8,
    refine_rot_deg: float = 15.0,
    refine_trans: float | None = None,
) -> CoarseResult:
    # RANSAC for a coarse solution, then a metaheuristic refines its neighbourhood.
    # the two are complementary: RANSAC samples discretely, the search is unreliable alone
    t0 = time.perf_counter()

    ransac = fpfh_ransac(source_down, target_down, voxel_size, seed=seed)
    T_r = ransac.transform

    src = np.asarray(source_down.points)
    tgt = np.asarray(target_down.points)
    src_moved = (np.hstack([src, np.ones((len(src), 1))]) @ T_r.T)[:, :3]

    if refine_trans is None:
        refine_trans = voxel_size * 5.0       # neighbourhood scales with the cloud

    rot = np.deg2rad(refine_rot_deg)
    bounds = np.array([[-refine_trans, refine_trans]] * 3 + [[-rot, rot]] * 3)

    # already aligned by RANSAC, so centroid pre-alignment would undo that work
    res = _run_metaheuristic(opt_name, src_moved, tgt, seed, pop_size, max_iter,
                             trim_ratio, bounds, center_align=False)

    res.transform = res.transform @ T_r
    res.elapsed = time.perf_counter() - t0
    return res


def global_register(
    method: str,
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    voxel_size: float,
    seed: int | None = None,
    pop_size: int = 30,
    max_iter: int = 200,
    trim_ratio: float = 0.8,
    bound_scale: float = 0.2,
    center_align: bool = True,
    refine_rot_deg: float = 15.0,
) -> CoarseResult:
    # Entry point: none, fpfh_ransac, gwo, woa, iwoa, or a ransac_ hybrid.
    method = method.lower()

    if method == "none":
        return CoarseResult(transform=np.eye(4), fitness=float("nan"))

    if method in ("fpfh", "fpfh_ransac", "ransac"):
        return fpfh_ransac(source_down, target_down, voxel_size, seed=seed)

    if method in HYBRID:
        return hybrid_register(
            HYBRID[method], source_down, target_down, voxel_size,
            seed=seed, pop_size=pop_size, max_iter=max_iter,
            trim_ratio=trim_ratio, refine_rot_deg=refine_rot_deg)

    if method in OPTIMIZERS:
        return _run_metaheuristic(
            method, np.asarray(source_down.points), np.asarray(target_down.points),
            seed, pop_size, max_iter, trim_ratio,
            default_bounds(bound_scale), center_align)

    raise ValueError(f"unknown coarse registration method: {method}")