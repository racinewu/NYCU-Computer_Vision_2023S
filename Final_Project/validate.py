from __future__ import annotations

USAGE = """Usage: python validate.py <point_cloud.ply>

Runs mealpy's reference WOA and GWO on the same fitness function as the
hand-written versions, to confirm the hand-written ones behave correctly.
Requires: pip install mealpy"""

import os
import sys

import numpy as np
import open3d as o3d

import utils
from coarse import default_bounds, gwo, woa, centroid_align

CONFIG = {
    "coarse_points": 1500,
    "trim_ratio": 0.8,
    "pop_size": 30,
    "max_iter": 200,
    "rot_deg": 60.0,
    "trans_ratio": 0.3,
    "bound_scale_ratio": 0.15,
    "n_seeds": 3,

    # A run counts as successful when the coarse solution is within this.
    "success_rre_deg": 5.0,

    # Show the initial pose, then the median run of each implementation.
    "show_initial": True,
    "show_each_stage": True,
}

# hand-written function <-> mealpy class name
PAIRS = [("woa", "OriginalWOA"), ("gwo", "OriginalGWO")]

COLS = [("seed", 6), ("own fitness", 15), ("mealpy fitness", 17),
        ("own RRE", 12), ("mealpy RRE", 14), ("own evals", 12),
        ("mealpy evals", 15)]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def make_problem(pcd_raw, cfg, diag, seed):
    # Build one problem and return its fitness, bounds, truth, frame map and clouds.
    T_gt = utils.random_transform(cfg["rot_deg"], diag * cfg["trans_ratio"], seed=seed)
    source, target = utils.make_pair(pcd_raw, T_gt)

    v = utils.voxel_for_target_count(target, cfg["coarse_points"])
    src_d = utils.preprocess(source, v)
    tgt_d = utils.preprocess(target, v)
    src = np.asarray(src_d.points)
    tgt = np.asarray(tgt_d.points)

    # Same frame as coarse.py: align centroids then shift to the origin.
    T_pre = centroid_align(src, tgt)
    c = np.mean(tgt, axis=0)
    fitness = utils.RegistrationFitness((src + T_pre[:3, 3]) - c, tgt - c,
                                        trim_ratio=cfg["trim_ratio"])

    def compose(T_search):
        T_o, T_b = np.eye(4), np.eye(4)
        T_o[:3, 3], T_b[:3, 3] = -c, c
        return T_b @ T_search @ T_o @ T_pre

    bounds = default_bounds(diag * cfg["bound_scale_ratio"])
    return fitness, bounds, T_gt, compose, source, target


def run_mealpy(class_name, fitness, bounds, cfg, seed):
    # Call mealpy's reference implementation using only the fitness and bounds.
    from mealpy import FloatVar
    import mealpy

    fitness.reset_counter()
    problem = {
        "obj_func": fitness,
        "bounds": FloatVar(lb=bounds[:, 0], ub=bounds[:, 1]),
        "minmax": "min",
        "log_to": None,
    }
    cls = getattr(mealpy, class_name, None)
    if cls is None:                       # import path differs between versions
        from mealpy.swarm_based import WOA, GWO
        cls = getattr(WOA, class_name, None) or getattr(GWO, class_name)

    model = cls(epoch=cfg["max_iter"], pop_size=cfg["pop_size"])
    g = model.solve(problem, seed=seed)
    return np.asarray(g.solution), float(g.target.fitness), fitness.eval_count


def main() -> None:
    if len(sys.argv) != 2:
        log(USAGE)
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        log(f"File not found: {path}")
        sys.exit(1)

    try:
        import mealpy  # noqa: F401
    except ImportError:
        log("mealpy is required: pip install mealpy")
        sys.exit(1)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    cfg = CONFIG

    pcd_raw = utils.load_point_cloud(path)
    diag = utils.bbox_diagonal(pcd_raw)
    log(f"Input: {path}   {len(pcd_raw.points)} points   bbox diagonal {diag:.4f}")
    log(f"Setup: pop {cfg['pop_size']} x epoch {cfg['max_iter']}, "
        f"rotation {cfg['rot_deg']} deg, {cfg['n_seeds']} seeds")
    log(f"Success: coarse RRE < {cfg['success_rre_deg']} deg")

    # Show the problem; every seed differs, so seed zero represents them.
    if cfg["show_initial"]:
        _, _, _, _, src0, tgt0 = make_problem(pcd_raw, cfg, diag, 0)
        log()
        log("Initial pose for seed 0 (orange is source, blue is target). "
            "Close the window to continue.")
        utils.show(src0, tgt0, T=None, show_axes=False,
                   window_name="Before registration (seed 0)")

    for own_name, meal_name in PAIRS:
        log()
        log(f"{own_name.upper()} vs mealpy {meal_name}")

        header = "".join(f"{n:>{w}}" for n, w in COLS)
        log(header)
        log("-" * len(header))

        runs, body = [], []
        for seed in range(cfg["n_seeds"]):
            fitness, bounds, T_gt, compose, source, target = make_problem(
                pcd_raw, cfg, diag, seed)

            own_fn = woa if own_name == "woa" else gwo
            r = own_fn(fitness, bounds, pop_size=cfg["pop_size"],
                       max_iter=cfg["max_iter"], seed=seed)
            own_T = compose(r.transform)
            own_rre = utils.rotation_error_deg(own_T, T_gt)
            own_eval = r.n_eval

            sol, m_fit, m_eval = run_mealpy(meal_name, fitness, bounds, cfg, seed)
            m_T = compose(utils.params_to_matrix(sol))
            m_rre = utils.rotation_error_deg(m_T, T_gt)

            log(f"{seed:>6}{r.fitness:>15.6f}{m_fit:>17.6f}"
                f"{own_rre:>12.2f}{m_rre:>14.2f}{own_eval:>12}{m_eval:>15}")

            runs.append({"seed": seed, "own_fit": r.fitness, "m_fit": m_fit,
                         "own_rre": own_rre, "m_rre": m_rre,
                         "own_eval": own_eval, "m_eval": m_eval,
                         "own_T": own_T, "m_T": m_T,
                         "source": source, "target": target})

        # Results are bimodal, so average successful runs only.
        thr = cfg["success_rre_deg"]
        own_ok = [r for r in runs if r["own_rre"] < thr]
        m_ok = [r for r in runs if r["m_rre"] < thr]
        n = len(runs)

        def cell(vals, key, prec):
            v = [r[key] for r in vals]
            return (f"{np.mean(v):.{prec}f}\u00b1{np.std(v):.{prec}f}"
                    if v else "n/a")

        srows = [
            f"{'hand-written':<16}"
            f"{f'{100.0 * len(own_ok) / n:.0f}% {len(own_ok)}/{n}':>14}"
            f"{cell(own_ok or runs, 'own_fit', 6):>20}"
            f"{cell(own_ok or runs, 'own_rre', 2):>16}"
            f"{int(np.mean([r['own_eval'] for r in runs])):>14}",
            f"{'mealpy':<16}"
            f"{f'{100.0 * len(m_ok) / n:.0f}% {len(m_ok)}/{n}':>14}"
            f"{cell(m_ok or runs, 'm_fit', 6):>20}"
            f"{cell(m_ok or runs, 'm_rre', 2):>16}"
            f"{int(np.mean([r['m_eval'] for r in runs])):>14}",
        ]
        shead = (f"{'implementation':<16}{'success':>14}{'fitness':>20}"
                 f"{'RRE(deg)':>16}{'evals':>14}")

        log()
        for line in utils.box_table([shead], srows):
            log(line)
        log("statistics cover successful runs only")

        if cfg["show_each_stage"]:
            median = sorted(runs, key=lambda r: r["own_rre"])[len(runs) // 2]
            log(f"median run is seed={median['seed']} "
                f"(own RRE={median['own_rre']:.2f}, mealpy RRE={median['m_rre']:.2f})")
            utils.show(median["source"], median["target"], T=median["own_T"],
                       show_axes=False,
                       window_name=f"{own_name.upper()} hand-written (median run)")
            utils.show(median["source"], median["target"], T=median["m_T"],
                       show_axes=False,
                       window_name=f"mealpy {meal_name} (median run)")


if __name__ == "__main__":
    main()