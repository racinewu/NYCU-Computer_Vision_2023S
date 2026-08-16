from __future__ import annotations

USAGE = """Usage: python experiment.py <point_cloud.ply>

Rotates one cloud by a known amount to build the target, so the true transform
is known and rotation and translation errors can be measured directly.
Set CONFIG ablation to make the problem harder."""

import copy
import os
import sys
import time

import numpy as np
import open3d as o3d

import utils
from coarse import global_register, iters_for_budget
from fine import refine

CONFIG = {
    # ---- main experiment ----
    "angles": [30, 90, 180],       # rotation magnitudes to test, in degrees
    "trans_ratio": 0.3,            # translation = bbox diagonal * this
    "n_seeds": 5,                  # repetitions per angle (metaheuristics vary)

    # ---- ablation: change ONE condition and see how far results degrade ----
    # "none"     ideal: 100% overlap, matched density, no noise
    # "density"  source and target sampled to different point counts
    # "noise"    Gaussian noise added to the target
    # "crop"     part of the target removed, simulating partial overlap
    "ablation": "none",

    # Suffix of the lower resolution file next to the input, or downsample if absent.
    "density_target_suffix": "_res4",
    "density_target_points": 500,
    "noise_ratio": 0.002,          # ablation=noise: sigma = diagonal * this
    "crop_keep_ratio": 0.7,        # ablation=crop: fraction of target kept

    # ---- preprocessing ----
    "coarse_points": 1500,
    "fine_points": 6000,
    "remove_outlier": False,

    # ---- coarse registration ----
    "trim_ratio": 0.8,             # automatically lowered to 0.4 when ablation=crop
    "pop_size": 30,
    "max_iter": 200,
    # Rescale epochs per method so every optimizer gets the same evaluation count.
    "equal_eval_budget": True,
    "eval_budget": 6000,
    "center_align": True,
    "bound_scale_ratio": 0.15,
    "refine_rot_deg": 15.0,

    # ---- fine registration ----
    "icp_type": "icp_plane",       # 'icp' | 'icp_plane' | 'iicp'
    "normal_angle_deg": 30.0,
    "icp_dist_factor": 1.5,
    "icp_max_iter": 100,

    # ---- success criteria (translation threshold scales with the cloud) ----
    "success_rre_deg": 5.0,
    "success_rte_ratio": 0.02,     # RTE < diagonal * this

    # ---- visualisation ----
    # Each method shows its median run; 8 methods by 3 angles means 48 windows.
    "show_each_coarse": True,
    "show_each_fine": True,
}

METHODS = ["none", "fpfh_ransac", "gwo", "woa", "iwoa",
           "ransac_gwo", "ransac_woa", "ransac_iwoa"]

# (group label, [(column name, width), ...]) for the per-angle table
GROUPS = [
    ("",       [("method", 14)]),
    ("coarse", [("RRE(deg)", 15)]),
    ("fine",   [("success", 10), ("RRE(deg)", 15), ("RTE", 16)]),
    ("",       [("evals", 8), ("time(s)", 13)]),
]
SEP = " | "

RUN_HEADER = (f"  {'method':<14}{'seed':>5}{'coarse RRE':>13}{'fine RRE':>11}"
              f"{'fine RTE':>11}{'result':>9}{'time':>9}")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def ms(values, prec: int, std_prec: int) -> str:
    v = np.asarray(values, dtype=float)
    if len(v) == 0 or np.all(np.isnan(v)):
        return "n/a"
    return f"{np.nanmean(v):.{prec}f}\u00b1{np.nanstd(v):.{std_prec}f}"


def build_pair(pcd_raw, cfg, diag, angle, problem_seed, target_base=None):
    # Build one source, target and T_gt triple with a known correct answer.
    T_gt = utils.random_transform(angle, diag * cfg["trans_ratio"], seed=problem_seed)
    source_raw = copy.deepcopy(pcd_raw)
    target_raw = copy.deepcopy(target_base if target_base is not None else pcd_raw)
    target_raw.transform(T_gt)

    ab = cfg["ablation"]
    if ab == "noise":
        rng = np.random.default_rng(problem_seed)
        pts = np.asarray(target_raw.points)
        target_raw.points = o3d.utility.Vector3dVector(
            pts + rng.normal(scale=diag * cfg["noise_ratio"], size=pts.shape))
    elif ab == "crop":
        target_raw = utils.crop_partial(target_raw, cfg["crop_keep_ratio"],
                                        seed=problem_seed)

    return source_raw, target_raw, T_gt


def prepare_angle(pcd_raw, cfg, diag, angle, problem_seed, target_base=None):
    # Build and preprocess the single problem shared by every run at one angle.
    source_raw, target_raw, T_gt = build_pair(pcd_raw, cfg, diag, angle,
                                              problem_seed, target_base)

    # Density ablation thins only the coarse target, isolating the coarse stage.
    tgt_coarse_n = (cfg["density_target_points"]
                    if cfg["ablation"] == "density" and target_base is None
                    else cfg["coarse_points"])

    cv_s = utils.voxel_for_target_count(source_raw, cfg["coarse_points"])
    cv_t = utils.voxel_for_target_count(target_raw, tgt_coarse_n)
    fv_s = utils.voxel_for_target_count(source_raw, cfg["fine_points"])
    fv_t = utils.voxel_for_target_count(target_raw, cfg["fine_points"])

    return {
        "source": source_raw,
        "target": target_raw,
        "T_gt": T_gt,
        "src_c": utils.preprocess(source_raw, cv_s, remove_outlier=cfg["remove_outlier"]),
        "tgt_c": utils.preprocess(target_raw, cv_t, remove_outlier=cfg["remove_outlier"]),
        "src_f": utils.preprocess(source_raw, fv_s, remove_outlier=cfg["remove_outlier"]),
        "tgt_f": utils.preprocess(target_raw, fv_t, remove_outlier=cfg["remove_outlier"]),
        "cv_t": cv_t,
        "fv_t": fv_t,
    }


def run_once(prob, cfg, diag, angle, method, seed, iters):
    # Run one method once on the prepared problem. seed varies the algorithm only.
    T_gt = prob["T_gt"]
    eval_pts = np.asarray(prob["src_f"].points)
    trim = 0.4 if cfg["ablation"] == "crop" else cfg["trim_ratio"]

    c = global_register(method, prob["src_c"], prob["tgt_c"], prob["cv_t"], seed=seed,
                        pop_size=cfg["pop_size"], max_iter=iters, trim_ratio=trim,
                        bound_scale=diag * cfg["bound_scale_ratio"],
                        center_align=cfg["center_align"],
                        refine_rot_deg=cfg["refine_rot_deg"])
    m_c = utils.evaluate(c.transform, T_gt, eval_pts)

    f = refine(cfg["icp_type"], prob["src_f"], prob["tgt_f"], c.transform,
               prob["fv_t"] * cfg["icp_dist_factor"],
               max_iteration=cfg["icp_max_iter"],
               normal_angle_deg=cfg["normal_angle_deg"])
    m_f = utils.evaluate(f.transform, T_gt, eval_pts)

    ok = (m_f["RRE_deg"] < cfg["success_rre_deg"]
          and m_f["RTE"] < diag * cfg["success_rte_ratio"])

    return {"angle": angle, "method": method, "seed": seed,
            "coarse_RRE": m_c["RRE_deg"], "coarse_RTE": m_c["RTE"],
            "fine_RRE": m_f["RRE_deg"], "fine_RTE": m_f["RTE"],
            "n_src": len(prob["src_c"].points), "n_tgt": len(prob["tgt_c"].points),
            "n_eval": c.n_eval, "time": c.elapsed + f.elapsed,
            "success": int(ok),
            "T_coarse": c.transform, "T_fine": f.transform}


def angle_table(rows, angle, cfg) -> None:
    # Per-angle summary, averaging successful runs only because results are bimodal.
    top, bottom = [], []
    for label, cols in GROUPS:
        top.append(f"{label:^{sum(w for _, w in cols)}}")
        bottom.append("".join(f"{n:>{w}}" if n != "method" else f"{n:<{w}}"
                              for n, w in cols))
    body = []
    for method in METHODS:
        rs = [r for r in rows if r["method"] == method and r["angle"] == angle]
        if not rs:
            continue
        good = [r for r in rs if r["success"]]
        sel = good if good else rs        # fall back so the row is never blank
        cells = [
            f"{method:<14}",
            f"{ms([r['coarse_RRE'] for r in sel], 2, 2):>15}",
            f"{f'{100.0 * len(good) / len(rs):.0f}% {len(good)}/{len(rs)}':>10}",
            f"{ms([r['fine_RRE'] for r in sel], 3, 3):>15}",
            f"{ms([r['fine_RTE'] for r in sel], 5, 4):>16}",
            f"{rs[0]['n_eval']:>8}",
            f"{ms([r['time'] for r in rs], 2, 2):>13}",
        ]
        body.append(SEP.join(["".join(cells[0:1]), cells[1],
                              "".join(cells[2:5]), "".join(cells[5:7])]))

    log()
    log(f"  rotation {angle} deg - summary")
    for line in utils.box_table([SEP.join(top), SEP.join(bottom)], body, indent="  "):
        log(line)


def main() -> None:
    if len(sys.argv) != 2:
        log(USAGE)
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        log(f"File not found: {path}")
        sys.exit(1)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    cfg = CONFIG

    pcd_raw = utils.load_point_cloud(path)
    diag = utils.bbox_diagonal(pcd_raw)

    target_base, target_path = None, None
    if cfg["ablation"] == "density":
        stem, ext = os.path.splitext(path)
        target_path = stem + cfg["density_target_suffix"] + ext
        if os.path.isfile(target_path):
            target_base = utils.load_point_cloud(target_path)
        else:
            log(f"No {target_path}; downsampling the target to "
                f"{cfg['density_target_points']} points instead")
    n_total = len(cfg["angles"]) * len(METHODS) * cfg["n_seeds"]
    n_meta = len(cfg["angles"]) * 6 * cfg["n_seeds"]   # 6 methods use a search

    log(f"Input: {path}   {len(pcd_raw.points)} points   bbox diagonal {diag:.4f}")
    log(f"Ablation: {cfg['ablation']}   ICP: {cfg['icp_type']}")
    if target_base is not None:
        log(f"Density target: {target_path}   {len(target_base.points)} points")
    log(f"Angles: {cfg['angles']}   {cfg['n_seeds']} seeds each   {n_total} runs")
    log(f"Estimated: about {n_meta * 12 / 60:.0f} min "
        f"(the search-based methods take ~12s per run, the rest are negligible)")
    log("Reduce n_seeds or max_iter if that is too slow.")

    # Report the counts actually fed in, so an ablation can be confirmed.
    sample = prepare_angle(pcd_raw, cfg, diag, cfg["angles"][0], 0, target_base)
    log("Points after downsampling, for the first angle")
    for stage, a, b in (("coarse", sample["src_c"], sample["tgt_c"]),
                        ("fine", sample["src_f"], sample["tgt_f"])):
        ns, nt = len(a.points), len(b.points)
        log(f"  {stage}: source={ns} target={nt} ratio={ns / max(nt, 1):.2f}")

    rows = []

    for problem_seed, angle in enumerate(cfg["angles"]):
        # One fixed problem per angle, so the algorithm is the only variable.
        prob = prepare_angle(pcd_raw, cfg, diag, angle, problem_seed, target_base)

        log()
        log(f"rotation {angle} deg (one fixed problem, seed varies the algorithm)")
        log(RUN_HEADER)
        log("  " + "-" * (len(RUN_HEADER) - 2))

        for method in METHODS:
            iters = cfg["max_iter"]
            if cfg["equal_eval_budget"]:
                n = iters_for_budget(method, cfg["pop_size"], cfg["eval_budget"])
                if n:
                    iters = n

            runs = []
            for seed in range(cfg["n_seeds"]):
                r = run_once(prob, cfg, diag, angle, method, seed, iters)
                runs.append(r)
                rows.append(r)
                log(f"  {method:<14}{seed:>5}{r['coarse_RRE']:>13.2f}"
                    f"{r['fine_RRE']:>11.2f}{r['fine_RTE']:>11.5f}"
                    f"{'OK' if r['success'] else 'FAIL':>9}{r['time']:>8.2f}s")

            if cfg["show_each_coarse"] or cfg["show_each_fine"]:
                median = sorted(runs, key=lambda x: x["coarse_RRE"])[len(runs) // 2]
                log(f"  {method:<14}  median run is seed={median['seed']} "
                    f"(coarse RRE={median['coarse_RRE']:.3f})")
                if cfg["show_each_coarse"]:
                    utils.show(prob["source"], prob["target"],
                               T=median["T_coarse"], show_axes=False,
                               window_name=f"{angle}deg coarse - {method} (median)")
                if cfg["show_each_fine"]:
                    utils.show(prob["source"], prob["target"],
                               T=median["T_fine"], show_axes=False,
                               window_name=f"{angle}deg fine ({cfg['icp_type']}) "
                                           f"- {method} (median)")

        angle_table(rows, angle, cfg)




if __name__ == "__main__":
    main()