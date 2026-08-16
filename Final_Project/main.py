from __future__ import annotations

USAGE = """Usage: python main.py <source.ply> <target.ply> [--conf <bun.conf>]

  --conf   read Stanford .conf reference poses and also report RRE and RTE.
           Without it only chamfer and ICP fitness are available.

An extra random 3D rotation and translation is applied to the source; see
extra_rot_deg and extra_trans_ratio in CONFIG. Set both to zero to use the raw
relative pose of the two scans."""

import os
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import utils
from coarse import global_register, iters_for_budget
from fine import refine

CONFIG = {
    # Voxel size comes from a target point count, since count drives runtime.
    "coarse_points": 1500,
    "fine_points": 6000,
    "remove_outlier": True,

    # Key under partial overlap: a high value lets non-overlapping points mislead.
    "trim_ratio": 0.4,

    "pop_size": 30,
    "max_iter": 200,
    # Rescale epochs per method so every optimizer gets the same evaluation count.
    "equal_eval_budget": True,
    "eval_budget": 6000,

    "center_align": True,
    "bound_scale_ratio": 0.15,
    "refine_rot_deg": 15.0,        # hybrid methods: neighbourhood half-width

    # Extra transform on the source, with a fixed seed so the problem never changes.
    "extra_rot_deg": 70.0,
    "extra_trans_ratio": 0.10,     # translation = bbox diagonal * this
    "extra_seed": 0,

    "icp_type": "icp_plane",       # 'icp' | 'icp_plane' | 'iicp'
    "normal_angle_deg": 60.0,      # iicp only: normal angle threshold
    "icp_dist_factor": 1.5,
    "icp_max_iter": 100,

    "n_seeds": 5,

    # Success is fine RRE below the threshold, or chamfer near the best seen.
    "success_rre_deg": 5.0,
    "success_chamfer_factor": 2.0,

    # Narrow the summary table; the per-run log always carries the full set.
    "compact_table": False,

    # Each method shows its median run; 8 methods means 16 windows in total.
    "show_initial": True,
    "show_each_coarse": True,
    "show_each_fine": True,

    "out_dir": "stitch",
}

METHODS = ["none", "fpfh_ransac", "gwo", "woa", "iwoa",
           "ransac_gwo", "ransac_woa", "ransac_iwoa"]

# Methods with no randomness, so repeating them changes nothing. RANSAC is random.
DETERMINISTIC = {"none"}

# Group label with its columns and widths, shared by the header and the rows.
GROUPS = [
    ("",       [("method", 14)]),
    ("coarse", [("chamfer", 16), ("RRE(deg)", 13)]),
    # Fine columns often match across methods, which is itself a finding.
    ("fine",   [("success", 10), ("chamfer", 16), ("ICP fitness", 15),
                ("RRE(deg)", 15), ("RTE", 16)]),
    ("",       [("time(s)", 13)]),
]
# Trimmed set for narrow terminals; enable with compact_table in CONFIG.
COMPACT_DROP = ("ICP fitness", "RTE")
SEP = " | "
NO_GT = ("RRE(deg)", "RTE")        # dropped when no reference pose is available


def log(msg: str = "") -> None:
    print(msg, flush=True)


def ms(values, prec: int, std_prec: int) -> str:
    # Format a list as mean plus or minus standard deviation.
    v = np.asarray(values, dtype=float)
    if np.all(np.isnan(v)):
        return "n/a"
    return f"{np.nanmean(v):.{prec}f}\u00b1{np.nanstd(v):.{std_prec}f}"


def describe(name: str, pcd) -> None:
    bb = pcd.get_axis_aligned_bounding_box()
    log(f"  {name:<8}{len(pcd.points):>8} points   "
        f"bbox {np.array2string(bb.get_extent(), precision=4, floatmode='fixed')}   "
        f"centroid {np.array2string(pcd.get_center(), precision=4, floatmode='fixed')}")


def visualise(source, target, T, title: str, out_path: str | None) -> None:
    # Show the overlay, falling back to writing a .ply when headless.
    shown = utils.show(source, target, T=T, window_name=title, show_axes=False)
    if out_path:
        utils.save_pair_ply(source, target, T, out_path)
        if not shown:
            log(f"      no display available, wrote {out_path}")


def parse_args() -> tuple[str, str, str | None]:
    argv, conf_path, positional = sys.argv[1:], None, []
    i = 0
    while i < len(argv):
        if argv[i] in ("--conf", "--config"):
            i += 1
            if i >= len(argv):
                log("--conf requires a file path")
                sys.exit(1)
            conf_path = argv[i]
        else:
            positional.append(argv[i])
        i += 1

    if len(positional) != 2:
        log(USAGE)
        sys.exit(1)
    for p in positional:
        if not os.path.isfile(p):
            log(f"File not found: {p}")
            sys.exit(1)
    return positional[0], positional[1], conf_path


# --------------------------------------------------------------------------
def summary_table(stats: list[dict], has_gt: bool, compact: bool = False) -> None:
    # Two-row header: group labels centred over their block, then column names.
    drop = set() if has_gt else set(NO_GT)
    if compact:
        drop |= set(COMPACT_DROP)
    groups = [(label, [c for c in cols if c[0] not in drop])
              for label, cols in GROUPS]
    groups = [(label, cols) for label, cols in groups if cols]

    top, bottom = [], []
    for label, cols in groups:
        block = sum(w for _, w in cols)
        top.append(f"{label:^{block}}")
        bottom.append("".join(f"{n:>{w}}" if n != "method" else f"{n:<{w}}"
                              for n, w in cols))

    rows = []
    for s in stats:
        cells = []
        for label, cols in groups:
            cells.append("".join(f"{s[(label, n)]:>{w}}" if n != "method"
                                 else f"{s[(label, n)]:<{w}}" for n, w in cols))
        rows.append(SEP.join(cells))

    log()
    for line in utils.box_table([SEP.join(top), SEP.join(bottom)], rows):
        log(line)


# --------------------------------------------------------------------------
def main() -> None:
    src_path, tgt_path, conf_path = parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    cfg = CONFIG
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # ---------------- 1. Load ----------------
    log("[1/5] Loading point clouds")
    source_raw = utils.load_point_cloud(src_path)
    target_raw = utils.load_point_cloud(tgt_path)
    describe("source", source_raw)
    describe("target", target_raw)
    diag = utils.bbox_diagonal(target_raw)

    # ---------------- Reference pose (optional) ----------------
    T_gt = None
    if conf_path:
        log()
        log("[1b] Reading reference poses")
        if not os.path.isfile(conf_path):
            log(f"  {conf_path} not found; reporting chamfer / ICP fitness only")
        else:
            try:
                poses = utils.parse_conf(conf_path)
                # Try all 8 convention combinations and keep the best fitting one.
                T_gt, tag, ch = utils.resolve_gt(
                    poses, src_path, tgt_path,
                    np.asarray(source_raw.points), np.asarray(target_raw.points))
                log(f"  convention={tag}  verification_chamfer={ch:.5f}")
                log(f"  reference_rotation="
                    f"{np.degrees(Rotation.from_matrix(T_gt[:3, :3]).magnitude()):.2f}deg")
                if ch > diag * 0.01:
                    log("  WARNING: verification chamfer is large; the .conf layout "
                        "may differ from what is assumed. Treat RRE/RTE with caution.")
            except Exception as e:
                log(f"  parse failed ({type(e).__name__}: {e}); "
                    "reporting chamfer / ICP fitness only")
                T_gt = None

    # ---------------- Extra random transform (fixed seed) ----------------
    if cfg["extra_rot_deg"] != 0.0 or cfg["extra_trans_ratio"] != 0.0:
        extra_trans = diag * cfg["extra_trans_ratio"]
        log()
        log(f"[1c] Extra random 3-D transform on source: "
            f"rotation={cfg['extra_rot_deg']}deg  translation={extra_trans:.4f}  "
            f"seed={cfg['extra_seed']} (fixed)")
        source_raw, T_gt = utils.compose_extra_transform(
            source_raw, T_gt, cfg["extra_rot_deg"], extra_trans,
            seed=cfg["extra_seed"])
        if T_gt is not None:
            log(f"  reference updated: rotation="
                f"{np.degrees(Rotation.from_matrix(T_gt[:3, :3]).magnitude()):.2f}deg  "
                f"translation={np.linalg.norm(T_gt[:3, 3]):.4f}")

    coarse_voxel = utils.voxel_for_target_count(target_raw, cfg["coarse_points"])
    fine_voxel = utils.voxel_for_target_count(target_raw, cfg["fine_points"])
    log(f"  bbox_diagonal={diag:.4f}  auto_voxel: coarse={coarse_voxel:.4f} "
        f"fine={fine_voxel:.4f}")

    # ---------------- 2. Initial pose ----------------
    log()
    log("[2/5] Initial relative pose (orange=source, blue=target, raw coordinates)")
    if cfg["show_initial"]:
        log("      Close the window to continue.")
        visualise(source_raw, target_raw, None, "Before registration",
                  os.path.join(cfg["out_dir"], "before.ply"))

    # ---------------- 3. Preprocess ----------------
    log()
    log("[3/5] Preprocessing (downsample + outlier removal + normals)")
    t0 = time.perf_counter()
    src_c = utils.preprocess(source_raw, coarse_voxel, remove_outlier=cfg["remove_outlier"])
    tgt_c = utils.preprocess(target_raw, coarse_voxel, remove_outlier=cfg["remove_outlier"])
    src_f = utils.preprocess(source_raw, fine_voxel, remove_outlier=cfg["remove_outlier"])
    tgt_f = utils.preprocess(target_raw, fine_voxel, remove_outlier=cfg["remove_outlier"])
    log(f"  coarse: source={len(src_c.points)} target={len(tgt_c.points)}")
    log(f"  fine:   source={len(src_f.points)} target={len(tgt_f.points)}")
    log(f"  time={time.perf_counter() - t0:.2f}s")

    src_f_pts, tgt_f_pts = np.asarray(src_f.points), np.asarray(tgt_f.points)
    icp_dist = fine_voxel * cfg["icp_dist_factor"]

    # ---------------- 4. Registration ----------------
    log()
    log(f"[4/5] Registration ({cfg['n_seeds']} run(s) per method, "
        f"optimizer seed varies, problem fixed)")
    run_hdr = (f"  {'method':<14}{'seed':>5}{'coarse chamf':>14}{'coarse RRE':>12}"
               f"{'evals':>8}{'fine chamf':>12}{'fitness':>9}{'fine RRE':>10}"
               f"{'time':>9}")
    log(run_hdr)
    log("  " + "-" * (len(run_hdr) - 2))
    all_runs, best_T, best_key = {}, {}, {}

    for method in METHODS:
        runs = []
        n_runs = 1 if method in DETERMINISTIC else cfg["n_seeds"]
        iters = cfg["max_iter"]
        if cfg["equal_eval_budget"]:
            n = iters_for_budget(method, cfg["pop_size"], cfg["eval_budget"])
            if n:
                iters = n

        for seed in range(n_runs):
            c = global_register(
                method, src_c, tgt_c, coarse_voxel,
                seed=seed, pop_size=cfg["pop_size"], max_iter=iters,
                trim_ratio=cfg["trim_ratio"],
                bound_scale=diag * cfg["bound_scale_ratio"],
                center_align=cfg["center_align"],
                refine_rot_deg=cfg["refine_rot_deg"])
            ch_c = utils.chamfer_distance(src_f_pts, tgt_f_pts, c.transform,
                                          trim_ratio=cfg["trim_ratio"])
            rre_c = (utils.rotation_error_deg(c.transform, T_gt)
                     if T_gt is not None else float("nan"))

            f = refine(cfg["icp_type"], src_f, tgt_f, c.transform, icp_dist,
                       max_iteration=cfg["icp_max_iter"],
                       normal_angle_deg=cfg["normal_angle_deg"])
            ch_f = utils.chamfer_distance(src_f_pts, tgt_f_pts, f.transform,
                                          trim_ratio=cfg["trim_ratio"])
            rre_f = rte_f = float("nan")
            if T_gt is not None:
                m = utils.evaluate(f.transform, T_gt, src_f_pts)
                rre_f, rte_f = m["RRE_deg"], m["RTE"]

            runs.append({"seed": seed,
                         "coarse_chamfer": ch_c, "coarse_rre": rre_c,
                         "chamfer": ch_f, "fitness": f.fitness,
                         "rre": rre_f, "rte": rte_f,
                         "time": c.elapsed + f.elapsed,
                         "T_coarse": c.transform, "T": f.transform})

            log(f"  {method:<14}{seed:>5}{ch_c:>14.5f}{rre_c:>12.3f}"
                f"{c.n_eval:>8}{ch_f:>12.5f}{f.fitness:>9.4f}{rre_f:>10.3f}"
                f"{c.elapsed + f.elapsed:>8.2f}s")

        # The median run represents the method, since rotations cannot be averaged.
        key = "coarse_rre" if T_gt is not None else "coarse_chamfer"
        median = sorted(runs, key=lambda r: r[key])[len(runs) // 2]
        best_T[method] = median["T"]
        best_key[method] = float(np.nanmean([r[key] for r in runs]))

        if n_runs > 1:
            log(f"  {method:<14}  median run is seed={median['seed']} "
                f"({key}={median[key]:.5f})")
        if cfg["show_each_coarse"]:
            visualise(source_raw, target_raw, median["T_coarse"],
                      f"Coarse - {method} (median run)",
                      os.path.join(cfg["out_dir"], f"coarse_{method}.ply"))
        if cfg["show_each_fine"]:
            visualise(source_raw, target_raw, median["T"],
                      f"Fine ({cfg['icp_type']}) - {method} (median run)",
                      os.path.join(cfg["out_dir"], f"fine_{method}.ply"))

        all_runs[method] = runs

    # ---------------- 5. Summary ----------------
    # Results are bimodal, so report a success rate and average successes only.
    if T_gt is not None:
        def ok(r):
            return r["rre"] < cfg["success_rre_deg"]
    else:
        floor = min(r["chamfer"] for rs in all_runs.values() for r in rs)
        limit = floor * cfg["success_chamfer_factor"]

        def ok(r):
            return r["chamfer"] <= limit

    stats = []
    for method in METHODS:
        runs = all_runs[method]
        good = [r for r in runs if ok(r)]
        rate = 100.0 * len(good) / len(runs)
        # Fall back to all runs when nothing succeeded, so the row is not blank
        sel = good if good else runs
        stats.append({
            ("", "method"): method,
            ("coarse", "chamfer"): ms([r["coarse_chamfer"] for r in sel], 5, 4),
            ("coarse", "RRE(deg)"): ms([r["coarse_rre"] for r in sel], 2, 2),
            ("fine", "success"): f"{rate:.0f}% {len(good)}/{len(runs)}",
            ("fine", "chamfer"): ms([r["chamfer"] for r in sel], 5, 4),
            ("fine", "ICP fitness"): ms([r["fitness"] for r in sel], 4, 4),
            ("fine", "RRE(deg)"): ms([r["rre"] for r in sel], 3, 3),
            ("fine", "RTE"): ms([r["rte"] for r in sel], 5, 4),
            ("", "time(s)"): ms([r["time"] for r in runs], 2, 2),
        })

    summary_table(stats, T_gt is not None, compact=cfg["compact_table"])
    crit = (f"fine RRE < {cfg['success_rre_deg']} deg" if T_gt is not None
            else f"fine chamfer <= {limit:.5f}")
    log(f"success = {crit}; the coarse/fine statistics above cover successful "
        f"runs only.")

    if T_gt is not None:
        log("Primary metrics: success rate, then coarse RRE among successes. ICP pulls")
        log("every converged run to the same optimum, so fine RRE is near-identical.")
    else:
        log("No --conf given, so RRE is unavailable. Chamfer separates success from")
        log("failure, but is insensitive to rotation error among failed solutions.")

    best = min(METHODS, key=lambda m: best_key[m])
    log()
    log(f"Best by {'coarse RRE' if T_gt is not None else 'coarse chamfer'}: "
        f"{best} ({best_key[best]:.3f})")
    log()
    log("Estimated transform T (source -> target), median run:")
    for row in best_T[best]:
        log("  ".join(f"{v:9.5f}" for v in row))

    for m in METHODS:
        utils.save_pair_ply(source_raw, target_raw, best_T[m],
                            os.path.join(cfg["out_dir"], f"result_{m}.ply"))

    log()
    log(f"[5/5] Overlays written to {cfg['out_dir']}/")
    visualise(source_raw, target_raw, best_T[best], f"Best - {best} (median run)", None)


if __name__ == "__main__":
    main()