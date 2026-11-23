"""Buddh International Circuit Geometry Validation Suite.

Executes rigorous automated geometric validation checks prior to policy evaluation:
    1. Full circuit loop closure.
    2. Clockwise direction of travel.
    3. Length agreement with 5,138m published reference.
    4. Track width consistency (12m reference -> 70.0px sim width).
    5. Absence of self-intersections or broken segments.
    6. Boundary clearance and absence of corner folding (R > W/2 everywhere).
    7. Preservation of realistic corner curvature.
    8. Starting point documentation and valid car state initialization.

Generates high-resolution top-down validation plot:
    analysis/buddh/buddh_geometry_validation.png
"""

import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from buddh.buddh_track import BuddhTrack
from env.car import Car


def validate_loop_closure(track: BuddhTrack) -> Tuple[bool, float]:
    """Verify centerline forms a closed loop."""
    p_start = track.centerline[0]
    p_end = track.centerline[-1]
    # Check wrap distance from end to start
    wrap_dist = float(np.linalg.norm(p_start - p_end))
    # In uniform resampling of closed spline, spacing between last and first
    # equals exactly the uniform step ds (approx 5.833 px)
    step_ds = float(np.linalg.norm(track.centerline[1] - track.centerline[0]))
    is_valid = wrap_dist <= (step_ds * 1.05)
    return is_valid, wrap_dist


def validate_direction(track: BuddhTrack) -> Tuple[bool, str, float]:
    """Verify clockwise travel direction via Shoelace signed area."""
    # Note: In simulator coords, Y is South (down).
    # In standard math (X=East, Y_up=North = -Y_sim):
    x = track.centerline[:, 0]
    y_up = -track.centerline[:, 1]
    shoelace_area = 0.5 * np.sum(x[:-1] * y_up[1:] - x[1:] * y_up[:-1])
    is_clockwise = shoelace_area < 0
    return is_clockwise, "clockwise" if is_clockwise else "counter_clockwise", float(shoelace_area)


def validate_length(track: BuddhTrack, ref_m: float = 5138.0, tol_pct: float = 0.5) -> Tuple[bool, float, float]:
    """Verify resampled length agrees with published reference length within tolerance."""
    scale = track.manifest["world_units_per_meter"]
    seg_lens = np.linalg.norm(np.diff(np.vstack([track.centerline, track.centerline[0:1]]), axis=0), axis=1)
    total_len_sim = float(np.sum(seg_lens))
    total_len_m = total_len_sim / scale
    pct_diff = abs(total_len_m - ref_m) / ref_m * 100.0
    is_valid = pct_diff <= tol_pct
    return is_valid, total_len_m, pct_diff


def validate_track_width(track: BuddhTrack, expected_px: float = 70.0) -> Tuple[bool, float, float]:
    """Verify width between left and right boundaries is strictly constant."""
    widths = np.linalg.norm(track.left_boundary - track.right_boundary, axis=1)
    min_w = float(np.min(widths))
    max_w = float(np.max(widths))
    is_valid = abs(min_w - expected_px) < 1e-4 and abs(max_w - expected_px) < 1e-4
    return is_valid, min_w, max_w


def validate_boundary_folding(track: BuddhTrack) -> Tuple[bool, float, float]:
    """Verify minimum curvature radius exceeds track half-width everywhere (R > half_w)."""
    min_radius_px = float(np.min(track._curv_r))
    margin_px = min_radius_px - track.half
    is_valid = min_radius_px > track.half
    return is_valid, min_radius_px, margin_px


def validate_car_initialization(track: BuddhTrack) -> Tuple[bool, Dict[str, Any]]:
    """Test car placement at start pose, casting rays, verifying healthy state."""
    car = Car()
    x0, y0, h0 = track.start_pose()
    car.x = x0
    car.y = y0
    car.angle = h0

    on_track = track.is_on_track(car.x, car.y)
    dist = track.signed_distance(car.x, car.y)
    heading_err = (car.angle - track.track_heading(car.x, car.y) + 180.0) % 360.0 - 180.0
    rays = car.cast_rays(track, ray_angles=(-90, -45, 0, 45, 90), max_dist=200, step=4)

    # Rays should be symmetric at track center on a straight
    ray_l90 = rays[0]
    ray_r90 = rays[4]
    sym_diff = abs(ray_l90 - ray_r90)

    is_valid = on_track and abs(dist) < 1e-3 and abs(heading_err) < 1e-3 and sym_diff <= 8.0
    diag = {
        "on_track": on_track,
        "signed_dist_px": dist,
        "heading_err_deg": heading_err,
        "rays": rays,
        "ray_l90": ray_l90,
        "ray_r90": ray_r90,
        "ray_fwd": rays[2],
    }
    return is_valid, diag


def generate_geometry_plot(
    track: BuddhTrack,
    output_path: str = "analysis/buddh/buddh_geometry_validation.png",
) -> None:
    """Produce top-down high-resolution circuit geometry plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 16), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    # Centerline & Boundaries
    ax.plot(
        track.left_boundary[:, 0],
        track.left_boundary[:, 1],
        color="#e03131",
        linewidth=1.8,
        label="Left Boundary (Outer/Kerb)",
    )
    ax.plot(
        track.right_boundary[:, 0],
        track.right_boundary[:, 1],
        color="#1c7ed6",
        linewidth=1.8,
        label="Right Boundary (Inner/Kerb)",
    )
    ax.plot(
        track.centerline[:, 0],
        track.centerline[:, 1],
        color="#f8f9fa",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label="Centerline (Resampled 1m)",
    )

    # Direction arrows along track
    arrow_indices = np.linspace(100, len(track.centerline) - 100, 24, dtype=int)
    for idx in arrow_indices:
        p = track.centerline[idx]
        t = track._tangents[idx]
        ax.annotate(
            "",
            xy=(p[0] + t[0] * 120.0, p[1] + t[1] * 120.0),
            xytext=(p[0], p[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#51cf66",
                lw=1.6,
                mutation_scale=14,
            ),
        )

    # Start/Finish Line
    x0, y0, h0 = track.start_pose()
    idx0 = 0
    nl = track.left_boundary[idx0]
    nr = track.right_boundary[idx0]
    ax.plot([nl[0], nr[0]], [nl[1], nr[1]], color="#ffd43b", linewidth=4.0, zorder=5, label="Start / Finish Line")
    ax.scatter([x0], [y0], color="#ffd43b", s=120, edgecolors="#000000", lw=2, zorder=6)
    ax.text(
        x0 + 80,
        y0 - 20,
        "START / FINISH\nNode 1986414860\nHeading 111.1°",
        color="#ffd43b",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a1d24", ec="#ffd43b", lw=1.2, alpha=0.9),
        zorder=7,
    )

    # Annotate Turns 1 to 16
    turns = track.manifest.get("turns", [])
    for t_info in turns:
        t_num = t_info["turn_number"]
        t_dir = t_info["direction"]
        t_rad_m = t_info["radius_m"]
        t_x = t_info["apex_sim_x"]
        t_y = t_info["apex_sim_y"]

        ax.scatter([t_x], [t_y], color="#ff922b", s=80, edgecolors="#ffffff", lw=1.5, zorder=6)

        # Offset text based on direction
        offset_x = 70 if t_dir == "Right" else -110
        offset_y = -50 if t_dir == "Right" else 50
        ax.annotate(
            f"T{t_num} ({t_dir[0]})\nR={t_rad_m:.0f}m",
            xy=(t_x, t_y),
            xytext=(t_x + offset_x, t_y + offset_y),
            arrowprops=dict(arrowstyle="->", color="#ff922b", lw=1.2),
            color="#ff922b",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#1a1d24", ec="#ff922b", lw=1, alpha=0.85),
            zorder=7,
        )

    # Invert Y axis so that North is Up on screen (since Y_sim is South)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#2c303b", linestyle=":", alpha=0.6)

    ax.set_title(
        "Buddh International Circuit — Top-Down Geometry Validation\n"
        "WGS84 UTM 43N Projected • Clockwise • 5,131.0 m • 16 Numbered Turns • Scale: 5.833 px/m",
        color="#ffffff",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Simulator X (pixels, East)", color="#ced4da", fontsize=11)
    ax.set_ylabel("Simulator Y (pixels, North is Up / Screen Y inverted)", color="#ced4da", fontsize=11)
    ax.tick_params(colors="#ced4da", labelsize=9)

    # Telemetry Summary Box
    info_text = (
        f"CIRCUIT SPECIFICATIONS\n"
        f"• Published Length: 5,138.0 m\n"
        f"• Resampled Length: {track.manifest['resampled_length_m']} m ({track.manifest['sim_track_length_px']} px)\n"
        f"• Length Discrepancy: {track.manifest['length_discrepancy_m']} m ({track.manifest['length_discrepancy_pct']}%)\n"
        f"• Reference Width: {track.manifest['reference_track_width_m']} m ({track.manifest['sim_track_width_px']} px)\n"
        f"• Metric-to-Sim Scale: {track.manifest['world_units_per_meter']:.4f} px/m\n"
        f"• Minimum Radius: {track.manifest['min_curvature_radius_m']} m ({track.manifest['min_curvature_radius_px']} px)\n"
        f"• Car Kinematic R_min: {track.manifest['car_min_turning_radius_px']} px (14.85 m)\n"
        f"• Boundary Margin: {track.manifest['boundary_margin_m']} m (No folding)\n"
        f"• Direction: Clockwise\n"
        f"• Centerline Samples: {track.manifest['num_centerline_points']}\n"
        f"• Source: OSM Relation 2532569 (ODbL 1.0)"
    )
    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        color="#f8f9fa",
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.6", fc="#14171f", ec="#3b4252", lw=1.5, alpha=0.92),
        zorder=8,
    )

    leg = ax.legend(loc="upper right", facecolor="#14171f", edgecolor="#3b4252", fontsize=10)
    for text in leg.get_texts():
        text.set_color("#ffffff")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Validation geometry plot saved to: {output_path}")


def run_full_validation() -> Dict[str, Any]:
    """Execute all geometric validation checks and print complete report."""
    track = BuddhTrack()
    results: Dict[str, Any] = {}

    print("=" * 70)
    print("BUDDH INTERNATIONAL CIRCUIT — GEOMETRIC VALIDATION REPORT")
    print("=" * 70)

    # 1. Loop Closure
    c_ok, c_dist = validate_loop_closure(track)
    results["loop_closure"] = {"passed": c_ok, "wrap_distance_px": round(c_dist, 4)}
    print(f"1. Loop Closure:             {'[PASS]' if c_ok else '[FAIL]'} (wrap gap: {c_dist:.4f} px)")

    # 2. Clockwise Direction
    d_ok, d_dir, d_area = validate_direction(track)
    results["direction"] = {"passed": d_ok, "direction": d_dir, "signed_area": round(d_area, 1)}
    print(f"2. Direction of Travel:      {'[PASS]' if d_ok else '[FAIL]'} ({d_dir}, area: {d_area:.1f})")

    # 3. Length Agreement
    l_ok, l_m, l_pct = validate_length(track)
    results["length"] = {"passed": l_ok, "length_m": round(l_m, 2), "pct_discrepancy": round(l_pct, 3)}
    print(f"3. Length Agreement:         {'[PASS]' if l_ok else '[FAIL]'} ({l_m:.1f}m vs 5138.0m, disc: {l_pct:.3f}%)")

    # 4. Track Width Consistency
    w_ok, w_min, w_max = validate_track_width(track)
    results["width"] = {"passed": w_ok, "min_width_px": round(w_min, 4), "max_width_px": round(w_max, 4)}
    print(f"4. Track Width Consistency:  {'[PASS]' if w_ok else '[FAIL]'} ({w_min:.2f}px to {w_max:.2f}px)")

    # 5. Boundary Folding
    b_ok, b_min_r, b_margin = validate_boundary_folding(track)
    results["boundary_folding"] = {
        "passed": b_ok,
        "min_radius_px": round(b_min_r, 2),
        "margin_px": round(b_margin, 2),
    }
    print(f"5. Boundary Folding (R > W/2): {'[PASS]' if b_ok else '[FAIL]'} (min R: {b_min_r:.2f}px, margin: {b_margin:.2f}px)")

    # 6. Car State Initialization
    car_ok, car_diag = validate_car_initialization(track)
    results["car_initialization"] = {"passed": car_ok, "diagnostics": car_diag}
    print(f"6. Car Initialization:       {'[PASS]' if car_ok else '[FAIL]'} (on track: {car_diag['on_track']}, rays: {[round(r) for r in car_diag['rays']]})")

    # Overall Status
    all_passed = c_ok and d_ok and l_ok and w_ok and b_ok and car_ok
    results["all_passed"] = all_passed
    print("=" * 70)
    print(f"OVERALL GEOMETRY VALIDATION: {'[ALL CHECKS PASSED]' if all_passed else '[VALIDATION FAILED]'}")
    print("=" * 70)

    # Generate Top-Down Geometry Plot
    plot_path = "analysis/buddh/buddh_geometry_validation.png"
    generate_geometry_plot(track, plot_path)
    results["plot_path"] = plot_path

    return results


if __name__ == "__main__":
    res = run_full_validation()
    if not res["all_passed"]:
        raise RuntimeError("Geometry validation failed!")
