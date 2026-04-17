
"""
Global gamma analysis between 2 ITK images.
If ddpercent is True, dd is taken as a percentage of the max TPS dose (not ideal if hotspots of 110%).
TODO: Modify for local gamma analysis.
"""

import numpy as np
import itk
from scipy.spatial import cKDTree


def get_gamma_index(ref, target, **kwargs):
    """
    Compare two 3D images using the gamma index formalism introduced by Daniel Low (1998).

    :param ref: Reference image (should behave like an ITK image object)
    :type ref: itk.Image or similar
    :param target: Target image (should behave like an ITK image object)
    :type target: itk.Image or similar
    :param **kwargs: Additional keyword arguments (see below)
    :return: Tuple of (global gamma image, local gamma image), with the same geometry as the target image
    :rtype: tuple of itk.Image

    **Keyword arguments:**

    - **dd**: Dose difference scale as a relative value, in units of percent
    - **ddpercent**: Boolean flag; True (default) means that dd is given in percent
    - **dta**: Distance to agreement in millimeters (e.g., 3mm)
    - **threshold**: Minimum dose value (exclusive) for calculating gamma values
    - **verbose**: Boolean flag; True will result in progress output
    """
    return gamma_index_3d(ref, target, **kwargs)


def closest_voxel_index(coord, origin, spacing):
    """
    Calculates the index of the closest voxel to a given coordinate in physical space.
    Accepts a single point (3,) or an array of points (N, 3).
    """
    return np.round((coord - origin) / spacing).astype(int)



def GetGamma(d0, d1, x0, x1, y0, y1, z0, z1, Max, dd, dta, gamma_method):
    """
    Calculates the gamma index between two dose points in 3D space.
    Accepts either scalars or numpy arrays for vectorised computation.
    """
    if gamma_method == 'local':
        norm_val = d0
    elif gamma_method == 'global':
        norm_val = Max
    else:
        raise ValueError("gamma_method must be 'local' or 'global'.")

    return np.sqrt(
        (d1 - d0) ** 2 / (0.01 * dd * norm_val) ** 2 +
        (x1 - x0) ** 2 / dta ** 2 +
        (y1 - y0) ** 2 / dta ** 2 +
        (z1 - z0) ** 2 / dta ** 2
    )


def gamma_index_3d(imgref, imgtarget, dta=3., dd=3., ddpercent=True,
                   threshold=10., defvalue=-1., verbose=False):
    """
    Computes the global and local gamma index between two 3D images.

    Before computation, below-threshold voxels are zeroed out in both arrays
    (shape and origin are unchanged). The KDTree is built only over above-threshold
    reference voxels, and only above-threshold target voxels are evaluated.
    Output gamma images are returned at the full original geometry with defvalue
    for all below-threshold voxels.

    Vectorised implementation:
      - Pass 1: Batch KDTree query for k nearest neighbours across ALL target
                voxels simultaneously, then evaluate gamma with NumPy array ops.
      - Pass 3: Trilinear interpolation fallback only for voxels that still fail
                after Pass 1.

    :param imgref: Reference image (ITK image object)
    :param imgtarget: Target image (ITK image object)
    :param dta: Distance to agreement criterion in mm (default 3.0)
    :param dd: Dose difference criterion in % (default 3.0)
    :param ddpercent: If True, dd is a percentage of max dose (default True)
    :param threshold: Minimum dose as % of max for gamma calculation (default 10.0)
    :param defvalue: Default gamma value for voxels outside calculation (default -1.0)
    :param verbose: If True, outputs progress information (default False)
    :return: Tuple of (global gamma image, local gamma image), geometry matching imgref
    :rtype: tuple of itk.Image
    """
    # NOTE: We swap ref <> target here to correct historical naming confusion.
    # After this swap:
    #   referenceImage = the dose we SEARCH over (ground truth / TPS)
    #   targetImage    = the dose we LOOP over and pass/fail each voxel (evaluated / MC)
    referenceImage = imgtarget
    targetImage    = imgref

    # Get arrays (swap axes from ITK zyx to xyz ordering)
    referenceArray = itk.array_view_from_image(referenceImage).swapaxes(0, 2).copy().astype(float)
    targetArray    = itk.array_view_from_image(targetImage).swapaxes(0, 2).copy().astype(float)

    if referenceArray.ndim != 3 or targetArray.ndim != 3:
        print("ERROR: Both images must be 3D.")
        return None

    referenceOrigin  = np.array(referenceImage.GetOrigin())
    referenceSpacing = np.array(referenceImage.GetSpacing())
    targetOrigin     = np.array(targetImage.GetOrigin())
    targetSpacing    = np.array(targetImage.GetSpacing())

    max_ref_dose = float(np.max(referenceArray))
    max_tgt_dose = float(np.max(targetArray))

    ref_threshold_abs = (threshold / 100.0) * max_ref_dose
    tgt_threshold_abs = (threshold / 100.0) * max_tgt_dose

    # -------------------------------------------------------------------------
    # Zero out below-threshold voxels in both arrays (shape and origin unchanged).
    # -------------------------------------------------------------------------
    tgt_mask = targetArray > tgt_threshold_abs

    if verbose:
        
        n_tgt_active = int(np.sum(tgt_mask))
        print(f"  Target:    {n_tgt_active:,} / {targetArray.size:,} voxels above threshold "
              f"({100 * n_tgt_active / targetArray.size:.1f}%)")

    # -------------------------------------------------------------------------
    # Initialise output gamma arrays (full size, matching targetImage geometry)
    # -------------------------------------------------------------------------
    gamma_array       = np.full(targetArray.shape, defvalue, dtype=float)
    gamma_array_local = np.full(targetArray.shape, defvalue, dtype=float)

    # -------------------------------------------------------------------------
    # Build REFERENCE coordinate grid — only above-threshold voxels enter the KDTree
    # -------------------------------------------------------------------------
    I, J, K = referenceArray.shape
    x_ref = referenceOrigin[0] + np.arange(I) * referenceSpacing[0]
    y_ref = referenceOrigin[1] + np.arange(J) * referenceSpacing[1]
    z_ref = referenceOrigin[2] + np.arange(K) * referenceSpacing[2]

    X_ref, Y_ref, Z_ref = np.meshgrid(x_ref, y_ref, z_ref, indexing='ij')
    ref_coords_all = np.column_stack([X_ref.ravel(), Y_ref.ravel(), Z_ref.ravel()])
    ref_doses_all  = referenceArray.ravel()

    
    ref_coords = ref_coords_all
    ref_doses  = ref_doses_all
    if verbose:
        print(f"  Building KDTree over {len(ref_coords):,} above-threshold reference voxels...")
    tree = cKDTree(ref_coords)

    # -------------------------------------------------------------------------
    # Build TARGET coordinate grid — extract above-threshold voxels for the loop
    # -------------------------------------------------------------------------
    I_t, J_t, K_t = targetArray.shape
    x_tgt = targetOrigin[0] + np.arange(I_t) * targetSpacing[0]
    y_tgt = targetOrigin[1] + np.arange(J_t) * targetSpacing[1]
    z_tgt = targetOrigin[2] + np.arange(K_t) * targetSpacing[2]

    X_tgt, Y_tgt, Z_tgt = np.meshgrid(x_tgt, y_tgt, z_tgt, indexing='ij')
    tgt_coords_all = np.column_stack([X_tgt.ravel(), Y_tgt.ravel(), Z_tgt.ravel()])
    tgt_doses_all  = targetArray.ravel()

    tgt_coords = tgt_coords_all[tgt_mask.ravel()]
    tgt_doses  = tgt_doses_all[tgt_mask.ravel()]

    N = len(tgt_coords)
    print(f"  Target voxels above {threshold}% threshold: {N:,} / {targetArray.size:,} "
          f"({100 * N / targetArray.size:.1f}%)")

    # Flat indices into the gamma arrays for the above-threshold target voxels
    all_flat_indices = np.where(tgt_mask.ravel())[0]

    # Mark all above-threshold voxels as initially failing (gamma = 1.1)
    np.put(gamma_array,       all_flat_indices, 1.1)
    np.put(gamma_array_local, all_flat_indices, 1.1)

    # -------------------------------------------------------------------------
    # Pass 1 (VECTORISED): Batch KDTree query + NumPy gamma evaluation
    # -------------------------------------------------------------------------
    K_NEIGHBOURS = 4

    if verbose:
        print(f"  Pass 1: querying {K_NEIGHBOURS} nearest neighbours for all {N} voxels...")

    # Single batched call — shape (N, K_NEIGHBOURS)
    _, nn_indices = tree.query(tgt_coords, k=K_NEIGHBOURS, workers=-1)

    # Neighbour coordinates and doses: (N, K, 3) and (N, K)
    nn_coords = ref_coords[nn_indices]
    nn_doses  = ref_doses[nn_indices]

    tgt_coords_exp = tgt_coords[:, np.newaxis, :]   # (N, 1, 3)
    tgt_doses_exp  = tgt_doses[:, np.newaxis]        # (N, 1)

    dist_sq   = np.sum((nn_coords - tgt_coords_exp) ** 2, axis=2)  # (N, K)
    dose_diff = nn_doses - tgt_doses_exp                            # (N, K)

    gamma_sq_global = (
        dist_sq / dta ** 2 +
        dose_diff ** 2 / (0.01 * dd * max_ref_dose) ** 2
    )
    gamma_sq_local = (
        dist_sq / dta ** 2 +
        dose_diff ** 2 / (0.01 * dd * tgt_doses_exp) ** 2
    )

    best_gamma_sq_global = np.min(gamma_sq_global, axis=1)
    best_gamma_sq_local  = np.min(gamma_sq_local,  axis=1)

    pass1_global = best_gamma_sq_global <= 1.0
    pass1_local  = best_gamma_sq_local  <= 1.0

    np.put(gamma_array,       all_flat_indices[pass1_global],
           np.sqrt(best_gamma_sq_global[pass1_global]))
    np.put(gamma_array_local, all_flat_indices[pass1_local],
           np.sqrt(best_gamma_sq_local[pass1_local]))

    n_pass1_global = int(np.sum(pass1_global))
    n_pass1_local  = int(np.sum(pass1_local))

    if verbose:
        print(f"  Pass 1 result — global: {n_pass1_global}/{N} pass, "
              f"local: {n_pass1_local}/{N} pass")

    # -------------------------------------------------------------------------
    # Pass 3 (trilinear fallback): only for voxels that failed at least one
    # -------------------------------------------------------------------------
    need_pass3   = ~(pass1_global & pass1_local)
    n_need_pass3 = int(np.sum(need_pass3))

    if verbose:
        print(f"  Pass 3: trilinear fallback for {n_need_pass3} voxels...")

    dd_abs_sq_global = (0.01 * dd * max_ref_dose) ** 2
    dta_sq           = dta ** 2

    p3_coords      = tgt_coords[need_pass3]
    p3_doses       = tgt_doses[need_pass3]
    p3_flat_idx    = all_flat_indices[need_pass3]
    p3_best_global = np.sqrt(best_gamma_sq_global[need_pass3])
    p3_best_local  = np.sqrt(best_gamma_sq_local[need_pass3])
    p3_need_global = ~pass1_global[need_pass3]
    p3_need_local  = ~pass1_local[need_pass3]

    gamma_count      = n_pass1_global
    gammaLocal_count = n_pass1_local

    for n in range(n_need_pass3):
        Xt, Yt, Zt = p3_coords[n]
        tgt_value  = p3_doses[n]
        flat_idx   = p3_flat_idx[n]

        dd_abs_sq_local = (0.01 * dd * tgt_value) ** 2

        ref_center_idx = closest_voxel_index(
            np.array([Xt, Yt, Zt]),
            referenceOrigin,
            referenceSpacing
        )

        gs_global, gs_local = get_best_gamma_squared_trilinear(
            target_pos=(Xt, Yt, Zt),
            target_dose=tgt_value,
            ref_center_idx=tuple(ref_center_idx),
            reference_array=referenceArray,
            ref_origin=referenceOrigin,
            ref_spacing=referenceSpacing,
            dta_squared=dta_sq,
            dd_absolute_squared_global=dd_abs_sq_global,
            dd_absolute_squared_local=dd_abs_sq_local,
            current_best_gamma=p3_best_global[n],
            current_best_gamma_local=p3_best_local[n],
            steps_per_voxel=5
        )

        if p3_need_global[n]:
            best = min(p3_best_global[n] ** 2, gs_global)
            if best <= 1.0:
                gamma_array.flat[flat_idx] = np.sqrt(best)
                gamma_count += 1

        if p3_need_local[n]:
            best = min(p3_best_local[n] ** 2, gs_local)
            if best <= 1.0:
                gamma_array_local.flat[flat_idx] = np.sqrt(best)
                gammaLocal_count += 1

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    n_evaluated_global = int(np.sum(gamma_array > 0))
    n_failing_global   = int(np.sum(gamma_array > 1))
    if n_evaluated_global > 0:
        print(f"Global Gamma pass rate = {100.0 - 100 * n_failing_global / n_evaluated_global:.2f}%")
    else:
        print("Global Gamma: no voxels evaluated.")

    if N > 0:
        print(f"Local Gamma pass rate  = {100 * gammaLocal_count / N:.2f}%")
    else:
        print("Local Gamma: no voxels evaluated.")

    # Convert full arrays back to ITK images (swap axes back to ITK zyx ordering)
    gimg = itk.image_from_array(gamma_array.swapaxes(0, 2).astype(np.float32).copy())
    gimg.CopyInformation(targetImage)

    gimg_local = itk.image_from_array(gamma_array_local.swapaxes(0, 2).astype(np.float32).copy())
    gimg_local.CopyInformation(targetImage)

    return gimg, gimg_local


def min_gamma(x1, x2, y1, y2, tx, ty, dd, dta, Max, spacing, gamma_method):
    """
    Finds the optimal position along a dimension to minimize the gamma index contribution.
    Returns (optimal_coordinate, interpolated_dose) or (False, False) if not between the two points.
    """
    grad = (y2 - y1) / (x2 - x1)
    c    = y2 - grad * x2

    if gamma_method == 'local':
        x_opto = (
            grad * ty / (dd * 0.01 * ty) ** 2 + tx / dta ** 2 - grad * c / (dd * 0.01 * ty) ** 2
        ) / (grad ** 2 / (dd * 0.01 * ty) ** 2 + 1 / dta ** 2)
    elif gamma_method == 'global':
        x_opto = (
            grad * ty / (dd * 0.01 * Max) ** 2 + tx / dta ** 2 - grad * c / (dd * 0.01 * Max) ** 2
        ) / (grad ** 2 / (dd * 0.01 * Max) ** 2 + 1 / dta ** 2)
    else:
        print('Please specify gamma method correctly, local or global.')
        return False, False

    if min(x1, x2) < x_opto < max(x1, x2):
        return x_opto, grad * x_opto + c
    return False, False


def get_best_gamma_squared_trilinear(
    target_pos,
    target_dose,
    ref_center_idx,
    reference_array,
    ref_origin,
    ref_spacing,
    dta_squared,
    dd_absolute_squared_global,
    dd_absolute_squared_local,
    current_best_gamma,
    current_best_gamma_local,
    steps_per_voxel=5
):
    """
    Phase 3: Full 3D trilinear interpolation search.

    Performs exhaustive search with trilinear interpolation to find minimum gamma.
    Samples at step size = voxel_resolution / steps_per_voxel in each dimension.

    :return: Tuple (best global gamma squared, best local gamma squared)
    """
    Xt, Yt, Zt = target_pos
    rx, ry, rz = ref_center_idx

    I, J, K = reference_array.shape

    x0 = max(0, rx - 1)
    x1 = min(I - 2, rx)
    y0 = max(0, ry - 1)
    y1 = min(J - 2, ry)
    z0 = max(0, rz - 1)
    z1 = min(K - 2, rz)

    if x1 < x0 or y1 < y0 or z1 < z0:
        return np.inf, np.inf

    best_gamma_sq_global = current_best_gamma ** 2       if current_best_gamma       <= 1 else np.inf
    best_gamma_sq_local  = current_best_gamma_local ** 2 if current_best_gamma_local <= 1 else np.inf

    for cube_x in range(x0, x1 + 1):
        for cube_y in range(y0, y1 + 1):
            for cube_z in range(z0, z1 + 1):
                D000 = reference_array[cube_x,     cube_y,     cube_z    ]
                D100 = reference_array[cube_x + 1, cube_y,     cube_z    ]
                D010 = reference_array[cube_x,     cube_y + 1, cube_z    ]
                D110 = reference_array[cube_x + 1, cube_y + 1, cube_z    ]
                D001 = reference_array[cube_x,     cube_y,     cube_z + 1]
                D101 = reference_array[cube_x + 1, cube_y,     cube_z + 1]
                D011 = reference_array[cube_x,     cube_y + 1, cube_z + 1]
                D111 = reference_array[cube_x + 1, cube_y + 1, cube_z + 1]

                cube_ox = ref_origin[0] + cube_x * ref_spacing[0]
                cube_oy = ref_origin[1] + cube_y * ref_spacing[1]
                cube_oz = ref_origin[2] + cube_z * ref_spacing[2]

                for step_x in range(steps_per_voxel + 1):
                    tx   = step_x / steps_per_voxel
                    omtx = 1.0 - tx
                    sx   = cube_ox + tx * ref_spacing[0]
                    dx   = sx - Xt
                    dx2  = dx * dx
                    if dx2 > dta_squared:
                        continue

                    for step_y in range(steps_per_voxel + 1):
                        ty   = step_y / steps_per_voxel
                        omty = 1.0 - ty
                        sy   = cube_oy + ty * ref_spacing[1]
                        dy   = sy - Yt
                        dxy2 = dx2 + dy * dy
                        if dxy2 > dta_squared:
                            continue

                        c00 = omtx * omty
                        c10 = tx   * omty
                        c01 = omtx * ty
                        c11 = tx   * ty

                        for step_z in range(steps_per_voxel + 1):
                            tz    = step_z / steps_per_voxel
                            omtz  = 1.0 - tz
                            sz    = cube_oz + tz * ref_spacing[2]
                            dz    = sz - Zt
                            dist2 = dxy2 + dz * dz

                            if dist2 > dta_squared * 4.0:
                                continue

                            interp = (
                                D000 * c00 * omtz +
                                D100 * c10 * omtz +
                                D010 * c01 * omtz +
                                D110 * c11 * omtz +
                                D001 * c00 * tz   +
                                D101 * c10 * tz   +
                                D011 * c01 * tz   +
                                D111 * c11 * tz
                            )

                            dd2  = (interp - target_dose) ** 2
                            gs_g = dist2 / dta_squared + dd2 / dd_absolute_squared_global
                            gs_l = dist2 / dta_squared + dd2 / dd_absolute_squared_local

                            if gs_g < best_gamma_sq_global:
                                best_gamma_sq_global = gs_g
                            if gs_l < best_gamma_sq_local:
                                best_gamma_sq_local = gs_l

                            if best_gamma_sq_global <= 1.0 and best_gamma_sq_local <= 1.0:
                                return best_gamma_sq_global, best_gamma_sq_local

    return best_gamma_sq_global, best_gamma_sq_local


# =============================================================================
# CLI
# =============================================================================

#!/usr/bin/env python
"""
Gamma index analysis CLI tool.

Usage:
    python gamma_index.py <dose1.mhd> [scale1] <dose2.mhd> [scale2] ... -G/-L -dta 3 -dd 3

Each dose file may optionally be followed by a numeric scale factor (default 1.0).
Computes gamma analysis for all permutations of input dose maps.

Examples:
    # No scaling
    python gamma_index.py dose_a.mhd dose_b.mhd -G -dta 3 -dd 3

    # Scale gate_dose by 200
    python gamma_index.py ref.mhd 1 gate_dose.mhd 200 -G -dta 3 -dd 3

    # Three files, local gamma, custom threshold
    python gamma_index.py d1.mhd d2.mhd d3.mhd -L -dta 2 -dd 2 -t 5
"""

import argparse
import os
import sys
from itertools import permutations
from pathlib import Path


def parse_dose_files_and_scales(tokens):
    """
    Parse an interleaved list of dose filenames and optional scale factors.

    Rules:
      - Each token that ends in .mhd starts a new dose entry.
      - If the very next token is a positive number (not a .mhd path), it is
        consumed as the scale factor for that dose.  Otherwise scale defaults to 1.0.

    Examples:
        ["a.mhd", "b.mhd"]           -> [("a.mhd", 1.0), ("b.mhd", 1.0)]
        ["a.mhd", "1", "b.mhd", "200"] -> [("a.mhd", 1.0), ("b.mhd", 200.0)]
        ["a.mhd", "b.mhd", "0.5"]    -> [("a.mhd", 1.0), ("b.mhd", 0.5)]
    """
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.lower().endswith('.mhd'):
            raise argparse.ArgumentTypeError(
                f"Expected a .mhd file path but got: '{token}'. "
                "Make sure all scale factors immediately follow their dose file."
            )
        path  = token
        scale = 1.0
        # Peek at the next token: consume it as a scale if it's a positive float
        if i + 1 < len(tokens) and not tokens[i + 1].lower().endswith('.mhd'):
            try:
                candidate = float(tokens[i + 1])
                if candidate <= 0:
                    raise ValueError("Scale factor must be positive.")
                scale = candidate
                i += 2
            except ValueError:
                i += 1
        else:
            i += 1
        result.append((path, scale))
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gamma index analysis between pairs of dose maps (.mhd files).\n\n"
            "Each dose file may be followed by an optional scale factor, e.g.:\n"
            "  python gamma_index.py ref.mhd 1 gate_dose.mhd 200 -G -dta 3 -dd 3 -t 10"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "dose_tokens",
        nargs="+",
        help=(
            "Dose .mhd files, each optionally followed by a numeric scale factor. "
            "Example: ref.mhd 1 gate_dose.mhd 200"
        ),
    )
    parser.add_argument("-G",  action="store_true", dest="global_gamma",
                        help="Compute global gamma.")
    parser.add_argument("-L",  action="store_true", dest="local_gamma",
                        help="Compute local gamma.")
    parser.add_argument("-dta", type=float, default=3.0,
                        help="Distance to agreement in mm (default: 3).")
    parser.add_argument("-dd",  type=float, default=3.0,
                        help="Dose difference in %% (default: 3).")
    parser.add_argument("-o", "--output-dir", type=str, default=".",
                        help="Output directory (default: current directory).")
    parser.add_argument("-t", "--threshold", type=float, default=10.0,
                        help="Dose threshold %% of max (default: 10.0).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output.")

    args = parser.parse_args()

    # Parse interleaved filenames and scale factors from positional tokens
    try:
        args.dose_files = parse_dose_files_and_scales(args.dose_tokens)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    if len(args.dose_files) < 2:
        parser.error("At least two dose files are required.")

    #if not args.global_gamma and not args.local_gamma:
    #    parser.error("At least one of -G (global) or -L (local) must be specified.")

    for path, _scale in args.dose_files:
        if not os.path.isfile(path):
            parser.error(f"File not found: {path}")
        if not path.lower().endswith(".mhd"):
            parser.error(f"File is not .mhd: {path}")

    return args


def make_output_filename(ref_path, tgt_path, method, dta, dd, ref_scale, tgt_scale):
    """
    Build an output filename from the two input dose file stems and their scale factors.

    Scale factors are included in the name only when they differ from 1.0, e.g.:
        ref_gate_dose_x200_gamma_G_dta3.0_dd3.0.mhd
    """
    ref_stem = Path(ref_path).stem
    tgt_stem = Path(tgt_path).stem
    method_label = "G" if method == "global" else "L"

    ref_part = ref_stem if ref_scale == 1.0 else f"{ref_stem}_x{ref_scale:g}"
    tgt_part = tgt_stem if tgt_scale == 1.0 else f"{tgt_stem}_x{tgt_scale:g}"

    return f"{ref_part}_{tgt_part}_gamma_{method_label}_dta{dta}_dd{dd}.mhd"


def run_gamma_pair(ref_path, ref_scale, tgt_path, tgt_scale, args):
    """Load, optionally scale, and run gamma analysis for one (ref, target) pair."""
    print(f"\n{'='*70}")
    print(f"  Reference : {ref_path}  (scale × {ref_scale:g})")
    print(f"  Target    : {tgt_path}  (scale × {tgt_scale:g})")
    print(f"  DTA={args.dta} mm | DD={args.dd}% | Threshold={args.threshold}%")
    print(f"{'='*70}")

    print("  Loading reference dose...")
    ref_img = itk.imread(ref_path, itk.F)
    print(f"    Size: {itk.size(ref_img)}, Spacing: {itk.spacing(ref_img)}")

    print("  Loading target dose...")
    tgt_img = itk.imread(tgt_path, itk.F)
    print(f"    Size: {itk.size(tgt_img)}, Spacing: {itk.spacing(tgt_img)}")

    # ---- Apply scale factors ------------------------------------------------
    if ref_scale != 1.0:
        print(f"  Scaling reference by {ref_scale:g}...")
        ref_arr = itk.array_from_image(ref_img).astype(np.float32) * float(ref_scale)
        ref_img_scaled = itk.image_from_array(ref_arr)
        ref_img_scaled.CopyInformation(ref_img)
        ref_img = ref_img_scaled

    if tgt_scale != 1.0:
        print(f"  Scaling target by {tgt_scale:g}...")
        tgt_arr = itk.array_from_image(tgt_img).astype(np.float32) * float(tgt_scale)
        tgt_img_scaled = itk.image_from_array(tgt_arr)
        tgt_img_scaled.CopyInformation(tgt_img)
        tgt_img = tgt_img_scaled

    # ---- Run gamma ----------------------------------------------------------
    print("  Computing gamma index...")
    gamma_global_img, gamma_local_img = get_gamma_index(
        ref=ref_img,
        target=tgt_img,
        dta=args.dta,
        dd=args.dd,
        ddpercent=True,
        threshold=args.threshold,
        verbose=args.verbose,
    )

    results = {}
    if args.global_gamma or args.local_gamma:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.global_gamma:
        out_name = make_output_filename(ref_path, tgt_path, "global",
                                        args.dta, args.dd, ref_scale, tgt_scale)
        out_path = os.path.join(args.output_dir, out_name)
        print(f"  Saving global gamma map -> {out_path}")
        itk.imwrite(gamma_global_img, out_path)
        results["global"] = out_path

    if args.local_gamma:
        out_name = make_output_filename(ref_path, tgt_path, "local",
                                        args.dta, args.dd, ref_scale, tgt_scale)
        out_path = os.path.join(args.output_dir, out_name)
        print(f"  Saving local gamma map  -> {out_path}")
        itk.imwrite(gamma_local_img, out_path)
        results["local"] = out_path

    return results


def main():
    args = parse_args()

    dose_files = args.dose_files          # list of (path, scale)
    pairs      = list(permutations(dose_files, 2))
    pairs = pairs[0:1]
    print(f"\n{'#'*70}")
    print(f"  GAMMA INDEX ANALYSIS")
    print(f"  {len(dose_files)} dose files -> {len(pairs)} permutations")
    print(f"  Methods: {'Global ' if args.global_gamma else ''}{'Local' if args.local_gamma else ''}")
    print(f"  DTA = {args.dta} mm | DD = {args.dd}%")
    for path, scale in dose_files:
        print(f"    {path}  (scale × {scale:g})")
    print(f"{'#'*70}")

    all_results = {}

    for i, ((ref_path, ref_scale), (tgt_path, tgt_scale)) in enumerate(pairs, 1):
        print(f"\n>>> Pair {i}/{len(pairs)}")
        try:
            result = run_gamma_pair(ref_path, ref_scale, tgt_path, tgt_scale, args)
            all_results[(ref_path, tgt_path)] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[(ref_path, tgt_path)] = {"error": str(e)}

    print(f"\n\n{'#'*70}")
    print("  SUMMARY")
    print(f"{'#'*70}")
    for (ref, tgt), result in all_results.items():
        ref_s = Path(ref).stem
        tgt_s = Path(tgt).stem
        if "error" in result:
            print(f"  {ref_s} vs {tgt_s}: FAILED - {result['error']}")
        else:
            saved = ", ".join(result.values())
            print(f"  {ref_s} vs {tgt_s}: {saved}")

    print(f"\nDone. {len(all_results)} pairs processed.")


if __name__ == "__main__":
    main()