#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spherical_surface.py — Plot MHD Rossby waves on spherical surface

Reconstructs the full 2D field from the 1D mu-profile stored by Dedalus:

    f(theta, phi) = Re[ (A(mu) + i*B(mu)) * exp(i*m*phi) ]
                  = A(mu)*cos(m*phi) - B(mu)*sin(m*phi)

where A = real part and B = imaginary part of the stored complex profile.

For a field like 'u_theta_real' (stored as Re(u_theta)), the script
automatically loads the companion 'u_theta_imag' so the full phase
information is included.

Usage:
    # Single frame (default field: u_theta_real)
    python3 plot_spherical_surface.py snapshots/*.h5 --time-index 0

    # Single frame of a specific field
    python3 plot_spherical_surface.py snapshots/*.h5 --field u_phi_real --time-index 50

    # Generate frames for a movie
    python3 plot_spherical_surface.py snapshots/*.h5 --frames --field u_theta_real --time-step 1.0

    # With parameters shown
    python3 plot_spherical_surface.py snapshots/*.h5 --input input.json --show-params
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os
import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_input_json(json_path):
    """Read parameters from input JSON file."""
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        print(f"Loaded parameters from {json_path}")
        return params
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load JSON file: {e}")
        return None


def field_to_latex(field_name):
    """Convert a field name like 'u_theta_real' to a LaTeX string."""
    # symbol_map: base name -> (base_letter, subscript_letter, has_greek_sub)
    # For fields like u_theta: render as $u_{\theta, r}$ — one subscript block
    # combining the Greek and the real/imag suffix, avoiding double-subscript.
    symbol_map = {
        'u_theta': (r'u',   r'\theta', True),
        'u_phi':   (r'u',   r'\phi',   True),
        'b_theta': (r'b',   r'\theta', True),
        'b_phi':   (r'b',   r'\phi',   True),
        'varphi':  (r'\varphi', None,  False),
        'psi':     (r'\Psi',    None,  False),
        'phi':     (r'\Phi',    None,  False),
    }
    suffix_map = {'real': 'r', 'imag': 'i'}

    lower = field_name.lower()
    base = None
    greek = None
    has_greek = False
    matched_len = 0
    for key, (b, g, hg) in symbol_map.items():
        if lower.startswith(key) and len(key) > matched_len:
            base = b
            greek = g
            has_greek = hg
            matched_len = len(key)

    if base is None:
        return field_name

    remainder = lower[matched_len:].lstrip('_')

    if has_greek:
        # Combine Greek + suffix into one subscript: u_{\theta, r}
        if remainder == 'abs':
            return rf'$|{base}_{{{greek}}}|$'
        elif remainder in suffix_map:
            s = suffix_map[remainder]
            return rf'${base}_{{{greek}, {s}}}$'
        elif not remainder:
            return rf'${base}_{{{greek}}}$'
        else:
            return rf'${base}_{{{greek}, {remainder}}}$'
    else:
        # Simple symbol like \Psi, \Phi — no Greek sub, just append suffix
        if remainder == 'abs':
            return rf'$|{base}|$'
        elif remainder in suffix_map:
            s = suffix_map[remainder]
            return rf'${base}_{{{s}}}$'
        elif not remainder:
            return rf'${base}$'
        else:
            return rf'${base}_{{{remainder}}}$'


def companion_field(field_name):
    """
    Return the companion imaginary-part field name for a _real field, or None.
    e.g. 'u_theta_real' -> 'u_theta_imag'
         'u_theta_imag' -> 'u_theta_real'
         'Psi_real'     -> 'Psi_imag'
         'u_theta_abs'  -> None   (no companion needed)
         'u_theta'      -> None   (complex field, handled directly)
    """
    if field_name.endswith('_real'):
        return field_name[:-5] + '_imag'
    if field_name.endswith('_imag'):
        return field_name[:-5] + '_real'
    return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def auto_detect_scale(field_values_flat):
    """
    Auto-detect the linear-to-log crossover scale for arcsinh/symlog normalization.

    Uses the 10th percentile of non-zero absolute values across the supplied
    data sample — this captures the 'typical small amplitude' of the field
    without being sensitive to the overall magnitude or parameters.

    Works for any field and any parameter set without manual tuning.
    """
    abs_vals = np.abs(field_values_flat)
    nonzero = abs_vals[abs_vals > 0]
    if len(nonzero) == 0:
        return 1.0
    scale = float(np.percentile(nonzero, 10))
    return max(scale, 1e-30)  # guard against underflow


def build_norm(field_values_flat, norm_type, vmin=None, vmax=None):
    """
    Build a matplotlib Normalize object for the given norm type.

    Parameters
    ----------
    field_values_flat : 1D array of all field values to be displayed
    norm_type : 'arcsinh' | 'symlog' | 'linear'
    vmin, vmax : optional manual color limits (symmetric assumed if None)

    Returns
    -------
    norm : matplotlib Normalize subclass
    vmin, vmax : the actual limits used
    """
    import matplotlib.colors as mcolors

    if vmin is None or vmax is None:
        vmax_calc = np.max(np.abs(field_values_flat))
        vmin = -vmax_calc if vmin is None else vmin
        vmax =  vmax_calc if vmax is None else vmax

    if norm_type == 'arcsinh':
        scale = auto_detect_scale(field_values_flat)
        print(f"  Arcsinh norm: linear_width (scale) = {scale:.3e}  "
              f"(auto-detected from 10th percentile of |field|)")
        try:
            norm = mcolors.AsinhNorm(linear_width=scale, vmin=vmin, vmax=vmax)
        except AttributeError:
            # Fallback for matplotlib < 3.2
            print("  Warning: AsinhNorm not available in this matplotlib version; "
                  "falling back to SymLogNorm")
            norm = mcolors.SymLogNorm(linthresh=scale, vmin=vmin, vmax=vmax, base=10)

    elif norm_type == 'symlog':
        scale = auto_detect_scale(field_values_flat)
        print(f"  SymLog norm: linthresh = {scale:.3e}  "
              f"(auto-detected from 10th percentile of |field|)")
        norm = mcolors.SymLogNorm(linthresh=scale, vmin=vmin, vmax=vmax, base=10)

    else:  # linear
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    return norm, vmin, vmax



def read_all_snapshots(file_paths):
    """Read all snapshot data from HDF5 files."""
    h5_files = []
    for pattern in file_paths:
        expanded = glob.glob(pattern)
        if expanded:
            h5_files.extend(expanded)
        elif os.path.exists(pattern):
            h5_files.append(pattern)

    h5_files = sorted(set(h5_files))
    if not h5_files:
        raise FileNotFoundError("No .h5 files found")

    print(f"Found {len(h5_files)} snapshot files")

    # Get mu grid and available tasks from the first file
    with h5py.File(h5_files[0], 'r') as f:
        mu = None
        for scale_name in f['scales'].keys():
            if 'mu' in scale_name.lower():
                mu = np.array(f['scales'][scale_name][:])
                break
        if mu is None:
            raise KeyError("Could not find mu coordinates in HDF5 file")
        tasks = list(f['tasks'].keys())

    print(f"Available fields: {tasks}")

    # Read all time snapshots
    all_times = []
    all_data = {}

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            times = f['scales/sim_time'][:]
            all_times.extend(times)
            for task in tasks:
                if task not in all_data:
                    all_data[task] = []
                task_data = f['tasks'][task][:]
                for i in range(task_data.shape[0]):
                    all_data[task].append(task_data[i, :])

    all_times = np.array(all_times)
    for task in all_data:
        all_data[task] = np.array(all_data[task])

    print(f"Loaded {len(all_times)} time snapshots")
    return mu, all_times, all_data


# ---------------------------------------------------------------------------
# Field reconstruction
# ---------------------------------------------------------------------------

def reconstruct_2d_field(field_name, time_index, all_data, mu, m, nphi=360,
                         background=None):
    """
    Reconstruct the full 2D field on (theta, phi) from stored 1D mu-profiles.

    The correct formula is:
        f(theta, phi) = Re[ (A(mu) + i*B(mu)) * exp(i*m*phi) ]
                      = A(mu)*cos(m*phi) - B(mu)*sin(m*phi)

    For '_real' fields: A is the stored profile, B is loaded from companion '_imag'.
    For '_imag' fields: A=0, B is the stored profile  ->  -B(mu)*sin(m*phi)
    For '_abs'  fields: A is the stored profile, B=0  ->  A(mu)*cos(m*phi)
                        (absolute value has no meaningful phase partner)
    For complex fields (e.g. 'Psi'): split into real/imag directly.

    background : optional 1D array (Nmu,)
        A real, phi-independent offset to be added after wave reconstruction.
        Used for u_phi, which contains a differential-rotation background term
        (-s2*mu^2*R - s4*mu^4*R) that must NOT be multiplied by cos/sin(m*phi).
        The background is stripped from A before wave reconstruction, then
        added back as a uniform column.
    """
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)

    stored = all_data[field_name][time_index, :]  # shape (Nmu,)

    if field_name.endswith('_real'):
        A = np.real(stored)
        comp = companion_field(field_name)
        if comp and comp in all_data:
            B = np.real(all_data[comp][time_index, :])
        else:
            B = np.zeros_like(A)
            print(f"  Warning: companion field '{comp}' not found; "
                  f"reconstruction will be incomplete (sin term missing)")

    elif field_name.endswith('_imag'):
        A = np.zeros(len(mu))
        B = np.real(stored)

    elif field_name.endswith('_abs'):
        A = np.real(stored)
        B = np.zeros_like(A)

    else:
        # Complex field stored directly (e.g. 'Psi', 'u_theta')
        if np.iscomplexobj(stored):
            A = np.real(stored)
            B = np.imag(stored)
        else:
            A = stored.astype(float)
            B = np.zeros_like(A)

    # Strip background from A so only the wave part is multiplied by cos/sin
    if background is not None:
        A_wave = A - background
    else:
        A_wave = A

    # Full 2D wave reconstruction:  A_wave*cos(m*phi) - B*sin(m*phi)
    cos_mphi = np.cos(m * phi)   # shape (nphi,)
    sin_mphi = np.sin(m * phi)   # shape (nphi,)

    field_2d = (A_wave[:, np.newaxis] * cos_mphi[np.newaxis, :]
                - B[:, np.newaxis] * sin_mphi[np.newaxis, :])  # shape (Nmu, nphi)

    # Add phi-independent background back as a uniform offset
    if background is not None:
        field_2d += background[:, np.newaxis]

    return field_2d


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spherical_surface(ax, field_2d, mu, phi,
                           cmap='RdBu_r', norm=None):
    """Plot field on 3D spherical surface."""
    theta = np.arccos(np.clip(mu, -1, 1))
    theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')

    x = np.sin(theta_2d) * np.cos(phi_2d)
    y = np.sin(theta_2d) * np.sin(phi_2d)
    z = np.cos(theta_2d)

    if norm is None:
        import matplotlib.colors as mcolors
        vmax_calc = np.max(np.abs(field_2d))
        norm = mcolors.Normalize(vmin=-vmax_calc, vmax=vmax_calc)

    try:
        cmap_obj = plt.colormaps[cmap]
    except (AttributeError, KeyError):
        cmap_obj = cm.get_cmap(cmap)

    ax.plot_surface(x, y, z, facecolors=cmap_obj(norm(field_2d)),
                    rstride=1, cstride=1, shade=False, alpha=0.95)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_axis_off()

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(field_2d)
    return mappable


def plot_2d_projection(ax, field_2d, mu, phi,
                       cmap='RdBu_r', norm=None):
    """Plot 2D Mollweide projection of spherical field."""
    # Mollweide expects longitude in [-pi, pi] and latitude in [-pi/2, pi/2]
    lon = phi - np.pi
    lat = np.pi / 2 - np.arccos(np.clip(mu, -1, 1))

    lon_2d, lat_2d = np.meshgrid(lon, lat)

    if norm is None:
        import matplotlib.colors as mcolors
        vmax_calc = np.max(np.abs(field_2d))
        norm = mcolors.Normalize(vmin=-vmax_calc, vmax=vmax_calc)

    im = ax.pcolormesh(lon_2d, lat_2d, field_2d,
                       cmap=cmap, norm=norm, shading='auto')

    ax.grid(True, alpha=0.3)
    return im


def create_figure(field_2d, mu, phi, m, time, field_name, projection,
                  cmap, norm, params=None, show_params=False):
    """Create figure with 3D globe, Mollweide, or both.

    In 'both' mode a single shared colorbar is placed on the right side,
    using the same norm object for both subplots.
    """
    latex_title = rf'{field_to_latex(field_name)} at $t = {time:.2f}$'

    if projection == 'both':
        # Leave room on the right for the shared colorbar
        fig = plt.figure(figsize=(17, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='mollweide')
    elif projection == '3d':
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111, projection='3d')
        ax2 = None
    else:  # mollweide
        fig = plt.figure(figsize=(12, 6))
        ax1 = None
        ax2 = fig.add_subplot(111, projection='mollweide')

    # Single centered title for the whole figure
    fig.suptitle(latex_title, fontsize=20, y=0.97, ha='center')

    # Plot both subplots with the same norm — no individual subplot titles
    mappable = None
    if ax1 is not None:
        mappable = plot_spherical_surface(ax1, field_2d, mu, phi,
                                          cmap=cmap, norm=norm)
    if ax2 is not None:
        mappable = plot_2d_projection(ax2, field_2d, mu, phi,
                                      cmap=cmap, norm=norm)

    # Colorbar — single shared one on the right for 'both', per-plot otherwise
    if projection == 'both':
        # Reserve right margin for colorbar, then place it manually
        fig.subplots_adjust(left=0.03, right=0.88, top=0.92,
                            bottom=0.05, wspace=0.15)
        cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.74])  # [left, bottom, w, h]
        fig.colorbar(mappable, cax=cbar_ax)
    elif projection == '3d':
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92,
                            bottom=0.05, wspace=0.1)
        fig.colorbar(mappable, ax=ax1, shrink=0.5, pad=0.1)
    else:
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92,
                            bottom=0.05, wspace=0.1)
        fig.colorbar(mappable, ax=ax2, fraction=0.046, pad=0.04)

    # Parameter text box
    if params and show_params:
        param_text = (
            f"$s_2 = {params.get('s2', 'N/A')}$, "
            f"$s_4 = {params.get('s4', 'N/A')}$\n"
            f"$\\beta^2 = {params.get('beta_sq', 'N/A')}$\n"
            f"IC: $A = {params.get('IC_amp', 'N/A')}$"
        )
        fig.text(0.90, 0.97, param_text,
                 transform=fig.transFigure,
                 fontsize=13,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 family='monospace')

    return fig


# ---------------------------------------------------------------------------
# Frame generators
# ---------------------------------------------------------------------------

def uphi_background(mu, params):
    """
    Return the real, phi-independent differential-rotation background of u_phi:
        background(mu) = -s2*mu^2*R - s4*mu^4*R
    Returns None if params is missing or lacks the required keys.
    """
    if params is None:
        return None
    s2 = params.get('s2')
    s4 = params.get('s4')
    R  = params.get('R')
    if any(v is None for v in (s2, s4, R)):
        print("  Warning: s2/s4/R not found in params; u_phi background not subtracted")
        return None
    return (( - s2 * mu**2 * R - s4 * mu**4 * R) * np.sqrt(1 - mu**2)).astype(float)


def get_background(field_name, mu, params):
    """Return the phi-independent background for fields that need it, else None."""
    base = field_name.lower()
    # u_phi and all its variants (u_phi_real, u_phi_imag, u_phi_abs) carry
    # the differential-rotation background.
    if base == 'u_phi' or base.startswith('u_phi_'):
        return uphi_background(mu, params)
    return None


def apply_display_convention(field_2d, field_name):
    """
    Apply sign convention for display in geographic latitude coordinates.

    u_theta is stored in co-latitude convention (positive = southward).
    For display in geographic latitude convention (positive = poleward/northward),
    flip the sign: u_lat = -u_theta.
    """
    base = field_name.lower()
    if base == 'u_theta' or base.startswith('u_theta_'):
        return -field_2d
    return field_2d


def generate_single_frame(args, m, mu, all_times, all_data, params=None):
    """Generate a single output frame."""
    idx = args.time_index if args.time_index >= 0 else len(all_times) + args.time_index

    if not (0 <= idx < len(all_times)):
        print(f"Error: Time index {args.time_index} out of range "
              f"[0, {len(all_times) - 1}]")
        return

    if args.field not in all_data:
        print(f"Error: Field '{args.field}' not found.")
        print(f"Available fields: {list(all_data.keys())}")
        return

    time = all_times[idx]
    phi = np.linspace(0, 2 * np.pi, args.nphi, endpoint=False)
    print(f"Plotting t = {time:.3f}  (index {idx})")

    bg = get_background(args.field, mu, params)
    if bg is not None:
        print(f"  Applying differential-rotation background for u_phi")

    field_2d = reconstruct_2d_field(
        args.field, idx, all_data, mu, m, args.nphi, background=bg)
    field_2d = apply_display_convention(field_2d, args.field)

    print(f"  Field range: [{field_2d.min():.3e}, {field_2d.max():.3e}]")

    norm, vmin, vmax = build_norm(field_2d.ravel(), args.norm,
                                  vmin=args.vmin, vmax=args.vmax)

    fig = create_figure(field_2d, mu, phi, m, time, args.field,
                        args.projection, args.cmap,
                        norm, params, args.show_params)

    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved to {args.output}")
    plt.close()


def generate_frames(args, m, mu, all_times, all_data, params=None):
    """Generate multiple frames for movie creation."""
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Frames will be saved to: {args.output_dir}/")

    if args.field not in all_data:
        print(f"Error: Field '{args.field}' not found.")
        print(f"Available fields: {list(all_data.keys())}")
        return

    time_start = args.time_start if args.time_start is not None else all_times.min()
    time_end   = args.time_end   if args.time_end   is not None else all_times.max()

    print(f"Time range: [{time_start:.2f}, {time_end:.2f}]  step={args.time_step}")

    phi = np.linspace(0, 2 * np.pi, args.nphi, endpoint=False)

    # Select frame indices
    selected = []
    t = time_start
    while t <= time_end + 1e-10:
        idx = np.argmin(np.abs(all_times - t))
        if abs(all_times[idx] - t) < args.time_step * 0.5:
            selected.append((idx, all_times[idx]))
        t += args.time_step

    print(f"Will generate {len(selected)} frames")

    bg = get_background(args.field, mu, params)
    if bg is not None:
        print(f"  Applying differential-rotation background for u_phi")

    # Always sample across all frames to build a stable global norm.
    # This ensures the colorbar is fixed for the entire movie.
    print("Computing global color scale and normalization...")
    sample = selected[::max(1, len(selected) // 20)]  # up to ~20 sample frames
    all_vals = []
    for idx, _ in sample:
        f2d = reconstruct_2d_field(args.field, idx, all_data, mu, m, args.nphi,
                                   background=bg)
        f2d = apply_display_convention(f2d, args.field)
        all_vals.append(f2d.ravel())
    all_vals = np.concatenate(all_vals)

    norm, vmin_g, vmax_g = build_norm(all_vals, args.norm,
                                      vmin=args.vmin, vmax=args.vmax)
    print(f"  Color limits: [{vmin_g:.3e}, {vmax_g:.3e}]")

    for frame_num, (idx, time) in enumerate(selected):
        field_2d = reconstruct_2d_field(
            args.field, idx, all_data, mu, m, args.nphi, background=bg)
        field_2d = apply_display_convention(field_2d, args.field)

        fig = create_figure(field_2d, mu, phi, m, time, args.field,
                            args.projection, args.cmap,
                            norm, params, args.show_params)

        fname = os.path.join(args.output_dir, f"frame_{frame_num:05d}.png")
        plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')
        plt.close()

        if frame_num == 0 or (frame_num + 1) % 10 == 0:
            print(f"  Frame {frame_num + 1}/{len(selected)}  t = {time:.2f}")

    print(f"\nAll {len(selected)} frames saved to {args.output_dir}/")
    print(f"Create movie with:  python3 make_movie.py {args.output_dir}/ --fps 30")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot MHD Rossby wave fields on spherical surface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 plot_spherical_surface.py snapshots/*.h5
  python3 plot_spherical_surface.py snapshots/*.h5 --field u_phi_real --time-index 50
  python3 plot_spherical_surface.py snapshots/*.h5 --frames --field u_theta_real \\
      --time-step 1.0 --fixed-colorbar
        """
    )
    parser.add_argument('files', nargs='+', help='Snapshot .h5 files')
    parser.add_argument('--field', type=str, default='u_theta_real',
                        help='Field to plot (default: u_theta_real)')
    parser.add_argument('--frames', action='store_true',
                        help='Generate multiple frames for a movie')
    parser.add_argument('--time-index', type=int, default=0,
                        help='Time index for single frame (default: 0)')
    parser.add_argument('--time-step', type=float, default=1.0,
                        help='Time interval between frames (default: 1.0)')
    parser.add_argument('--time-start', type=float, default=None,
                        help='Start time for frames (default: first snapshot)')
    parser.add_argument('--time-end', type=float, default=None,
                        help='End time for frames (default: last snapshot)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input JSON file with simulation parameters')
    parser.add_argument('--m', type=int, default=None,
                        help='Azimuthal wavenumber m (overrides JSON value)')
    parser.add_argument('--output', '-o', type=str,
                        default='spherical_surface.png',
                        help='Output filename for single frame')
    parser.add_argument('--output-dir', type=str, default='frames',
                        help='Output directory for frame mode')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Figure DPI (default: 150)')
    parser.add_argument('--nphi', type=int, default=360,
                        help='Number of phi grid points (default: 360)')
    parser.add_argument('--projection', type=str, default='both',
                        choices=['3d', 'mollweide', 'both'],
                        help='Plot projection (default: both)')
    parser.add_argument('--cmap', type=str, default='RdBu_r',
                        help='Colormap (default: RdBu_r)')
    parser.add_argument('--norm', type=str, default='arcsinh',
                        choices=['arcsinh', 'symlog', 'linear'],
                        help='Color normalization: arcsinh (default, best for waves), '
                             'symlog, or linear. Scale is auto-detected from data.')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum color value (default: auto from global max)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum color value (default: auto from global max)')
    parser.add_argument('--show-params', action='store_true',
                        help='Display simulation parameters on the plot')

    args = parser.parse_args()

    # Load parameters
    params = None
    m = args.m
    if args.input:
        params = read_input_json(args.input)
        if params and m is None:
            m = params.get('m', 1)

    if m is None:
        m = 1
        print(f"Warning: m not specified; using default m={m}")
    else:
        print(f"Azimuthal wavenumber m = {m}")

    # Load data
    print("Loading data...")
    mu, all_times, all_data = read_all_snapshots(args.files)
    print(f"Loaded {len(all_times)} snapshots, {len(mu)} mu points\n")

    # Check requested field exists
    if args.field not in all_data:
        print(f"Error: field '{args.field}' not found in snapshots.")
        print(f"Available fields: {list(all_data.keys())}")
        return

    # Report companion field status
    comp = companion_field(args.field)
    if comp:
        if comp in all_data:
            print(f"Reconstruction: '{args.field}' + '{comp}'  "
                  f"->  A*cos(m*phi) - B*sin(m*phi)")
        else:
            print(f"Warning: companion '{comp}' not found. "
                  f"Using cos term only.")
    print()

    if args.frames:
        generate_frames(args, m, mu, all_times, all_data, params)
    else:
        generate_single_frame(args, m, mu, all_times, all_data, params)


if __name__ == "__main__":
    main()