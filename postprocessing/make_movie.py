#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_movie.py — Assemble PNG frames into a movie

Usage:
    python3 make_movie.py frames/
    python3 make_movie.py frames/ --fps 24 --output wave_movie.mp4
    python3 make_movie.py frames/ --fps 10 --pattern "frame_%05d.png"
    python3 make_movie.py frames/ --fps 30 --codec libx265 --quality 20
"""

import argparse
import os
import subprocess
import sys
import glob


def find_frames(frames_dir, pattern):
    """Check that frames matching the pattern exist in the directory."""
    search = os.path.join(frames_dir, pattern.replace('%05d', '*')
                                               .replace('%04d', '*')
                                               .replace('%03d', '*')
                                               .replace('%d', '*'))
    found = sorted(glob.glob(search))
    return found


def check_ffmpeg():
    """Verify ffmpeg is available on PATH."""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                                capture_output=True, text=True)
        # Extract just the version line for a tidy print
        first_line = result.stdout.splitlines()[0] if result.stdout else 'unknown'
        print(f"ffmpeg found: {first_line}")
        return True
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install it first:")
        print("  Ubuntu/Debian : sudo apt install ffmpeg")
        print("  macOS (brew)  : brew install ffmpeg")
        print("  Conda         : conda install -c conda-forge ffmpeg")
        return False


def build_ffmpeg_command(args, input_pattern):
    """Construct the ffmpeg command from parsed arguments."""
    cmd = [
        'ffmpeg',
        '-y',                          # overwrite output without asking
        '-framerate', str(args.fps),   # input frame rate
        '-i', input_pattern,           # input pattern
    ]

    # Video codec and quality
    if args.codec == 'libx264':
        cmd += [
            '-c:v', 'libx264',
            '-crf', str(args.quality),      # 0=lossless, 23=default, 51=worst
            '-preset', args.preset,
            '-pix_fmt', 'yuv420p',          # required for broad compatibility
        ]
    elif args.codec == 'libx265':
        cmd += [
            '-c:v', 'libx265',
            '-crf', str(args.quality),
            '-preset', args.preset,
            '-pix_fmt', 'yuv420p',
            '-tag:v', 'hvc1',               # makes .mp4 play in QuickTime/macOS
        ]
    elif args.codec == 'vp9':
        cmd += [
            '-c:v', 'libvpx-vp9',
            '-crf', str(args.quality),
            '-b:v', '0',                    # required for CRF mode in vp9
            '-pix_fmt', 'yuv420p',
        ]
    else:
        # User supplied a raw codec string
        cmd += ['-c:v', args.codec]

    # Build -vf filter chain.
    # H.264/H.265 require even width & height — crop silently to even dimensions.
    even_filter = 'crop=trunc(iw/2)*2:trunc(ih/2)*2'
    if args.scale:
        vf = f'scale={args.scale},{even_filter}'
    else:
        vf = even_filter
    cmd += ['-vf', vf]

    cmd.append(args.output)
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description='Assemble PNG frames into a movie via ffmpeg',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default settings (30 fps, H.264)
  python3 make_movie.py frames/

  # Slow motion playback at 10 fps
  python3 make_movie.py frames/ --fps 10

  # Custom output name and frame rate
  python3 make_movie.py frames/ --fps 24 --output rossby_wave.mp4

  # High quality H.265 (smaller file, same quality)
  python3 make_movie.py frames/ --fps 30 --codec libx265 --quality 18

  # If your frames are numbered differently (e.g. frame_0001.png)
  python3 make_movie.py frames/ --pattern "frame_%04d.png"
        """
    )

    parser.add_argument('frames_dir', type=str,
                        help='Directory containing the PNG frame files')
    parser.add_argument('--fps', type=float, default=30,
                        help='Frames per second for the output movie (default: 30)')
    parser.add_argument('--output', '-o', type=str, default='movie.mp4',
                        help='Output movie filename (default: movie.mp4)')
    parser.add_argument('--pattern', type=str, default='frame_%05d.png',
                        help='Frame filename pattern (default: frame_%%(05d.png)')
    parser.add_argument('--codec', type=str, default='libx264',
                        choices=['libx264', 'libx265', 'vp9'],
                        help='Video codec (default: libx264)')
    parser.add_argument('--quality', type=int, default=23,
                        help='CRF quality: lower = better (default: 23). '
                             'Typical range: 18 (high) – 28 (small file)')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster',
                                 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        help='Encoding speed/compression trade-off (default: medium)')
    parser.add_argument('--scale', type=str, default=None,
                        help='Rescale output, e.g. "1920:1080" or "1280:-2" '
                             '(default: keep original size)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the ffmpeg command without executing it')

    args = parser.parse_args()

    # --- Sanity checks ---
    if not os.path.isdir(args.frames_dir):
        print(f"Error: frames directory not found: {args.frames_dir}")
        sys.exit(1)

    frames = find_frames(args.frames_dir, args.pattern)
    if not frames:
        print(f"Error: no frames matching '{args.pattern}' found in {args.frames_dir}")
        print("Tip: use --pattern to specify the correct filename format")
        sys.exit(1)

    print(f"Found {len(frames)} frames in '{args.frames_dir}'")
    print(f"  First : {os.path.basename(frames[0])}")
    print(f"  Last  : {os.path.basename(frames[-1])}")
    duration = len(frames) / args.fps
    print(f"  FPS   : {args.fps}  →  movie duration ≈ {duration:.1f} s")

    if not check_ffmpeg():
        sys.exit(1)

    # --- Build command ---
    input_pattern = os.path.join(args.frames_dir, args.pattern)
    cmd = build_ffmpeg_command(args, input_pattern)

    print(f"\nffmpeg command:\n  {' '.join(cmd)}\n")

    if args.dry_run:
        print("Dry run — not executing.")
        sys.exit(0)

    # --- Run ffmpeg ---
    result = subprocess.run(cmd)

    if result.returncode == 0:
        size_mb = os.path.getsize(args.output) / 1e6
        print(f"\nMovie saved to '{args.output}'  ({size_mb:.1f} MB)")
    else:
        print(f"\nffmpeg exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()