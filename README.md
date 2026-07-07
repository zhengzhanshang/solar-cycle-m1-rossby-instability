# Solar Cycle Dependence of the Unstable m=1 Rossby Mode

Dedalus simulation code, postprocessing scripts, and videos supporting a study of the barotropic instability of the solar high-latitude m=1 Rossby mode and its dependence on the solar cycle phase.

## Abstract

Observations show that the solar high-latitude m=1 unstable mode depends on cycle phase, appearing near minima and disappearing near maxima. This work studies the m=1 Rossby mode under latitudinal differential rotation Ω(θ) = A + B cos²θ + C cos⁴θ, using 2D (θ, φ) linear spherical hydrodynamic equations in the rotating frame of the solar equator. The mode is found to be generally unstable to barotropic instability, with growth rate governed by C and period governed by B. The unstable mode has order ℓ=3 (u_θ) / ℓ=4 (u_φ), consistent with observations. Since C anticorrelates with cycle phase (larger near minima), the model explains the mode's cyclic appearance.

*Paper currently unpublished — citation/link will be added upon publication.*

## Repository structure

```
.
├── simulation/
│   └── spherical-2d-hydro.py      # 2D spherical hydrodynamic simulation (linear, barotropic instability)
├── postprocessing/
│   ├── plot_spherical_hovmoller.py    # Hovmöller diagrams
│   ├── plot_spherical_surface.py      # 3D globe / Mollweide projections
│   └── make_movie.py                  # ffmpeg wrapper for animations
├── input.json                     # Simulation parameters (A, B, C, ℓ, m, resolution, etc.)
├── videos/                        # Rendered animations of the unstable mode
└── README.md
```

## Requirements

- Python 3.10+
- [Dedalus v3](https://dedalus-project.org/) (install via conda-forge)
- numpy, scipy, matplotlib, h5py
- ffmpeg (for `make_movie.py`)

```bash
conda create -n rossby-m1 -c conda-forge python=3.10 dedalus
conda activate rossby-m1
pip install numpy scipy matplotlib h5py
```

## Usage

```bash
# Run the simulation (parameters set in input.json)
python simulation/spherical-2d-hydro.py input.json

# Generate a Hovmöller diagram
python postprocessing/plot_spherical_hovmoller.py --input <output_file>

# Generate a surface/Mollweide animation frame set, then compile to video
python postprocessing/plot_spherical_surface.py --input <output_file>
python postprocessing/make_movie.py --frames <frame_dir> --out videos/output.mp4
```

## License

MIT

## Citation

Citation details will be added once the paper is published.
