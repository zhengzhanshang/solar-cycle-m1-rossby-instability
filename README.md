# Solar Cycle Dependence of the Unstable m=1 Rossby Mode

Dedalus simulation code, postprocessing scripts, and videos supporting a study of the barotropic instability of the solar high-latitude m=1 Rossby mode and its dependence on the solar cycle phase.

## Abstract

Observations show that the solar high-latitude m=1 unstable mode depends on cycle phase, appearing near minima and disappearing near maxima. This work studies the m=1 Rossby mode under latitudinal differential rotation Ω(θ) = A + B cos²θ + C cos⁴θ, using 2D (θ, φ) linear spherical hydrodynamic equations in the rotating frame of the solar equator. The mode is found to be generally unstable to barotropic instability, with growth rate governed by C and period governed by B. The unstable mode has order ℓ=3 (u_θ) / ℓ=4 (u_φ), consistent with observations. Since C anticorrelates with cycle phase (larger near minima), the model explains the mode's cyclic appearance.

*Paper currently unpublished — citation/link will be added upon publication.*

## Contents

- Simulation script(s) — 2D spherical hydrodynamic linear model of the m=1 Rossby mode
- Postprocessing scripts — surface/Mollweide plots, video generation
- `input.json` — simulation parameters (A, B, C, ℓ, m, resolution, etc.)
- Video files — animations of the unstable mode

## Requirements

- Python 3.10+, [Dedalus v3](https://dedalus-project.org/), numpy, scipy, matplotlib, h5py, ffmpeg

## License

MIT

## Citation

Citation details will be added once the paper is published.
