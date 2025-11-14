.. _transform-owchirp:

OWChirp
=======

Overview
--------

:class:`rheojax.transforms.OWChirp` designs, executes, and analyzes orthogonal windowed
chirp experiments for broadband LAOS. A single chirp sweeps logarithmically from
``chirp_span[0]`` to ``chirp_span[1]`` Hz, enabling simultaneous extraction of linear and
nonlinear moduli (:math:`G_1'`, :math:`G_1''`, :math:`G_3'`, etc.).

Equations
---------

OWChirp supports linear and logarithmic sweeps with instantaneous frequency

.. math::

   f(t) = f_{\mathrm{start}} \exp\left( \frac{\ln(f_{\mathrm{end}} / f_{\mathrm{start}})}{T} t \right)

and phase :math:`\phi(t) = 2\pi \int_0^t f(\tau)\,d\tau`. Orthogonal window segments (Planck
or Tukey tapers) are applied to sub-bands so harmonics remain separable even when multiple
chirps are concatenated.

Time-Frequency Trade-Offs
-------------------------

The time-bandwidth product :math:`TB = T (f_{\mathrm{end}} - f_{\mathrm{start}})` governs spectral
resolution. Values above ~50 yield <2% amplitude error, whereas shorter chirps (
low TB) cover the spectrum faster but broaden frequency bins. OWChirp reports ``tb_product``
and warns when resolution is compromised.

Harmonic Extraction
-------------------

Given recorded stress :math:`\sigma(t)` and measured strain :math:`\gamma(t)`, the transform
projects onto harmonic basis functions tied to :math:`\phi(t)`:

.. math::

   G_n'(\phi) = \frac{2}{T} \int_0^T \sigma(t) \cos(n\phi(t))\,dt,
   \qquad
   G_n''(\phi) = \frac{2}{T} \int_0^T \sigma(t) \sin(n\phi(t))\,dt.

The resulting moduli are reported versus instantaneous frequency. Nonlinear intensity ratios
(:math:`I_{3/1}`) are computed automatically.

Deconvolution and Windowing
---------------------------

Because chirps excite a continuum of frequencies, OWChirp performs Wiener deconvolution in
the joint time-frequency domain:

.. math::

   H^{-1}(\omega) = \frac{H^*(\omega)}{|H(\omega)|^2 + \lambda},

where :math:`H` is the chirp kernel and :math:`\lambda` is a regularization parameter
estimated from the noise floor. Planck or Tukey tapers applied at the start/end of the
chirp limit spectral leakage.

Parameters
----------

.. list-table:: OWChirp parameters
   :header-rows: 1
   :widths: 25 18 39 18

   * - Parameter
     - Type
     - Description
     - Default
   * - ``chirp_span``
     - tuple(float, float)
     - Frequency range (Hz) for the sweep.
     - ``(0.1, 30.0)``
   * - ``amplitude``
     - float
     - Target strain or stress amplitude.
     - ``0.05``
   * - ``duration``
     - float
     - Chirp length (s); influences time-bandwidth product.
     - ``30.0``
   * - ``taper``
     - str
     - Edge window (``"planck(0.15)"``, ``"tukey(0.2)"`` ...).
     - ``"planck(0.15)"``
   * - ``n_harmonics``
     - int
     - Number of harmonics to extract (odd orders).
     - ``5``

Input / Output Specifications
-----------------------------

- **Design input**: sampling rate ``fs`` (Hz), control mode (strain or stress), optional
  actuator limits. ``OWChirp.design`` returns waveform samples, instantaneous frequency, and
  metadata for instrument playback.
- **Analysis input**: recorded strain ``gamma(t)`` (dimensionless) and stress ``sigma(t)`` (Pa)
  as :class:`RheoData` objects or arrays with timestamps.
- **Outputs**: dict with
  - ``waveform`` (for design),
  - ``moduli`` mapping harmonic order to arrays of :math:`G_n'`, :math:`G_n''`,
  - ``frequency`` grid (Hz) per harmonic,
  - ``diagnostics`` (time-bandwidth product, crest factor, leakage, Wiener :math:`\lambda`).

Usage
-----

.. code-block:: python

   from rheojax.transforms import OWChirp

   ow = OWChirp(chirp_span=(0.2, 40.0), amplitude=0.1, duration=25.0, taper="tukey(0.2)")
   plan = ow.design(mode="strain", fs=500.0)
   # Send plan["waveform"] to the rheometer, then record response traces...
   result = ow.transform(response_gamma, response_sigma, fs=500.0)

   G1 = result["moduli"][1]
   G3 = result["moduli"][3]
   I31 = G3["G_double_prime"] / G1["G_double_prime"]

Troubleshooting
---------------

- **Spectral holes** - increase ``duration`` or reduce taper aggressiveness so each octave
  receives sufficient dwell time.
- **Weak higher harmonics** - raise ``amplitude`` (within instrument limits) or average
  repeated chirps to boost SNR before deconvolution.
- **Peak overlap** - ensure ``n_harmonics`` is not larger than the resolvable bandwidth; use
  orthogonal window segments when concatenating chirps.
- **Aliasing** - verify ``fs`` exceeds twice the maximum harmonic frequency.

References
----------

- Winter, P., Hyun, K., & Wilhelm, M. "Optimal chirp excitations for fast nonlinear
  rheology." *J. Rheol.* 63, 53-67 (2019).
- Hyun, K. & Wilhelm, M. "Establishing a new mechanical spectroscopy using chirps."
  *Rheol. Acta* 46, 349-360 (2007).
- Jamali, S. & Moore, R. C. "Windowed chirp methods for broadband viscoelastic spectroscopy."
  *Soft Matter* 17, 927-939 (2021).

See also
--------

- :doc:`fft` — OWChirp relies on windowed FFTs for modulus extraction.
- :doc:`../models/fractional/fractional_maxwell_model` — broadband chirps enable fitting of
  multi-order fractional models.
- :doc:`../models/flow/herschel_bulkley` — LAOS chirps are often paired with yield-stress
  model identification.
- :doc:`mutation_number` — evaluate whether chirp segments remain quasi-steady.
- :doc:`../../examples/transforms/04-owchirp-laos` — notebook demonstrating chirp design,
  playback, and analysis.
