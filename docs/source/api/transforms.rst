Transforms API
==============

Concise reference for all built-in transforms. For workflows, diagrams, and tuning guidance
see the :doc:`/transforms/index` handbook and :doc:`/user_guide/transforms`.

FFTAnalysis
~~~~~~~~~~~

:class:`rheojax.transforms.fft_analysis.FFTAnalysis` | Handbook: :doc:`/transforms/fft`
Convert time-domain data to frequency-domain spectra with optional detrending, windowing,
and power spectral density output.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 28 72

   * - Parameter (default)
     - Description
   * - ``window`` (``'hann'``)
     - Window function before FFT. Choices: ``'hann'``, ``'hamming'``, ``'blackman'``, ``'bartlett'``, ``'none'``.
   * - ``detrend`` (``True``)
     - Remove linear trend before transforming.
   * - ``return_psd`` (``False``)
     - Return power spectral density instead of magnitude.
   * - ``normalize`` (``True``)
     - Normalize FFT amplitude by sample count.

.. autoclass:: rheojax.transforms.fft_analysis.FFTAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

Mastercurve
~~~~~~~~~~~

:class:`rheojax.transforms.mastercurve.Mastercurve` | Handbook: :doc:`/transforms/mastercurve`
Time-temperature superposition with WLF, Arrhenius, or manual shift factors plus optional
vertical shifts and shift optimization.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 28 72

   * - Parameter (default)
     - Description
   * - ``reference_temp`` (``298.15`` K)
     - Target temperature in Kelvin for the mastercurve.
   * - ``method`` (``'wlf'``)
     - Shift factor model: ``'wlf'``, ``'arrhenius'``, or ``'manual'``.
   * - ``C1`` (``17.44``)
     - WLF constant :math:`C_1`.
   * - ``C2`` (``51.6``)
     - WLF constant :math:`C_2` in Kelvin.
   * - ``E_a`` (``None``)
     - Activation energy in J/mol for Arrhenius shifts (required when ``method='arrhenius'``).
   * - ``vertical_shift`` (``False``)
     - Apply vertical (modulus) shifting in addition to horizontal shifts.
   * - ``optimize_shifts`` (``True``)
     - Nonlinear least-squares refinement of supplied shift factors.

.. autoclass:: rheojax.transforms.mastercurve.Mastercurve
   :members:
   :undoc-members:
   :show-inheritance:

MutationNumber
~~~~~~~~~~~~~~

:class:`rheojax.transforms.mutation_number.MutationNumber` | Handbook: :doc:`/transforms/mutation_number`
Computes the mutation number :math:`\Delta` from relaxation data to quantify viscoelastic
character between perfectly elastic and perfectly viscous limits.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 28 72

   * - Parameter (default)
     - Description
   * - ``integration_method`` (``'trapz'``)
     - Numerical integration strategy: ``'trapz'``, ``'simpson'``, or ``'cumulative'``.
   * - ``extrapolate`` (``False``)
     - Estimate tail contributions beyond the measurement window.
   * - ``extrapolation_model`` (``'exponential'``)
     - Tail model when ``extrapolate=True``. Options: ``'exponential'`` or ``'powerlaw'``.

.. autoclass:: rheojax.transforms.mutation_number.MutationNumber
   :members:
   :undoc-members:
   :show-inheritance:

OWChirp
~~~~~~~

:class:`rheojax.transforms.owchirp.OWChirp` | Handbook: :doc:`/transforms/owchirp`
Optimal waveform chirp analysis for LAOS experiments, generating time-frequency maps,
harmonic spectra, and nonlinear indicators.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 28 72

   * - Parameter (default)
     - Description
   * - ``n_frequencies`` (``100``)
     - Number of frequency bins used in the time-frequency analysis.
   * - ``frequency_range`` (``(1e-2, 1e2)`` Hz)
     - Minimum and maximum frequencies analyzed.
   * - ``wavelet_width`` (``5.0``)
     - Width parameter controlling wavelet localization (higher = smoother).
   * - ``extract_harmonics`` (``True``)
     - Whether to compute discrete harmonic amplitudes (G1, G3, ...).
   * - ``max_harmonic`` (``7``)
     - Highest harmonic order reported when ``extract_harmonics=True``.

.. autoclass:: rheojax.transforms.owchirp.OWChirp
   :members:
   :undoc-members:
   :show-inheritance:

SmoothDerivative
~~~~~~~~~~~~~~~~

:class:`rheojax.transforms.smooth_derivative.SmoothDerivative` | Handbook: :doc:`/transforms/smooth_derivative`
Noise-robust numerical differentiation with Savitzky-Golay, finite-difference, spline,
or total-variation methods plus optional pre/post smoothing.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 28 72

   * - Parameter (default)
     - Description
   * - ``method`` (``'savgol'``)
     - Differentiation algorithm: ``'savgol'``, ``'finite_diff'``, ``'spline'``, ``'total_variation'``.
   * - ``window_length`` (``11``)
     - Odd window size for Savitzky-Golay or smoothing kernels.
   * - ``polyorder`` (``3``)
     - Polynomial order for Savitzky-Golay (must be < ``window_length``).
   * - ``deriv`` (``1``)
     - Derivative order to compute (>=1).
   * - ``smooth_before`` (``False``)
     - Apply moving-average smoothing prior to differentiation.
   * - ``smooth_after`` (``False``)
     - Apply smoothing to the derivative result.
   * - ``smooth_window`` (``5``)
     - Window size for the optional smoothing passes.

.. autoclass:: rheojax.transforms.smooth_derivative.SmoothDerivative
   :members:
   :undoc-members:
   :show-inheritance:
