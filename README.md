# Statistical analysis of ACA acquisition star data

## Flight acquisition probability model

Latest fit of FLIGHT model is: `fit_acq_prob_model-2018-04-poly-spline-tccd.ipynb`.

This has acquisition probability model coefficients corresponding to flight tools.
This is a 15-parameter model which uses a joint spline-polynominal parametrization
of the probability surface in magnitude / T_ccd space.

## Other fit `ipynb` notebooks

- `fit_acq_prob_model-2018-04-binned-poly-tccd`: supporting work for flight model
- `fit_acq_prob_model-2018-04-*-warmpix`: poly model fits using warm_pixels (failed, not as good as t_ccd)
- `fit_acq_prob_model-2018-03-sota`: attempt at updating the SOTA model (failed, poor fits for late 2017 / early 2018 data)
- `fit_acq_prob_model-2017-07-sota`: fit of SOTA model (was flight model until April 2018)
- `fit_sota_model*`: earlier iterations of fitting SOTA model
