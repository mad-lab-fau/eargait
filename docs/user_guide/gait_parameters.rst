.. _gait_parameters:

===============================================
Gait Parameters
===============================================

Spatio-temporal gait parameters on Step and Walking-Bout (i.e. average) level

.. list-table::
   :widths: 85 25 40 90
   :header-rows: 1


   * - Parameter
     - Naming
     - Unit
     - Details
   * - Stride time
     - stride_time
     - s
     -
   * - Stance time
     - stance_time
     - s
     -
   * - Swing time
     - swing_time
     - s
     -
   * - Step length
     - step_length
     - m
     -
   * - Stride length
     - stride_length
     - m
     -
   * - Gait velocity
     - gait_velocity
     - m/s
     - Ratio :math:`\frac{step length}{step time}`
|
Gait parameters on Walking-Bout (i.e. average) level

.. list-table::
   :widths: 85 25 40 90
   :header-rows: 1

   * - Parameter
     - Naming
     - Unit
     - Details
   * - Number of steps
     - number_of_steps
     - total number
     -
   * - Cadence
     - cadence
     - steps/min
     - Candence as steps per minute: :math:`\frac{60s}{t_{last_IC}-t_{first_IC}}* number\ of\ steps`
   * - Cadence dominant frequency
     - cadence_dom_freq
     - steps/min
     - Cadence based on dominant frequency.
   * - Standard deviation
     - *_std
     - Respective unit
     - Standard deviation over a walking sequence.
   * - Asymmetry
     - *_asymmetry
     - Respective unit
     - Absolute difference between average ipsi- and contralateral parameter
   * - Asymmetry Percent
     - *_asymmetry_percent
     - %
     - Ratio of absolute difference between average ipsi- and contralateral parameter and mean.
   * - Symmetry index
     - *_si
     - %
     - Symmetry index for a parameter: :math:`2 * \frac{X_{ipsi}-X_{contra}}{X_{ipsi}+X_{contra}}*100\%`
   * - Coefficient of variation
     - *_cv
     -
     - Ration of standard deviation :math:`\sigma` to mean :math:`\mu`: :math:`\frac{\sigma}{\mu}`