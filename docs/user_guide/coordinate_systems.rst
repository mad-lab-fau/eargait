.. _coordinate_systems:

===================

Coordinate  Systems
===================
Defining coordinate system is a crucial aspect when working with IMU data. The coordinate system definition of an internal gaitmap library [2]_. 


Hearing Aid Frame (HAF)
    * forms a right-handed coordinate system with axes called **X, Y, Z** 
    * uses right-hand-rule around each axis for definition of positive direction of the Gyroscope 
    * axis as defined by the sensor itself, depends on how sensor is located in hearing aid housing
    * can be different for different hearing aid firmware versions

Ear Sensor Frame (ESF)
    * forms a right-handed coordinate system with axes called **X, Y, Z**
    * uses right-hand-rule around each axis for definition of positive direction of the Gyroscope 
    * defines axes' directions as up (Z), to the tip of the shoe (X), and
      to the **left** (Y)

Foot Body Frame (FBF)
    * consists of the 3 axes *ML* (medial to lateral), *PA* (posterior to anterior), and *SI* (superior to inferior)
    * is **not** right handed and should not be used for any physical calculations
    * produces the same sensor signal independent of the ear (right/left) for the same anatomical movement (e.g.
      lateral acceleration = positive acceleration)
    * follows convention of directions from [1]_

.. _ff:

Ear Frame Overview
-------------------

.. figure:: /images/eargait_sensor_body_frame
    :alt: eargait foot frame
    :figclass: align-center

    The positive directions for the accelerometer (straight arrows) and the gyroscope (circular arrows) for the hearing aid frame (top), ear
    sensor frame (bottom left) and ear body frame (bottom right).
    `Click here for fullscreen version of image <../_images/eargait_sensor_body_frame.svg>`_

.. table:: Table showing the expected signal (positive or negative and in which axis) when a certain movement
           (displacement or rotation) of a ear happens for the sensor (ESF) and the body frame (EBF).

  +-------------------+------------------------+------------------------+
  |                   |          ESF           |          EBF           |
  +-------------------+-----------+------------+-----------+------------+
  |                   | Left Ear  | Right Ear  | Left Ear  | Right Ear  |
  +===================+===========+============+===========+============+
  |                              **Displacements**                      |
  +-------------------+-----------+------------+-----------+------------+
  | anterior          | +acc_x    | +acc_x     | +acc_pa   | +acc_pa    |
  +-------------------+-----------+------------+-----------+------------+
  | posterior         | -acc_x    | -acc_x     | -acc_pa   | -acc_pa    |
  +-------------------+-----------+------------+-----------+------------+
  | lateral           | +acc_y    | -acc_y     | +acc_ml   | +acc_ml    |
  +-------------------+-----------+------------+-----------+------------+
  | medial            | -acc_y    | +acc_y     | -acc_ml   | -acc_ml    |
  +-------------------+-----------+------------+-----------+------------+
  | inferior          | -acc_z    | -acc_z     | +acc_si   | +acc_si    |
  +-------------------+-----------+------------+-----------+------------+
  | superior          | +acc_z    | +acc_z     | -acc_si   | -acc_si    |
  +-------------------+-----------+------------+-----------+------------+



Algorithmic Implementation
==========================
The algorithmic implementation is also based on the internal gaitmap library [2]_. 

.. _alignment-algorithms:

Alignment with the Ear Sensor Frame
-----------------------------------

Aligning the coordinate system of a sensor with the eargait coordinate system can be a complicated, but usually requires prior knowledge about sensor orientation in the housing.  

For Signa devices (D11, or D12) the function :func:`~eargait.utils.preprocessing.rotations.convert_ear_to_esf` can be used to transform hearing aid sensor data into the ear sensor frame. 

.. note:: For D12 a different rotation is required. For all other firmware versions a default ration will be applied. 


Transformation into the Foot Body Frame
---------------------------------------

Once the data is properly aligned to the earmap-ESF, it is very easy to transform it into the respective BF.
For this you can use the function :func:`~eargait.utils.preprocessing.rotations.convert_ear_to_ebf`.


Transformation into the Foot Body Frame and Alignment to Gravity
----------------------------------------------------------------
Some times an alignment with gravity is necessary. 

The function :func:`~eargait.utils.preprocessing.rotations.aling_gravity_and_convert_ear_to_ebf` transforms data into eargait-ESF, then alignes data with gravity and then transforms it into body frame.


.. note:: Can only be applied for if data is in eargait-HAF.


Reference
---------

.. [1] Wu, G., Siegler, S., Allard, P., Kirtley, C., Leardini, A., Rosenbaum, D., … Stokes, I. (2002). ISB
       recommendation on definitions of joint coordinate system of various joints for the reporting of human joint
       motion - Part I: Ankle, hip, and spine. Journal of Biomechanics. https://doi.org/10.1016/S0021-9290(01)00222-6

.. [2] Küderle A., et al. Gaitmap. To be published soon. 
