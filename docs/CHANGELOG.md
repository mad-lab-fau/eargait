# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide and 
Scientific Changes section), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For more information see the 
[Gitlab Releases Page](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/releases) of this 
project.

## [1.4.0] - 

### Added

- New module for experimental features. Things implemented there might be changed or deleted without deprecation
  ([!122](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/122))
- New dataset class that will serve as the base class for all future datasets in `gaitmap.future`. 
  ([!122](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/122),
  [!132](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/132))
- New example on how to create custom datasets with the new dataset-class
  ([!132](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/132))
- The new gaitmap logo is now shown everywhere :)
  ([!129](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/129))
- New parameter `zero_division` added to all methods in `gaitmap.evaluation_utils.scores` allowing for setting the
  return value in case of a zero_division happening. 
  ([!134](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/134))
- A new interpolation method `gaitmap.utils.array_handling.multi_array_interpolation` to interpolate multiple 2D arrays 
  at once to the same length.
  ([!138](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/138))
- Baseclasses to build custom pipelines are added
  ([!128](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/128))
- A GridSearch optimizer (similar to GridSearchCV in sklearn, but without the CV part) with a respective example is
  added. ([!128](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/128))
- A generic wrapper for algorithms and pipelines that can be optimized/trained with respective example.
  ([!128](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/128))
- Added a warning about providing multiple methods when using any trajectory wrapper class to prevent user error.
  ([!142](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/142))
- Added a new `safe_run` method to pipelines that should be used over the normal `run` method.
  `safe_run` performance multiple checks to catch potential implementation errors in user created pipelines.
  ([!144](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/144))
- New parameter `enforce_consistency` added to `gaitmap.event_detection.RamppEventDetection` to make event detection 
  postprocessing optional.
  ([!136](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/136))
- A GridSearchCV optimizer (similar to GridSearchCV in sklearn) with respective example is added.
  This version is specifically implemented for gaitmap and has some fancy ways of performance optimization.
  ([!139](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/139))
- A cross-validation function is added to evaluate optimizable pipelines of any kind.
  ([!139](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/139))

### Changed

- Changed the output format of `gaitmap.evaluation_utils.scores.precision_recall_f1_score` so it will
  be easier to use with the pipelines and optimizer.
  ([!133](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/133))
- Made parameters for `gaitmap.evaluation_utils` keyword only to avoid input confusion.
  ([!135](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/135))
- Improved the performance of dtw template interpolation by a factor of 15-30 (depending on the interpolation type).
  ([!138](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/138))
- File based dtw templates do not implicitly cache the template data anymore.
  This means calling `get_data` on the template does not change the object anymore, which is desirable in the context of
  caching.
  ([!143](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/143))

### Deprecated

### Removed

- `gaitmap.utils.array_handling.interpolate1d` was removed as it is not needed anymore internally.
  ([!138](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/138))

### Fixed

### Migration Guide

### Scientific Changes



## [1.3.0] - 2021-01-25

### Added

- New evaluation function to compare the output of temporal or spatial parameter calculation with ground truth.
  ([!105](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/105))
- New function for merging intervals that are overlapping and that are within a specific distance from each other in
  `utils.array_handling`. ([!120](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/120))
- Some algorithms now support caching intermediate results with `joblib`.
  There is also a new example explaining caching in more detail.
  ([!126](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/126) (mostly reverted),
  [!127](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/127))
- Experimental support for ZUPT based orientation updates in the `RtsKalman` filter.
  This approach is not fully validated and should be used with care.
  If the feature is turned on, a runtime warning will indicate that as well.
  ([!113](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/113))
- `RtsKalman` has a new results `zupts_` that can be used to check which zero-velocity region were used by the filer.
  ([!113](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/113))
- A new guide on how to evaluate (gait)parameters using parameter optimization and cross validation.
  ([!124](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/124))

### Changed

- `evaluation_utils.stride_segmentation._match_label_lists` now works for n-D and should be faster for large stride 
  length. ([!110](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/110))
- Internal refactoring of how the algorithms combine the results of the single sensors to the final output.
  ([!116](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/116))
- Starting from this release the name "sensor data" instead of "dataset" is used to refer to data from multiple sensors
  within one recording.
  The term "dataset" will be used in the future to describe entire sets of multiple recordings potentially including
  multiple participants and further information besides raw IMU data.
  ([!117](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/117))
- Internal refactoring that should improve typing. `mypy` now only returns a couple of issues that are not easily
  fixable or actual `mypy` bugs.
  They will be resolved step by step in the future.
  mypy checking can be done using the new command `dodo type_check` and developer should try to not introduce further
  typing issues.
  ([!118](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/118))
  The refactoring also required some small user facing changes:
  - The `regions_of_interest` parameter of `RoiStrideSegmentation.segment` is now keyword only to be compatible with the
    `StrideSegmentation` base class.
- The `UllrichGaitSequenceDetection` now returns a dict also in the case of merged gait sequences. 
  The gait sequences stored for all sensor are set equal. 
  ([!121](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/121))
- `gait_detection.ullrich_gait_sequence_detection` now uses `utils.array_handling.merge_intervals` and should therefore
  be able to merge sequences faster. 
  ([!120](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/120))
- The `regions_of_interest` parameter of `RoiStrideSegmentation.segment` is now keyword only to be compatible with the
  `StrideSegmentation` base class.
- `RtsKalman.covariance` now has the shape `(len(data), 9 * 9)` instead of `(len(data) * 9, 9)` which should make it 
  easier to plot.
  

### Deprecated

### Removed
    
- `gait_detection.ullrich_gait_sequence_detection._gait_sequences_to_boolean_single` has been made redundant by 
  `utils.array_handling.merge_intervals` and was therefore removed. 
  ([!120](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/120))
- Removed old serialization format for DTW templates that was deprecated a couple of releases ago.
  ([#127](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/issues/127))  

### Fixed

- In `stride_segmentation.roi_stride_segmentation` the assignment of the offset corrected strides to the
 `combined_stride_list` was fixed.
  ([!114](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/114))
- Added a note to BarthDTW to clarify the calculation of the distance matrix.
- Changed the Chebyshev distance function in `evaluation_utils.stride_segmentation._match_label_lists` to a general one 
  that can use n-D input. ([!115](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/115))
- Fixed bug in `utils.static_moment_detection.find_static_samples` where extremely short sequences, which would only fit
  a single window, would fail.
  ([!109](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/109))

### Migration Guide

- Due to the renaming of dataset to sensordata, a couple of names and import path are changed:
  - All imports from `gaitmap.utils.dataset_helper` should be changed to `gaitmap.utils.datatype_helper`
  - `SingleSensorDataset`, `MultiSensorDataset` and `Dataset` are now called, `SingleSensorData`, `MultiSensorData` and
    `SensorData`, respectively.
  - The functions `is_single_sensor_dataset`, `is_multi_sensor_dataset`, `is_dataset`, and 
    `get_multi_sensor_dataset_names` are renamed to `is_single_sensor_data`, `is_multi_sensor_data`, `is_sensor_data`, 
    and `get_multi_sensor_names`.

### Scientific Changes


## [1.2.0] - 2020-11-11

### Added

- An Error-State-Kalman-Filter with rts smoothing (`RtsKalman`) was added that can be used to estimate orientation and
  position over longer regions by detecting its own ZUPTs.
  This can (should) be used in combination with `RegionLevelTrajectory`.
- A `RegionLevelTrajectory` class that calculates trajectories for entire ROIs/gait sequences.
  Otherwise the interface is very similar to `StrideLevelTrajectory`.
  ([!87](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/87),
  [!98](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/98),
  [!85](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/85))
- Added a new example for `RegionLevelTrajectory`.
  ([!87](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/87)) 
- Support for trajectory methods that can calculate orientation and position in one go is added for 
  `StrideLevelTrajectory` ([!87](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/87))
- Implemented local warping constraints for DTW. This should help in cases were only parts of a sequence are matched
  instead of the entire sequence.
  These constraints are available for `BaseDTW` and `BarthDTW`, but are **off** by default.
  A new class `ConstrainedBarthDTW` was added, that has the constraints **on** by default.
- Added support for `MultiSensorStrideLists` for all evaluation functions in `evaluation_utils.stride_segmentation`.
  ([!91](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/91))
- Global validation errors for region of interest list
  ([!88](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/88))
- Global validation errors for orientation/position/velocity lists
  ([!88](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/88), 
  [!95](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/95))
- New evaluation functions to compare the results of event detection methods with ground truth
  ([!92](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/92))
- Added the functionality to merge gait sequences from individual sensors in `UllrichGaitSequenceDetection` 
  ([!93](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/93))

### Changed

- The initial orientation for the Stride and Region Level trajectory now uses the correct number of samples as input.
  This might change the output of the integration method slightly!
  However, it also allows to pass 0 for `align_window_length` and hence, just use the first sample of the integration
  region to estimate the initial orientation.
  (mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/98)
- It is now ensured that in **all** in gaitmap where a start and an end sample is provided, the end sample is
  **exclusive**.
  Meaning if the respective region should be extracted from a dataarray, you can simply do `data[start:end]`.
  All functions that used different assumptions are changed.
  ([!97](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/97)~,
  [!103](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/103))
- The default check for orientation/position/velocity lists now only expects a "sample" column and not a "sample" and a
  "s_id" column by default.
  The `s_id` parameter was removed and replaced with a more generic `{position, velocity, orientation}_list_type`
  parameter than can be used to check for either stride or region based lists.
  See the migration guide for more information.
  ([!88](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/88))
- Internal restructuring of the evaluation utils module.
  In case you used direct imports of the functions from the submodule, you need to update these.
  ([!92](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/92))

### Deprecated

### Removed

### Fixed

- Fixed possible `ZeroDivisionError` when using `evaluation_utils.scores`.
  ([!104](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/104))

### Migration Guide

- In case your algorithm uses `is_{single,multi}_sensor_{position, velocity, orientation}_list` all usages must be 
  updated to include `{position, velocity, orientation}_list_type="stride"` to restore the default behaviour before the
  update.
  If you were using the function with `s_id=False`, you can update to `{position, velocity, orientation}_list_type=None`
  to get the same behaviour.

### Scientific Changes

- All Dtw based methods now produce outputs that follow our index guidelines.
  This means that they correctly produce matches, which end index is **exclusive**.
  In some cases this might change existing results by adding 1 to the end index.
  ([!103](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/103))
- The initial orientation in the StrideLevelTrajectory now has a window with exactly the number of samples specified
  (instead of one less due to Python indexing).
  This will lead to very small changes in the calculated trajectory.
  (mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/98)


## [1.1.0] - 2020-10-04

### Added

- Proper support for gait sequences and regions of interest in the stride segmentation
  ([!64](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/64), 
  [!71](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/71),
  [!75](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/75)).
  Note that support was first implemented in the segmentation algorithms directly and then reworked and moved into the
  `RoiStrideSegmentation` wrapper class.
- "Max. lateral excursion" and "max. sensor lift" are now implemented as two new spatial parameters.
  ([!79](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/79))
- The `align_heading_of_sensors` method has now an option to run an additional smoothing filter that can avoid
  misalignments in certain cases.
- `VelocityList` is now a separate dtype representing integrated velocity values
  ([!77](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/77))
- `UllrichGaitSequenceDetection` now has its own example in the Sphinx Gallery 
  ([!83](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83))
- Added a new module `gaitmap.utils.signal_processing` for general filtering and processing purposes (mainly for
  moving some functions from `UllrichGaitSequenceDetection` 
  ([!83](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83))


#### The new `evaluation_utils` module [#117](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/issues/117)

A new module was started that will contain helper functions to evaluate the results of gait analysis pipelines.
This is the first version and we already added:

- A set of function to compare stride lists based on start and end values 
  ([!66](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/66))
- A set of common metrics (recall, precision, f1) that can be calculated for segmentation results
  ([!66](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/66))
  
#### Global Validation Error Handling

[#72](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/issues/72),
[#16](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/issues/16)

To make developing new algorithms easier and error messages easier to understand for users,
we modified the data-object validation helper to raise proper descriptive error messages.
The data-object helpers now have an optional argument (`raise_exception=True`) to trigger the new error-messages.

- Proper validation for the dataset objects.
  ([!80](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/80))
- A new `is_dataset` method to validate either single or multi-sensor datasets.
  This should be used in functions that should work with both inputs.
  ([!80](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/80))
- Proper validation for the stride list objects.
  ([!86](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/86))
- A new `is_stride_list` method to validate either single or multi-sensor stride lists.
  This should be used in functions that should work with both inputs.
  ([!86](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/86))
- A custom error class `ValidationError` used for all validation related errors.
  ([!80](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/80))

All existing algorithms are updated to use these new validation methods.
([!80](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/80),
[!84](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/84),
[!86](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/86))

The remaining datatypes will get updated validation functions in a future release.

### Changed

- To get or load the data of a DtwTemplate you now need to call `get_data` on the method instead of just using the
  `data` property ([!73](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/73))
  This is done to conform with the basic class structure needed for proper serialization.
- Changed function name `_butter_lowpass_filter` in `gaitmap.gait_detection.UllrichGaitSequenceDetection` to 
`butter_lowpass_filter_1d` ([!83](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83))
- Moved `butter_lowpass_filter_1d` and `row_wise_autocorrelation` from `gaitmap.gait_detection
.UllrichGaitSequenceDetection` to the new module `gaitmap.utils.signal_processing`
([!83](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83))

### Deprecated

- The format in which DataFrame Attributes are stored in json has been changed.
  The old format can still be loaded, but this will be removed in future versions.
  Related to [!72](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/72).
  See the migration guide for actions to take. 

### Removed

### Fixed

- Fixed a bug in the madgwick algorithms that might have caused some incorrect results in earlier version
  ([!70](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/70))
- Fixed issue that templates that were stored in json do not preserve order when loaded again (see more info in migration guide)
  ([!72](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/72))
- Fixed an issue that `rotate_dataset_series` performed an unexpected inplace modification of the data.
  ([!78](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/78))
- Fixed a bug that would break the `UllrichGaitSequenceDetection` in case of no active signal windows
  ([95736](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/commit/95736e8d2676f98d4c43ea0bfa3dbf3566542f71))
- Fixed a bug that would break the `UllrichGaitSequenceDetection` in case the input signal was just as long as the
  window size.
  ([d516a](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/commit/d516a40520f86bfd39ddcdd813b5c18312785085))
- Adapted scaling factors for the usage of accelerometer data in the `UllrichGaitSequenceDetection` to work with
  values given in m/s^2 (in contrast to g as done in the publication / the usage with mGL-algorithms-py)
  ([!83](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83))
- Fixed install link in project README

### Migration Guide

- The format in which DataFrame Attributes are stored in json has been changed and the old format will be fully removed
  in the future.
  If you have DtwTemplates stored as json, load them (you will get a warning) once. Double check that the ordering of 
  the template data is correct. If it is not, sort it correctly and then save the object again.
  This should fix all issues going forward.
- To get or load the data of a DtwTemplate, you now need to use `template.get_data()` instead of `template.data`.
  Simply replacing the old with the new option should fix all issues.
- For custom algorithms using the gaitmap format, you should migrate to the new datatype validation functions!
  See "Added" section for more details.

## [1.0.1] - 2020-07-15

A small bug fix release that fixes some issues with the new template creation function.

### Fixed

- The stride interpolation template creation function now works correctly if the index of the individual strides is 
  different ([!68](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/68)).
- Templates that use datafraes to store the template data can now be serialized
  ([!69](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/69))

## [1.0.0] - 2020-06-26

This is the first official release of gaitmap (Wuhu!).
Therefore, we do not have a proper changelog for this release.

### Added
- All the things :)

### Changed

### Deprecated

### Removed

### Fixed

### Migration Guide
