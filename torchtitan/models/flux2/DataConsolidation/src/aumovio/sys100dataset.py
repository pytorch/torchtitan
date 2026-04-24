"""
A custom dataset class to read discretised radar measurements together with
the corresponding fisheye video files from the proprietary Conti SYS100 dataset.

"""

import os, sys
from typing import Callable, Dict, Tuple, Union, Any, cast
from pathlib import Path
from fractions import Fraction

import numpy as np
import torch
from torchvision.transforms import v2
import h5py
import cv2
from einops import rearrange
from tqdm.auto import tqdm
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams

try:
    # New API (Matplotlib 3.5+)
    cmap = plt.colormaps.get_cmap('bwr') # type: ignore[attr-defined]
except AttributeError:
    from matplotlib.cm import get_cmap
    cmap = get_cmap('bwr')

basepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if not basepath in sys.path:
    sys.path.insert(0, basepath)
from ..multifiledataset import MultiFileDataset
from ..videodataset import extract_frame_via_cv2, extract_frame_via_pyav
from .fisheye import Undistorter, PinholeCamera, FisheyeCamera



def point_from_gridentry_vectorized(grid, xmin=0, ymin=-100, ymax=100, inorder="H W C", thres=0.1):
    """
    Takes a grid of discretized points and returns a list of only those points.
    
    Assumes that the grid is of shape (discretization_steps, discretization_steps, feature_dim)
    and that the first feature dimension determines whether the point of that
    cell exists (TODO: also by how much the x direction will be scaled by).
    The second and third feature dimensions determine the x and y offset
    from the grid cell centre. All other feature dimensions will be appended as
    features of the extracted points.
    """
    
    grid = rearrange(grid, f"{inorder.lower()} -> h w c")
    discretization_steps_x, discretization_steps_y, feature_dim = grid.shape

    # Get mask of valid entries
    row_idx, col_idx = np.where(grid[..., 2] > thres)
    
    #entries = grid[row_idx, col_idx, :]
    #xmax = entries[:, 2] * 256
    #entries[:,0] = (entries[:, 0] + (row_idx + 0.5)) * (xmax - xmin) / discretization_steps_x + xmin
    #entries[:,1] = (entries[:, 1] + (col_idx + 0.5)) * (ymax - ymin) / discretization_steps_y + ymin
    #return entries

    # transform offsets in the first two channels into actual grid coordinates
    entries = grid[row_idx, col_idx, :]
    xmax = entries[:, 2] * 256
    x = (entries[:, 0] + (row_idx + 0.5)) * (xmax - xmin) / discretization_steps_x + xmin
    y = (entries[:, 1] + (col_idx + 0.5)) * (ymax - ymin) / discretization_steps_y + ymin
    #for i in range(len(row_idx)):
    #    print(f" row {row_idx[i]}, col_idx {col_idx[i]} -> ({x[i]:.2f}, {y[i]:.2f})")

    # Concatenate x, y, and the remaining features
    out = np.empty((entries.shape[0], 2 + entries.shape[1] - 2), dtype=entries.dtype)
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2:] = entries[:, 2:]
    return out

def expected_static_grid(vel, shape=(64,64), xmin=0, xmax=102.4, ymin=-100, ymax=100):
    pass

def gamma_correction(image, gamma=0.6, brightness=1.4):
    # https://confluence.auto.continental.cloud/pages/viewpage.action?pageId=1551636791
    invGamma = 1. / gamma
    image = (image/255.) ** invGamma
    # TODO: check if brightness is needed
    #brightness = np.clip(0.5/np.mean(image), 1.0, 3.0)
    #print(f"brightness {0.5/np.mean(image)} -> {brightness}")
    return np.clip( image * 255. * brightness +0.5, 0, 255).astype(np.uint8)

def approximate_gamma_correction(image, gamma=0.6, brightness=2.5):
    # images needed to be gamma corrected in the raw BAYER domain.
    # Shifting them back from the standard RGB (sRGB) into the linear intensity RGB domain
    # for the gamma correction is the best we can do
    # https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22)
    img = image.astype(np.float32) / 255.0
    img_linear = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    
    # Apply gamma (in linear domain)
    img_gamma = np.clip((img_linear ** (1.0 / gamma)) * brightness, 0, 1.0)
    
    # Back to sRGB
    img_srgb = np.where(img_gamma <= 0.0031308, img_gamma * 12.92, 1.055 * (img_gamma ** (1 / 2.4)) - 0.055)
    return (np.clip(img_srgb, 0, 1.0) * 255 +0.5).astype(np.uint8)

def approximate_soft_gamma_correction(image, gamma=0.6, brightness=1., perc=95):
    # The problem with the default gamma correction is that in night images,
    # where all the details are contained in the lower region of the [0...255]
    # pixel domain, the gamma correction will shift all values to very similar
    # darker values and effectively remove all details.
    # This approach softens the gamma compression in dark images by reducing gamma
    # by the fraction img.perc[p]/p where img.perc[p] is the p-th percentile of
    # the image pixel values, as we would assume this to be close
    # to one in bright images but much lower in dark images.
    img = image.astype(np.float32) / 255.0
    img_perc = np.percentile(np.max(img, axis=2).flatten(), perc)
    scaling = np.clip(img_perc / perc * 100, 0.1, 1.0)
    img_linear = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    #print(f"\npercentile {perc}% of the raw sRGB [0...1] image: {img_perc:.3f} skaling factor: {scaling:.3f}", end=" ")
    
    inv_gamma = 1 + (1./gamma - 1) * scaling
    #scaled_brightness = 1 + (brightness - 1) * scaling
    scaled_brightness = (1/0.5)**(inv_gamma)  * brightness
    # makes sure that (if brightness==1.) 0.5 will always be mapped to 1, so this only influences the slope
    # note that this factor 0.5 correponds to a raw uint value of 184 through
    # the nonlinear transformation
    #print(f"adaptive gamma: {1/inv_gamma:.2f} (orig {gamma:.2f}), brightness: {scaled_brightness:.2f} (orig {brightness:.2f})")
    # Apply gamma (in linear domain)
    img_gamma = np.clip((img_linear ** inv_gamma) * scaled_brightness, 0, 1.0)
    
    # Back to sRGB
    img_srgb = np.where(img_gamma <= 0.0031308, img_gamma * 12.92, 1.055 * (img_gamma ** (1 / 2.4)) - 0.055)
    #img_srgb[:,:,1] = img[:,:,1] + 0.5*(img_srgb[:,:,1] - img[:,:,1]) # half the influence of green channel adaptations due to the raw BAYER scheme
    img_srgb = (np.clip(img_srgb, 0, 1.0) * 255).astype(np.uint8)
    return img_srgb

def map_to_range(x, x_min=-10, x_max=10):
    return (np.clip(x, x_min, x_max) - x_min) / (x_max - x_min)

def sensor2world(yaw, pitch, roll, t_lon, t_lat, t_vert,):
    """ Transformation from Sensor to World System """
    cy = np.cos(np.radians(yaw))
    sy = np.sin(np.radians(yaw))
    cp = np.cos(np.radians(pitch))
    sp = np.sin(np.radians(pitch))
    cr = np.cos(np.radians(roll))
    sr = np.sin(np.radians(roll))

    # Compute combined rotation matrix: R = R_yaw @ R_pitch @ R_roll
    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr,  t_lon ],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr,  t_lat ],
        [    -sp,                 cp * sr,                 cp * cr,  t_vert],
        [      0,                       0,                       0,       1]
    ])


class Sys100Dataset(MultiFileDataset):
    """
    A custom dataset class to read the preprocessed SYS100 dataset.
    Each sample consist of a structured numpy array containing:

    >>> ImageTimestamp (<'numpy.int64'>)    # unix timestamp of the camera image
    >>> ImageFrameCounter (<'numpy.int64'>) # frame counter since last boot up of the camera
    >>> VideoId (<'numpy.int32'>)           # id of the videopath from which to extract the image from, see attribute 'VideoId2Path'
    >>> VideoFrameCounter (<'numpy.int64'>) # frame counter relative to the start of the recording of the video at videopath
    >>> RadarFrame ((64, 64, 7), float32)   # discretised radar clusters, the channels decode: depth(x)-offset, lateral(y)-offset, rangeGateLength, azimuth1, relative radial velocity, Radar Cross Section strength, predicted relative radial velocity for static measurements
    >>> CameraPoseCalibration (160 bytes):
    >>>     fRoll (<'numpy.float32'>)
    >>>     fPitch (<'numpy.float32'>)
    >>>     fYaw (<'numpy.float32'>)
    >>>     fRollSigma (<'numpy.float32'>)
    >>>     fPitchSigma (<'numpy.float32'>)
    >>>     fYawSigma (<'numpy.float32'>)
    >>>     uiRollQuality (<'numpy.int64'>)
    >>>     uiPitchQuality (<'numpy.int64'>)
    >>>     uiYawQuality (<'numpy.int64'>)
    >>>     uiTotalAngleQuality (<'numpy.int64'>)
    >>>     fTx (<'numpy.float32'>)
    >>>     fTy (<'numpy.float32'>)
    >>>     fTz (<'numpy.float32'>)
    >>>     fTxSigma (<'numpy.float32'>)
    >>>     fTySigma (<'numpy.float32'>)
    >>>     fTzSigma (<'numpy.float32'>)
    >>>     uiTxQuality (<'numpy.int64'>)
    >>>     uiTyQuality (<'numpy.int64'>)
    >>>     uiTzQuality (<'numpy.int64'>)
    >>>     uiTotalTranslQuality (<'numpy.int64'>)
    >>>     sTransform (48 bytes):
    >>>         fA00 (<'numpy.float32'>)
    >>>         fA10 (<'numpy.float32'>)
    >>>         fA20 (<'numpy.float32'>)
    >>>         fA01 (<'numpy.float32'>)
    >>>         fA11 (<'numpy.float32'>)
    >>>         fA21 (<'numpy.float32'>)
    >>>         fA02 (<'numpy.float32'>)
    >>>         fA12 (<'numpy.float32'>)
    >>>         fA22 (<'numpy.float32'>)
    >>>         fA03 (<'numpy.float32'>)
    >>>         fA13 (<'numpy.float32'>)
    >>>         fA23 (<'numpy.float32'>)
    >>> CameraPoseDynamic (160 bytes):
    >>>     fRoll (<'numpy.float32'>)
    >>>     fPitch (<'numpy.float32'>)
    >>>     fYaw (<'numpy.float32'>)
    >>>     fRollSigma (<'numpy.float32'>)
    >>>     fPitchSigma (<'numpy.float32'>)
    >>>     fYawSigma (<'numpy.float32'>)
    >>>     uiRollQuality (<'numpy.int64'>)
    >>>     uiPitchQuality (<'numpy.int64'>)
    >>>     uiYawQuality (<'numpy.int64'>)
    >>>     uiTotalAngleQuality (<'numpy.int64'>)
    >>>     fTx (<'numpy.float32'>)
    >>>     fTy (<'numpy.float32'>)
    >>>     fTz (<'numpy.float32'>)
    >>>     fTxSigma (<'numpy.float32'>)
    >>>     fTySigma (<'numpy.float32'>)
    >>>     fTzSigma (<'numpy.float32'>)
    >>>     uiTxQuality (<'numpy.int64'>)
    >>>     uiTyQuality (<'numpy.int64'>)
    >>>     uiTzQuality (<'numpy.int64'>)
    >>>     uiTotalTranslQuality (<'numpy.int64'>)
    >>>     sTransform (48 bytes):
    >>>         fA00 (<'numpy.float32'>)
    >>>         fA10 (<'numpy.float32'>)
    >>>         fA20 (<'numpy.float32'>)
    >>>         fA01 (<'numpy.float32'>)
    >>>         fA11 (<'numpy.float32'>)
    >>>         fA21 (<'numpy.float32'>)
    >>>         fA02 (<'numpy.float32'>)
    >>>         fA12 (<'numpy.float32'>)
    >>>         fA22 (<'numpy.float32'>)
    >>>         fA03 (<'numpy.float32'>)
    >>>         fA13 (<'numpy.float32'>)
    >>>         fA23 (<'numpy.float32'>)
    >>> GPS (124 bytes):
    >>>     Heading (<'numpy.float32'>)
    >>>     Latitude (<'numpy.float32'>)
    >>>     LatitudeDirection (<'numpy.int64'>)
    >>>     Longitude (<'numpy.float32'>)
    >>>     LongitudeDirection (<'numpy.int64'>)
    >>>     Speed (<'numpy.float32'>)
    >>>     DateTime (<'numpy.float32'>)
    >>>     Year (<'numpy.int64'>)
    >>>     Month (<'numpy.int64'>)
    >>>     Day (<'numpy.int64'>)
    >>>     Hour (<'numpy.int64'>)
    >>>     Minute (<'numpy.int64'>)
    >>>     Second (<'numpy.int64'>)
    >>>     Millisecond (<'numpy.int64'>)
    >>>     Ticks (<'numpy.int64'>)
    >>>     TotalSecondsOfDay (<'numpy.float32'>)
    >>>     TotalMillisecondOfDay (<'numpy.float32'>)
    >>>     TicksOfDay (<'numpy.int64'>)
    >>>     Valid (<'numpy.int64'>)
    >>> VehDyn (220 bytes):
    >>>     uiVersionNumber (<'numpy.int64'>)
    >>>     sSigHeader (32 bytes):
    >>>         uiTimeStamp (<'numpy.int64'>)
    >>>         uiMeasurementCounter (<'numpy.int64'>)
    >>>         uiCycleCounter (<'numpy.int64'>)
    >>>         eSigStatus (<'numpy.int64'>)
    >>>     longitudinal (44 bytes):
    >>>         Velocity (<'numpy.float32'>)
    >>>         VelocityTimestamp (<'numpy.int64'>)
    >>>         VelocityRaw (<'numpy.float32'>)
    >>>         Accel (<'numpy.float32'>)
    >>>         AccelTimestamp (<'numpy.int64'>)
    >>>         varVelocity (<'numpy.float32'>)
    >>>         varAccel (<'numpy.float32'>)
    >>>         VelocityCorrectionQuality (<'numpy.int64'>)
    >>>     lateral (112 bytes):
    >>>         yawRate (24 bytes):
    >>>             YawRate (<'numpy.float32'>)
    >>>             YawRateTimestamp (<'numpy.int64'>)
    >>>             YawRateRaw (<'numpy.float32'>)
    >>>             YawAngle (<'numpy.float32'>)
    >>>             Variance (<'numpy.float32'>)
    >>>         curve (28 bytes):
    >>>             Curve (<'numpy.float32'>)
    >>>             CurveTimestamp (<'numpy.int64'>)
    >>>             VarCurve (<'numpy.float32'>)
    >>>             CrvError (<'numpy.float32'>)
    >>>             CrvConf (<'numpy.int64'>)
    >>>         drvIntCurve (20 bytes):
    >>>             Curve (<'numpy.float32'>)
    >>>             CurveTimestamp (<'numpy.int64'>)
    >>>             Variance (<'numpy.float32'>)
    >>>             Gradient (<'numpy.float32'>)
    >>>         accel (16 bytes):
    >>>             LatAccel (<'numpy.float32'>)
    >>>             LatAccelTimestamp (<'numpy.int64'>)
    >>>             Variance (<'numpy.float32'>)
    >>>         slipAngle (8 bytes):
    >>>             SideSlipAngle (<'numpy.float32'>)
    >>>             Variance (<'numpy.float32'>)
    >>>         selfSteering (16 bytes):
    >>>             RoadBankAngle (<'numpy.float32'>)
    >>>             QuRoadBankAngle (<'numpy.float32'>)
    >>>             SelfSteerGradEst (<'numpy.float32'>)
    >>>             QuSelfSteerGradEst (<'numpy.float32'>)
    >>>     motionState (24 bytes):
    >>>         MotState (<'numpy.int64'>)
    >>>         Confidence (<'numpy.int64'>)
    >>>         bRollerTestBench (<'numpy.int64'>)
    
    As well as recording wide metadata saved as attributes of each h5 file:

    - 'valid_entries': int, the number of valid entries in the h5 file dataset 'recordings'
    - 'VideoId2Path': list of str, the paths to each video file corresponding to
        the video where the sample is synchronised to. 'VideoId' is one of the
        columns of the 'recordings' dataset.
    - 'CameraMounting':
        >>> LatPos (<'numpy.float32'>)
        >>> LongPos (<'numpy.float32'>)
        >>> VertPos (<'numpy.float32'>)
        >>> LongPosToCoG (<'numpy.float32'>)
        >>> PitchAngle (<'numpy.float32'>)
        >>> Orientation (<'numpy.int64'>)
        >>> RollAngle (<'numpy.float32'>)
        >>> YawAngle (<'numpy.float32'>)
    - 'VehPar':
        >>> uiVersionNumber (<'numpy.uint64'>)
        >>> sSigHeader (32 bytes):
        >>>     uiTimeStamp (<'numpy.int64'>)
        >>>     uiMeasurementCounter (<'numpy.int64'>)
        >>>     uiCycleCounter (<'numpy.int64'>)
        >>>     eSigStatus (<'numpy.int64'>)
        >>> vehParMain (64 bytes):
        >>>     SelfSteerGrad (<'numpy.float32'>)
        >>>     WheelBase (<'numpy.float32'>)
        >>>     TrackWidthFront (<'numpy.float32'>)
        >>>     TrackWidthRear (<'numpy.float32'>)
        >>>     VehWeight (<'numpy.float32'>)
        >>>     CntrOfGravHeight (<'numpy.float32'>)
        >>>     AxisLoadDistr (<'numpy.float32'>)
        >>>     WhlLoadDepFrontAxle (<'numpy.float32'>)
        >>>     WhlLoadDepRearAxle (<'numpy.float32'>)
        >>>     WhlCircumference (<'numpy.float32'>)
        >>>     DrvAxle (<'numpy.int64'>)
        >>>     WhlTcksPerRev (<'numpy.int64'>)
        >>>     FrCrnrStiff (<'numpy.float32'>)
        >>>     ReCrnrStiff (<'numpy.float32'>)
        >>> vehParAdd (44 bytes):
        >>>     VehicleWidth (<'numpy.float32'>)
        >>>     VehicleLength (<'numpy.float32'>)
        >>>     CurbWeight (<'numpy.float32'>)
        >>>     OverhangFront (<'numpy.float32'>)
        >>>     FrontAxleRoadDist (<'numpy.float32'>)
        >>>     WheelWidth (<'numpy.float32'>)
        >>>     PassableHeight (<'numpy.float32'>)
        >>>     DistCameraToHoodX (<'numpy.float32'>)
        >>>     DistCameraToHoodY (<'numpy.float32'>)
        >>>     SteeringVariant (<'numpy.int64'>)
        >>> sensorMounting (36 bytes):
        >>>     LatPos (<'numpy.float32'>)
        >>>     LongPos (<'numpy.float32'>)
        >>>     VertPos (<'numpy.float32'>)
        >>>     LongPosToCoG (<'numpy.float32'>)
        >>>     PitchAngle (<'numpy.float32'>)
        >>>     Orientation (<'numpy.int64'>)
        >>>     RollAngle (<'numpy.float32'>)
        >>>     YawAngle (<'numpy.float32'>)
        >>> sensor (24 bytes):
        >>>     CoverDamping (<'numpy.float32'>)
        >>>     fCoverageAngle (<'numpy.float32'>)
        >>>     fLobeAngle (<'numpy.float32'>)
        >>>     fCycleTime (<'numpy.float32'>)
        >>>     uNoOfScans (<'numpy.int64'>)
        >>> sensorFovCrop (16 bytes):
        >>>     FovMaxPitch (<'numpy.float32'>) (mostly 0)
        >>>     FovMinPitch (<'numpy.float32'>) (mostly 0)
        >>>     FovMaxYaw (<'numpy.float32'>)   (mostly 0)
        >>>     FovMinYaw (<'numpy.float32'>)   (mostly 0)
    """

    def __init__(
        self, 
        source: str | Path | list[str] | list[Path] | list[str | Path],
        *,
        extract_video : bool = True,
        out_img_size: tuple[int, int] = (512, 512), # in height x width !
        out_dtype: torch.dtype = torch.float32,
        source_is_summary_csv: bool = False,
        valid_file_ending: str = ".h5",
        valid_file_must_contain: str = "",
        recursive: bool = True,
        max_files: int | None = None,
        subsampling_step: int | None = None,
        target_dataset_size: int | None = None,
        random_subsampling: bool = True,
        seed: int | None = None,
        strict: bool = True,
        careful: bool = False,
        display_stats: bool = False,
        extraction_function = "cv2",
        undistort: bool = False,
        plot_radar: bool = False,
        project_radar: bool = False,
        threshold_cluster_present = 0.1,
        overlay_distortion_grid: bool = False,
        return_metadata: bool = False,
        gamma_adapt: bool = False,
        fovs = (110, 48.4),
        look_for_old_radar = False,
        toy_radar_data = False,
        # (110, 48.4) is the biggest field of view for undistorted frames without 
        # blind spots in the frame, recommended full resolution (544,1792)
        # (110, 75)  contains basically all details from the original fisheye
        # image, but with large blindspots, recommended full resolution ()
    ) -> None:
        
        self.careful = careful
        self.plot_radar = plot_radar
        self.project_radar = project_radar
        self.threshold_cluster_present = threshold_cluster_present
        self.gamma_adapt = gamma_adapt
        self.return_metadata = return_metadata
        self.overlay_distortion_grid = overlay_distortion_grid
        self.look_for_old_radar = look_for_old_radar
        self.toy_radar_data = toy_radar_data
        self._ready_for_video = False
        
        super().__init__(
            source,
            source_is_summary_csv=source_is_summary_csv,
            valid_file_ending=valid_file_ending,
            valid_file_must_contain=valid_file_must_contain,
            recursive=recursive,
            max_files=max_files,
            subsampling_step=subsampling_step,
            target_dataset_size=target_dataset_size,
            random_subsampling=random_subsampling,
            seed=seed,
            strict=strict,
            display_stats=display_stats
        )

        self.out_img_size = out_img_size
        self.undistort = undistort
        self.undistorter = None
        self.figure = None
        self.fovs = (fovs[0], fovs[1]) # to make sure we have a basic tuple not e.g. a OmegaConfList
        #print(f"Sys100dataset undistort {self.undistort} overlay_distortion_grid {self.overlay_distortion_grid} project_radar {self.project_radar}")
        if not self.undistort and not self.overlay_distortion_grid:
            self.image_transform = v2.Compose([
                # note that the extract functions will already return the outputs
                # in CHW format, so no need to transpoes them
                v2.Resize(self.out_img_size, antialias=False),
                v2.ToDtype(out_dtype),
                v2.Normalize(mean=[127.5, 127.5, 127.5], 
                            std=[127.5, 127.5, 127.5])  # Normalize to [-1, 1]
            ])
        else:
            self.image_transform = v2.Compose([
                # note that in this case we will only get the image in HWC format
                # but already as a out_img_size shaped image
                v2.ToImage(),
                v2.ToDtype(out_dtype),
                # Normalize to [-1, 1]
                v2.Normalize(mean=[127.5, 127.5, 127.5], 
                            std=[127.5, 127.5, 127.5])  
            ])
        
        self.radar_normalisation = np.array([2., 2., 1., np.pi/2, 20., 10., 10.], dtype=float)
        self.radar_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(out_dtype),
            # Format: x_off, y_off, rangegatelength, azimuth1, RCS, vel, staticVel
            # Normalize to to near N(-1, 1)
            v2.Normalize(mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                         std=self.radar_normalisation), # type: ignore[attr-defined]
            v2.RandomHorizontalFlip(p=1.0), # make the radar image more alligned 
            v2.RandomVerticalFlip(p=1.0),   # to the image when they are concatenated
                                            # along the channel dimension 
        ])

        self.extract_video = extract_video
        self.set_parameters(
            extraction_function=extraction_function
        )
    
    def set_parameters(
        self,
        *,
        extraction_function: str | None = None,
        max_files: int | None = None,
        subsampling_step: int | None = None,
        target_dataset_size: int | None = None,
        random_subsampling: bool | None = None,
        seed: int | None = None,
        extract_video: bool | None = None,
        project_radar: bool | None = None,
        plot_radar: bool | None = None,
        gamma_adapt: bool | None = None,
        return_metadata: bool | None = None,
    ) -> None:
        super().set_parameters(max_files=max_files, 
                               subsampling_step=subsampling_step, 
                               target_dataset_size=target_dataset_size, 
                               random_subsampling=random_subsampling, 
                               seed=seed)
        if extraction_function is not None:
            if extraction_function == "cv2":
                #print("Sys100Dataset using extract_frame_via_cv2")
                self.extraction_function = extract_frame_via_cv2
            #elif extraction_function == "pyav":
            #    print("using extract_frame_via_pyav")
            #    self.extraction_function = extract_frame_via_pyav
            else:
                raise ValueError(f"Invalid extract_function: {extraction_function}. "
                                +f"Must be 'cv2', 'pyav'.")
        if extract_video is not None:
            self.extract_video = extract_video
        if project_radar is not None:
            self.project_radar = project_radar
        if plot_radar is not None:
            self.plot_radar = plot_radar
        if gamma_adapt is not None:
            self.gamma_adapt = gamma_adapt
        if return_metadata is not None:
            self.return_metadata = return_metadata
    
    def _get_proper_video_path(self, video_path: str , file_path: str ) -> str:
        """
        Returns the proper video path for the given video path and file path.
        If the video path is a local file, it will return the local file path.
        If the video path is a remote file, it will return the remote file path.
        """
        video_Path = Path(video_path)
        if video_Path.is_absolute():
            dir_path   = str(video_Path.parent)
            video_path = str(video_Path.name)
        else:
            dir_path = os.path.dirname(file_path)
        local_file = [str(p) for p in Path(dir_path).glob(video_path+"*")
                      if not str(p).endswith(".lock")]
        if len(local_file) > 0:
            return local_file[0]
        dir_path2 = os.path.dirname(file_path).replace("sys100","sys100_preprocessed")
        local_file = [str(p) for p in Path(dir_path2).glob(video_path+"*")
                      if not str(p).endswith(".lock")]
        if len(local_file) > 0:
            return local_file[0]
        raise FileNotFoundError(f"Video file '{video_path}' not found. " +
                                f"Not even in '{dir_path}' or '{dir_path2}'")
        
        #local_file = os.path.join(os.path.dirname(file_path), os.path.basename(video_path))
        #if os.path.exists(local_file):
        #    return local_file
        #el
        #if video_path.startswith("/p/scratch/nxtaim-1/proprietary/continental/sys100/parquets/"):
        #    local_file2 = ("/p/data1/nxtaim/proprietary/continental/sys100_preprocessed/"
        #                   +video_path[60:-4] + "_gop2_crf16_veryfast.mp4")
        #    if not os.path.exists(local_file2):
        #        raise FileNotFoundError(f"Video file '{video_path}' not found. "
        #                            +f"Not even at '{local_file}' or '{local_file2}'")
        #    return local_file2
    
    def _get_samples_metadata_from_filepath(self, path: str | Path):
        with h5py.File(path, 'r') as f:
            valid = int(cast(int, f.attrs['valid_entries']))
            if not self.careful:
                return valid, None
            frames = []
            for video_path in cast(list[str], f.attrs['VideoId2Path']):
                #print(f"_get_samples_metadata_from_filepath on path '{path}' | video_path '{video_path}'")
                video_path = self._get_proper_video_path(video_path=str(video_path), file_path=str(path))
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video file '{video_path}'")
                frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            safe_valid = min(valid, min(frames)*len(frames))
            if np.abs(1-safe_valid / valid) > 0.02:
                print(f"WARNING: H5 file '{path}' defines {valid} samples, but "+
                      f"only {safe_valid} are safe as the video files only "+
                      f"contain {frames} ({sum(frames)}) frames")

            return safe_valid, None
    
    def _getitem(
        self,
        file_path: str,
        sample_idx: int,
        metadata: np.ndarray,
    ): # -> Dict[np.int64, torch.Tensor, torch.Tensor, str]:
        """
        Loads the discretized radar frame and the corresponding image

        Parameters
        ----------
        file_path : str
            Path of the file that contains the sample.
        sample_idx : int
            Index of the datapoint that shall be collected from that file.
            ``sample_idx = 0`` would request the first datapoint of this file,
            while ``sample_idx = n_samples-1`` would request the last one, where
            ``n_samples`` is the number of datapoints in this file as reported by
            :py:meth:`_get_samples_metadata_from_filepath`.
        metadata : structured ndarray | None
            Companion metadata for the file produced by
            :py:meth:`_get_samples_metadata_from_filepath`.
        """
        start5f = time.time()
        try:
            valid = "'Could not find number of valid entries'"
            with h5py.File(file_path, 'r') as f:
                valid = f.attrs['valid_entries']
                row_data = cast(h5py.Dataset, f['recordings'])[sample_idx]
                video_path = cast(list[str], f.attrs['VideoId2Path'])[row_data['VideoId']]
                CameraMounting = f.attrs['CameraMounting']
                VehPar = f.attrs['VehPar']
                old_radar = None
                old_radar_frame = None
                masked_points = None
                if self.look_for_old_radar:
                    try:
                        old_radar = f['old_radar'][sample_idx]
                    except Exception:
                        pass
                    try:
                        masked_points = f['masked_old_radar'][sample_idx]
                        #print("found masked points")
                    except Exception:
                        pass
        except Exception as e:
            print(f"\nWhen trying to open '{file_path}' at index {sample_idx} "+ 
                  f"({type(sample_idx)}) which should contain {valid} valid samples")
            print("Got the error:")
            raise e

        h5_time = time.time() - start5f
        img_time = 0
        undistort_time = 0
        gamma_time = 0
        
        # Extract choice IMU metadata
        try:
            vehdyn = row_data['VehDyn']
            velocity = float(vehdyn['longitudinal']['Velocity'])
            yaw_rate = vehdyn['lateral']['yawRate']['YawRate']
        except Exception as e:
            pprint_structnp(row_data)
            raise e
        
        # extract radar data
        radar_data = row_data['RadarFrame']
        dopplerRange = row_data['RadarClusterHead']['f_AmbFreeDopplerRange']
        SensorMounting = VehPar['sensorMounting'] # type: ignore[attr-defined]
        if SensorMounting['Orientation'] == 1:
            radar_data = radar_data[:,::-1,:]
            radar_data[:,:,1] *= -1 # the lateral offset also need to revert their direction
            radar_data = np.ascontiguousarray(radar_data)
        radar_frame = self.radar_transform(radar_data)


        if old_radar is not None:
            if SensorMounting['Orientation'] == 1:
                old_radar = old_radar[:,::-1,:]
                old_radar[:,:,1] *= -1 # the lateral offset also need to revert their direction
                old_radar = np.ascontiguousarray(old_radar)
                #if masked_points is not None:
                #    masked_points = np.ascontiguousarray(masked_points[:,::-1])
            old_radar_frame = self.radar_transform(old_radar)
            if masked_points is not None:
                masked_points = self.radar_transform(masked_points)

        rad_inv = sensor2world(
            yaw   = SensorMounting['YawAngle'],
            pitch = SensorMounting['PitchAngle'],
            roll  = SensorMounting['RollAngle'],
            t_lon = SensorMounting['LongPos'],
            t_lat = SensorMounting['LatPos'],
            t_vert= SensorMounting['VertPos'],
        )
        cameraPoseDynamic = row_data['CameraPoseDynamic']['sTransform'].reshape((1,)).view('<f4').reshape((4,3)).T
        cameraPoseCalibration = row_data['CameraPoseCalibration']['sTransform'].reshape((1,)).view('<f4').reshape((4,3)).T
        rad2cam = (cameraPoseDynamic @ rad_inv)[:3, :4]

        # Extract The frame from the corresponding video
        if self.extract_video:
            video_path = self._get_proper_video_path(video_path=video_path, file_path=file_path)
            video_frame_counter = row_data['VideoFrameCounter']
            
            if self.undistort or self.overlay_distortion_grid:
                # image in height, width, channels format
                imgstart = time.time()
                image = self.extraction_function(
                    video_path=video_path,
                    frame_idx=video_frame_counter,
                    metadata=None,
                    order="HWC",
                ).numpy()
                image_size = image.shape[:2]
                img_time = time.time() - imgstart

                if int(CameraMounting['Orientation']) == 1: # type: ignore[attr-defined]
                    image = image[:,::-1,:]
                # using adaptive gamma rates to not compress details in
                # dark images too much. As even in most bright images gamma
                # impact is reduced, we use an even lower gamma value
                if self.gamma_adapt:
                    gamma_start = time.time()
                    #image = approximate_soft_gamma_correction(image, gamma=0.5, brightness=1., perc=95)
                    image = approximate_soft_gamma_correction(image, gamma=0.25, brightness=10., perc=95)
                    gamma_time = time.time() - gamma_start
                fovs = self.fovs
                
                if (self.undistorter is None or 
                    self.undistorter.fisheye_cam.height != image.shape[0] or
                    self.undistorter.fisheye_cam.width  != image.shape[1] or
                    self.undistorter.pinhole_cam.height != self.out_img_size[0] or
                    self.undistorter.pinhole_cam.width  != self.out_img_size[1]
                ):  
                    self.undistorter = Undistorter(
                        pinhole_cam_params={
                            "height" : self.out_img_size[0],
                            "width" : self.out_img_size[1],
                            "hFov" : self.fovs[0],
                            "vFov" : self.fovs[1],
                            "rel_x_centre" : 0.5,
                            "rel_y_centre" : 0.4370,
                        },
                        fisheye_cam_params={
                            "height": image.shape[0],
                            "width": image.shape[1],
                        },
                        compute_grids=self.overlay_distortion_grid,
                    )
                
                if self.undistort:
                    undistort_start = time.time()
                    image = self.undistorter.undistort_fisheye_to_pinhole(image)
                    undistort_time = time.time() - undistort_start
                    if self.overlay_distortion_grid:
                        image = cv2.addWeighted(
                            src1=image, alpha=1,
                            src2=self.undistorter.pinhole_grid, beta=0.2,
                            gamma=0.)
                else:
                    if self.overlay_distortion_grid:
                        image = cv2.addWeighted(
                            src1=image, alpha=1,
                            src2=self.undistorter.fisheye_grid, beta=0.2,
                            gamma=0.)
            else:
                # image in channels, height, width format as expected by the model
                imgstart = time.time()
                image = self.extraction_function(
                    video_path=video_path,
                    frame_idx=video_frame_counter,
                    metadata=None,
                    order="CHW",
                )
                image_size = image.shape[1:]
                img_time = time.time() - imgstart
                if int(CameraMounting['Orientation']) == 1: # type: ignore[attr-defined]
                    image = image[:,::-1,:]
                if self.gamma_adapt:
                    gamma_start = time.time()
                    image = approximate_soft_gamma_correction(image, gamma=0.5, brightness=4., perc=95)
                    gamma_time = time.time() - gamma_start
                fovs = (110, 75)
            aspect_ratio = image.shape[1] / image.shape[0]
            aspect_fraction = Fraction(aspect_ratio).limit_denominator(16)
            aspect = (aspect_fraction.numerator, aspect_fraction.denominator)
            #print(f"image shape: {image.shape}, aspect ration {image.shape[1] / image.shape[0]:.6f}, fraction: {aspect_fraction} -> {aspect}")
            image = self.image_transform(image)

        #print(f"h5time {h5time} ({type(h5time)}), imgtime {imgtime} ({type(imgtime)})")
        if self.toy_radar_data:
            c, h, w = radar_frame.shape
            rdevice, rdtype = radar_frame.device, radar_frame.dtype
            hs, ws = (int(np.ceil(h/8)), int(np.ceil(w/8)))
            h_idx = torch.arange(hs, device=rdevice).view(1, hs, 1)
            w_idx = torch.arange(ws, device=rdevice).view(1, 1, ws)
            c_idx = torch.arange(c, device=rdevice).view(c, 1, 1)
            # Value per patch, restricted to -1, 0, 1
            patch_vals = (((h_idx + w_idx + c_idx) % 3)/2 - .5).to(rdtype)   # shape (H, W, C)
            # Expand each patch to 8x8 and have the same image for each batch
            toy_frame = patch_vals.repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)
            radar_frame = toy_frame[:c,:h,:w,].contiguous()

        out_dict = {
            'radar': radar_frame,  # assuming already a torch.Tensor
            'txt': f'A {"" if self.undistort else "fisheye"} photo from a high centered front-facing camera',
            'timestamp': torch.tensor(row_data['ImageTimestamp'], dtype=torch.int64),
            'velocity': torch.tensor(velocity, dtype=torch.float32),
            'yaw_rate': torch.tensor(yaw_rate, dtype=torch.float32),
            'dopplerRange': torch.tensor(dopplerRange, dtype=torch.float32),
            'rad2cam': torch.from_numpy(rad2cam).to(torch.float32),
            'h5_time': torch.tensor(h5_time, dtype=torch.float32),
            'img_time': torch.tensor(img_time, dtype=torch.float32),
            'undistort_time': torch.tensor(undistort_time, dtype=torch.float32),
            'gamma_time': torch.tensor(gamma_time, dtype=torch.float32),
            'total_time': torch.tensor(time.time() - start5f, dtype=torch.float32),
        }
        if self.extract_video:
            out_dict['jpg'] = image
            out_dict['real_hw'] = torch.tensor(image_size)
            out_dict['aspect'] = torch.tensor(aspect, dtype=torch.float32)
            out_dict['fovs'] = torch.tensor(fovs, dtype=torch.float32)
        if self.return_metadata:
            out_dict['meta_data'] = row_data
            out_dict['video_path'] = video_path
            out_dict['CameraMounting'] = CameraMounting
            out_dict['VehPar'] = VehPar
        if old_radar_frame is not None:
            out_dict['old_radar'] = old_radar_frame
        if masked_points is not None:
            out_dict['masked_points'] = masked_points
        return out_dict
    
    def _on_video_start(self):
        """
        Optional hook that is called before the video creation in 
        :py:meth:`create_sample_video` starts.
        Can be used to set up variables for the video creation, such as
        a matplotlib figure and artists to update.
        """
        self.pinhole_cam = None
        self.fisheye_cam = None
        
        if not self.extract_video:
            self.plot_radar = True
        
        if self.plot_radar:
            if self.extract_video:
                if self.project_radar:
                    self.figure = plt.figure(figsize=(12,8), layout="tight", dpi=200)
                    gs = gridspec.GridSpec(4, 6, figure=self.figure)
                    self.camera_axis = self.figure.add_subplot(gs[0:2, 0:4])
                    self.camera_axis.set_axis_off()
                    self.camera_img = self.camera_axis.imshow(
                        np.zeros((self.out_img_size[0], self.out_img_size[1], 3),
                                 dtype=np.uint8),
                        origin='upper',
                    )
                    offset = 2
                    self.scatter_axis = self.figure.add_subplot(gs[:2, 4:])
                    for r in range(10,200,10):
                        circle = Circle((0,0), r, fill=False, linestyle='-', alpha=0.1)
                        self.scatter_axis.add_patch(circle)
                    self.scatter_axis.set_title("BEV of reconstructed Radar clusters")
                    self.scatter_axis.set_xlim( (-50, 50) )
                    self.scatter_axis.set_ylim( 0, 104 )
                    if self.look_for_old_radar:
                        self.scatter_axis2 = self.figure.add_subplot(gs[2:4, 4:])
                        self.scatter_axis2.set_title("BEV of GT reconstructed Radar clusters")
                        self.scatter_axis2.set_axis_off()
                        for r in range(10,200,10):
                            circle = Circle((0,0), r, fill=False, linestyle='-', alpha=0.1)
                            self.scatter_axis2.add_patch(circle)
                        self.scatter_plot_masked  = self.scatter_axis.scatter(  [], [], color=[0.,1.,0.], marker='x', alpha=0.4, s=2*rcParams['lines.markersize'] ** 2)
                        self.scatter_plot2_masked = self.scatter_axis2.scatter( [], [], color=[0.,1.,0.], marker='x', alpha=0.4, s=2*rcParams['lines.markersize'] ** 2)
                        self.scatter_plot2 = self.scatter_axis2.scatter( [], [], )
                        self.scatter_axis2.set_xlim( (-50, 50) )
                        self.scatter_axis2.set_ylim( 0, 104 )
                    self.scatter_plot = self.scatter_axis.scatter( [], [], )

                else:
                    self.figure = plt.figure(figsize=(12,6), layout="tight", dpi=200)
                    gs = gridspec.GridSpec(4, 4, figure=self.figure)
                    self.camera_axis = self.figure.add_subplot(gs[0:2, 0:4])
                    self.camera_axis.set_axis_off()
                    self.camera_img = self.camera_axis.imshow(
                        np.zeros((self.out_img_size[0], self.out_img_size[1], 3),
                                 dtype=np.uint8),
                        origin='upper',
                    )
                    offset = 2
                    self.scatter_axis = None
                    self.scatter_plot = None
            else:
                self.figure = plt.figure(figsize=(6.4*2,2*2), layout="tight")
                gs = gridspec.GridSpec(2, 6, figure=self.figure,
                                       width_ratios=[1.1,1.1,1.1,1.1,1,1])
                self.camera_axis = None
                self.camera_img = None
                offset = 0
                self.scatter_axis = self.figure.add_subplot(gs[:2, 4:])
                for r in range(10,200,10):
                    circle = Circle((0,0), r, fill=False, linestyle='-', alpha=0.1)
                    self.scatter_axis.add_patch(circle)
                self.scatter_axis.set_title("BEV of reconstructed Radar clusters")
                self.scatter_axis.set_xlim( (-55, 55) )
                self.scatter_axis.set_ylim( 0, 110 )
                self.scatter_plot = self.scatter_axis.scatter( [], [], )

            self.radar_axis = [self.figure.add_subplot(gs[offset+i//4, i%4]) for i in range(8)]
            self.radar_imgs = []
            self.radar_cbars = []
            for ax, content, v, show in zip(
                self.radar_axis, 
                ['depth offset', '- lateral offset', 'rangeGateLength', 'azimuth1',
                  'RadarCrossSection', 'relativeRadialVelocity','expected static\nrelativeRadialVelocity', 
                  'RadialVelocity\nof dynamic objects'],
                [2,2,0.6,np.pi/2,
                 50,10,10,10,],
                [True,True,True,True,
                 True,True,True,True,]
            ):
                ax.set_axis_off()
                if show:
                    self.radar_imgs.append(ax.imshow(
                        np.zeros((64,64, 1)), cmap=cmap, vmin=-v, vmax=v,
                        origin='upper',
                    ))
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)  # Fixed size
                    self.radar_cbars.append(self.figure.colorbar(
                        self.radar_imgs[-1], cax=cax,
                    ))
                else:
                    self.radar_imgs.append(None)
                    self.radar_cbars.append(None)
                ax.set_title(content, fontsize=7)

            self.figure.tight_layout()
            self.figure.set_layout_engine('none')  # type: ignore[attr-defined]

        self._ready_for_video = True

    def _get_image_representation(
        self, 
        file_path: str,
        sample_idx: int,
        metadata: np.ndarray,
        batch = None,
        alternative_data = False
    ) -> Tuple[Any, str]:
        """
        Creates an image and additional text to visualise the specified data point
        and document from where it was taken for :py:meth:`create_sample_video`.

        Parameters
        ----------
        file_path : str
            Path of the file that contains the sample.
        sample_idx : int
            Index of the datapoint that shall be collected from that file.
            ``sample_idx = 0`` would request the first datapoint of this file,
            while ``sample_idx = n_samples-1`` would request the last one, where
            ``n_samples`` is the number of datapoints in this file as reported by
            :py:meth:`_get_samples_metadata_from_filepath`.
        metadata : structured ndarray | None
            Companion metadata for the file produced by
            :py:meth:`_get_samples_metadata_from_filepath`.
        batch :
            If the batch is already known, e.g. the output of 
            _getitem(file_path, sample_idx, metadata).
        
        Returns
        -------
        A tupel with the first element being an image representation:
          - A numpy array of shape (H, W, C) with dtype uint8 of RGB values
          - A PIL Image object
          - A matplotlib Figure object
        the second element being a string containing text that shall be printed
        over the image.
        """
        alt_out = {}
        if (   not self._ready_for_video 
            or self.figure is None 
            or not plt.fignum_exists(getattr(self.figure, "number", None))
        ):
            self._on_video_start()
        if batch is None:
            batch = self._getitem(file_path=file_path, 
                                  sample_idx=sample_idx, 
                                  metadata=metadata,
                                  )
        else:
            new_batch = {
                'radar':        batch['radar'] if batch['radar'].ndim == 3 else batch['radar'][0],
                'txt':          batch['txt'] if isinstance(batch['txt'], str) else batch['txt'][0],
                'velocity':     batch['velocity'] if batch['velocity'].ndim == 0 else batch['velocity'][0],
                'dopplerRange': batch['dopplerRange'] if batch['dopplerRange'].ndim == 0 else batch['dopplerRange'][0],
                'yaw_rate':     batch['yaw_rate'] if batch['yaw_rate'].ndim == 0 else batch['yaw_rate'][0],
                'rad2cam':      batch['rad2cam'] if batch['rad2cam'].ndim == 2 else batch['rad2cam'][0],
            }
            if self.extract_video:
                new_batch['jpg']    = batch['jpg'] if batch['jpg'].ndim == 3 else batch['jpg'][0]
                new_batch['real_hw']= batch['real_hw'] if batch['real_hw'].ndim == 1 else batch['real_hw'][0]
                new_batch['aspect'] = batch['aspect'] if batch['aspect'].ndim == 0 else batch['aspect'][0]
                new_batch['fovs']   = batch['fovs'] if batch['fovs'].ndim == 0 else batch['fovs'][0]
            if self.return_metadata:
                # cannot be a batched tensor then
                new_batch['CameraMounting'] = batch['CameraMounting']
                new_batch['VehPar'] = batch['VehPar']
            if 'old_radar' in batch:
                new_batch['old_radar'] = batch['old_radar'] if batch['old_radar'].ndim == 3 else batch['old_radar'][0]
            if 'masked_points' in batch:
                new_batch['masked_points'] = batch['masked_points'] if batch['masked_points'].ndim == 3 else batch['masked_points'][0]
            batch = new_batch

        
        velocity = batch['velocity'].item()
        doppler  = batch['dopplerRange'].item() if 'dopplerRange' in batch else 20
        yaw_rate = batch['yaw_rate'].item()
        title = (f"{sample_idx:>5} - "+
                 f"{os.path.basename(file_path)}\ntxt: \"{batch['txt']}\"\n"+
                 f"velo: {velocity:.2f}, yaw_rate: {yaw_rate:.3f}\n"
                 )
        if self.return_metadata:
            cam_orientation = batch['CameraMounting']['Orientation']
            rad_orientation = batch['VehPar']['sensorMounting']['Orientation']
            title += f"camera orientation: {cam_orientation} radar: {rad_orientation}\n"

        if self.extract_video:
            img = batch['jpg'].cpu().numpy().transpose(1, 2, 0)
            img = np.ascontiguousarray(127.5 * img + 127.5, dtype=np.uint8)
            title += f"aspect: '{batch['aspect']}', fovs: {batch['fovs']}\n"

        radar_data = batch['radar'].cpu().numpy()
        # channel dim is currently in front but we need it at the back
        radar_grid = radar_data.transpose(1, 2, 0)[::-1,::-1] * self.radar_normalisation[None, None, :]
        radar_clusters = point_from_gridentry_vectorized(grid=radar_grid, thres=self.threshold_cluster_present)
        cluster_vel = radar_clusters[:,5]-radar_clusters[:,6] # normalise relative velocity by substracting influence of ego velocity
        cluster_vel -= doppler * np.floor(cluster_vel/doppler +1/2)
        cluster_vel_colors = np.stack([
            np.clip(cluster_vel,0,10)/10,               # velocity > 0 -> red dot (yellow if close)
            np.zeros_like(cluster_vel),#np.clip(150-radar_clusters[:,0],0,150)/150, # close point  -> green, far away -> black
            np.clip(cluster_vel,-10,0)/-10,             # velocity < 0 -> blue dot (cyan if close)
        ], axis=1,)

        cluster_vel_sizes = (2*radar_clusters[:,4]/100 + 3).astype(int)

        if self.extract_video:
            if self.project_radar:
                # project the radar points onto the image
                rad2cam = batch['rad2cam'].cpu().numpy()
                camera_clusters = radar_clusters[:,:2] @ rad2cam[:3, :2].T + rad2cam[:3, 3:].T
                
                projected_circles = [
                    (np.array([[r*np.sin(a), r*np.cos(a)] 
                               for a in np.linspace(0, 2*np.pi, num=int(r**0.5)*10, endpoint=False)])
                        @ rad2cam[:3, :2].T + rad2cam[:3, 3:].T)
                    for r in range(10, 101, 10)
                ]
                img_scale = np.sqrt( img.shape[0]**2 + img.shape[1]**2 ) / 1000
                if self.undistort:
                    if self.pinhole_cam is None or (
                        self.pinhole_cam.height != img.shape[0] or
                        self.pinhole_cam.width  != img.shape[1]
                    ):
                        h,w = batch['real_hw']
                        self.pinhole_cam = PinholeCamera(
                            height = img.shape[0],
                            width = img.shape[1],
                            hFov = self.fovs[0],
                            vFov = self.fovs[1],
                            rel_x_centre = 0.5,
                            rel_y_centre = 0.4370,
                            fisheye_cam = FisheyeCamera(height=h, width=w)
                        )
                    img = self.pinhole_cam.project_lines(
                        image=img, lines_3d=projected_circles, alpha=0.4,
                        thickness=int(round(img_scale))
                    )
                    img = self.pinhole_cam.project_points(
                        image=img,
                        points=camera_clusters,
                        colors=(cluster_vel_colors*255).astype(np.uint8),
                        sizes=cluster_vel_sizes * img_scale,
                    )
                else:
                    if self.fisheye_cam is None or (
                        self.fisheye_cam.height != img.shape[0] or
                        self.fisheye_cam.width  != img.shape[1]
                    ):
                        self.fisheye_cam = FisheyeCamera(
                            height = img.shape[0],
                            width = img.shape[1],
                        )
                    img = self.fisheye_cam.project_lines(
                        image=img, lines_3d=projected_circles, alpha=0.4,
                        thickness=int(round(img_scale))
                    )
                    img = self.fisheye_cam.project_points(
                        image=img,
                        points=camera_clusters,
                        colors=cluster_vel_colors,
                        sizes=cluster_vel_sizes * img_scale,
                    )
                    
            if not self.plot_radar:
                return img, title
            self.camera_img.set_data(img)
            if alternative_data:
                alt_out["camera_img"] = img
        
        if self.project_radar:
            offsets = np.column_stack((-radar_clusters[:,1], radar_clusters[:,0]))
            if alternative_data:
                alt_out["offsets"] = offsets
                alt_out["facecolors"] = cluster_vel_colors
                alt_out["sizes"] = 2 + cluster_vel_sizes**2
            
            self.scatter_plot.set_offsets(offsets)
            self.scatter_plot.set_facecolors(cluster_vel_colors)
            self.scatter_plot.set_sizes(2 + cluster_vel_sizes**2)

            ymaxlim = 256 * radar_clusters[:,2].mean()
            ymaxlim = np.ceil(ymaxlim / 10) * 10
            #ymaxlim = 110
            self.scatter_axis.set_ylim( 0, ymaxlim )

            if self.look_for_old_radar:
                if "old_radar" in batch:
                    self.scatter_axis2.set_axis_on()
                    old_radar_grid = batch['old_radar'].cpu().numpy().transpose(1, 2, 0)[::-1,::-1] * self.radar_normalisation[None, None, :]
                    old_radar_clusters = point_from_gridentry_vectorized(grid=old_radar_grid, thres=self.threshold_cluster_present)

                    old_cluster_vel = old_radar_clusters[:,5]-old_radar_clusters[:,6]
                    old_cluster_vel -= doppler * np.floor(old_cluster_vel/doppler +1/2)
                    green = np.zeros_like(old_radar_clusters[:,0], dtype=float)

                    offsets = np.column_stack((-old_radar_clusters[:,1], old_radar_clusters[:,0]))
                    self.scatter_plot2.set_offsets(offsets)
                    facecolors = np.stack([
                        np.clip(old_cluster_vel,0,10)/10,               # velocity > 0 -> red dot (yellow if close)
                        green, #np.clip(150-old_radar_clusters[:,0],0,150)/150, # close point  -> green, far away -> black
                        np.clip(old_cluster_vel,-10,0)/-10,             # velocity < 0 -> blue dot (cyan if close)
                    ], axis=1,)
                    self.scatter_plot2.set_facecolors(facecolors)
                    old_sizes = (2 + (2*radar_clusters[:,4]/100 + 3).astype(int) ** 2)
                    self.scatter_plot2.set_sizes( old_sizes )
                    self.scatter_axis2.set_ylim( 0, ymaxlim )
                    if alternative_data:
                        alt_out["old_offsets"] = offsets
                        alt_out["old_facecolors"] = facecolors
                        alt_out["old_sizes"] = old_sizes
                    if "masked_points" in batch:
                        masked_radar_grid = batch['masked_points'].cpu().numpy().transpose(1, 2, 0)[::-1,::-1] * self.radar_normalisation[None, None, :]
                        masked_clusters = point_from_gridentry_vectorized(grid=masked_radar_grid, thres=self.threshold_cluster_present)
                        offsets = np.column_stack((-masked_clusters[:,1], masked_clusters[:,0]))
                        self.scatter_plot_masked.set_offsets(offsets)
                        self.scatter_plot2_masked.set_offsets(offsets)
                        if alternative_data:
                            alt_out["masked_offsets"] = offsets
                else:
                    self.scatter_axis2.set_axis_off()
            
        if self.plot_radar:
            for i in range(len(radar_data)):
                data = self.radar_normalisation[i] * radar_data[i]
                self.radar_imgs[i].set_data(data)

            mask = np.where(radar_data[2] > self.threshold_cluster_present)
            vel = self.radar_normalisation[5] * radar_data[5]
            vel[mask] -= self.radar_normalisation[6] * radar_data[6][mask]
            vel[mask] -= np.floor(1/2+vel[mask]/doppler)*doppler
            self.radar_imgs[7].set_data(vel)

        return self.figure, (title if not alternative_data else alt_out)

    def _on_video_end(self):
        """
        Optional hook that is called after the video creation in
        :py:meth:`create_sample_video` ended. Should be used to clean up the
        environment set up in :py:meth:`_on_video_start`.
        """
        if self.plot_radar:
            del self.radar_cbars
            del self.radar_imgs
            del self.radar_axis
            del self.camera_img
            del self.camera_axis
            del self.figure
        self.figure = None
        self._ready_for_video = False



if __name__ == "__main__":

    import getpass
    from time import perf_counter
    from torch.utils.data import DataLoader

    username = os.environ.get("SLURM_JOB_USER") or getpass.getuser()
    print(f"detected username: '{username}'")
    use_summary = False
    check_for = 0
    out_img_size=(948,1792) #(1896//2, 3584//2)  #(512, 512)
    #out_img_size=(1792, 948)
    undistorted = True
    only_vid = False
    show_grid = False
    project_radar = True

    data_source_id = 7
    source_path = [
        "/p/data1/nxtaim/proprietary/continental/sys100",
        "/p/data1/nxtaim/proprietary/continental/sys100_validation",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250730_200401_12014693/videos/RadarCond_038000_0.0155/2021.05.04_at_14.54.52_radargrids.h5",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250730_200401_12014693/videos/RadarCond_038000_0.0155/2021.06.09_at_11.26.11_radargrids.h5",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250730_182914/videos/RadarCond_043000_0.0155/2021.04.15_at_11.38.16_radargrids.h5",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250730_200401_12014693/videos/RadarCond_038000_0.0155/2021.04.26_at_08.59.57_radargrids.h5",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250909_175001_12394188/videos/RadarCond_034000_0.0248/2021.04.26_at_08.59.57_radargrids.h5",
        "/e/scratch/multiscale-wm/data/aumovio_sys100"
    ][data_source_id]
    save_path = [
        f"{basepath}/data/SYS100_Dataset.csv",
        f"{basepath}/data/SYS100_Validation_Dataset.csv",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250730_200401_12014693/videos/RadarCond_038000_0.0155/2021.05.04_at_14.54.52_summary.csv",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250730_200401_12014693/videos/RadarCond_038000_0.0155/2021.06.09_at_11.26.11_summary.csv",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250730_182914/videos/RadarCond_043000_0.0155/2021.04.15_at_11.38.16_radargrids.csv",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250730_200401_12014693/videos/RadarCond_038000_0.0155/2021.04.26_at_08.59.57_radargrids.csv",
        f"/p/scratch/nxtaim-1/users/{username}/logs/20250909_175001_12394188/videos/RadarCond_034000_0.0248/2021.04.26_at_08.59.57_radargrids.csv",
        "/e/scratch/multiscale-wm/data/aumovio_sys100.csv"
    ][data_source_id]
    video_path = [
        f"/p/scratch/nxtaim-1/users/{username}/sys100",
        f"/p/scratch/nxtaim-1/users/{username}/sys100_validation",
        f"/p/scratch/nxtaim-1/users/{username}/sys100_validation_sample_2021.05.04_at_14.54.52",
        f"/p/scratch/nxtaim-1/users/{username}/sys100_sample_2021.06.09_at_11.26.11",
        f"/p/scratch/nxtaim-1/users/{username}/sys100_training_sample_4files_2021.04.15_at_11.38.16",
        f"/p/scratch/nxtaim-1/users/{username}/sys100_training_sample_2021.04.26_at_08.59.57",
        f"/p/scratch/nxtaim-1/users/{username}/sys100_training_sample_FlowMatching_2021.04.26_at_08.59.57",
        "/e/scratch/multiscale-wm/data/"
    ][data_source_id]
    specific_files = [
        #"/p/data1/nxtaim/proprietary/continental/sys100/2021.04.15_at_11.18.03/",
        #"/p/data1/nxtaim/proprietary/continental/sys100/2021.04.17_at_05.10.53/",
        #"/p/data1/nxtaim/proprietary/continental/sys100/2021.06.04_at_08.35.52/",
        #"/p/data1/nxtaim/proprietary/continental/sys100/2021.06.04_at_08.49.25/",
        #"/p/data1/nxtaim/proprietary/continental/sys100/2021.06.04_at_08.50.44/",
        #"/p/data1/nxtaim/proprietary/continental/sys100/2021.07.07_at_23.42.23/",
        #"/p/data1/nxtaim/proprietary/continental/sys100/2021.07.10_at_00.06.09/", # frame 10663 in EVEN couldn't be loaded
    ]

    specific = False
    if specific_files is not None and len(specific_files) > 0:
        specific = True
        source_path = [
            os.path.join(source_path, p) for p in specific_files
        ]
        use_summary = False
        save_path = f"{save_path[:-4]}_specific.csv"
        video_path = f"{video_path}_specific"

    print(f"surveying files from '{save_path if use_summary else source_path}'")
    
    if only_vid:
        video_path = f"{video_path}_only_vid"
    if show_grid:
        video_path = f"{video_path}_grid"
    if undistorted:
        video_path = f"{video_path}_undistorted"
    if project_radar:
        video_path = f"{video_path}_wClusters"

    simple_dataset = Sys100Dataset(
        source=save_path if use_summary else source_path,
        source_is_summary_csv=use_summary,
        out_img_size=out_img_size,
        subsampling_step=1,
        max_files=None,
        random_subsampling=False,
        extract_video=True,
        careful=True,
        strict=False,
        undistort=undistorted,
        plot_radar=not only_vid,
        overlay_distortion_grid=show_grid,
        project_radar=project_radar,
        look_for_old_radar=True,
    )
    print(
        f"Found {len(simple_dataset)} datapoints when taking every valid entry"
       +f"from each h5 table."
    )
    if not use_summary:
        print(f"saving as '{save_path}'")
        simple_dataset.save(path=save_path)
    
    if False:
        # Benchmarking the speed
        old_max_files = simple_dataset.nbr()
        for maxfiles in [15, 25, 30, 40, 50, 60, 70, 80]:
            simple_dataset.set_parameters(max_files=maxfiles, return_metadata=False, project_radar=False)
            workers = 12
            batch_size = 32
            loader = DataLoader(simple_dataset, num_workers=workers, batch_size=batch_size, shuffle=True, prefetch_factor=20)
            pbar = tqdm(loader, desc=f"Speed test, h5={0:.4f}s, img={0:.4f}, wks={workers}, b={batch_size}, f={simple_dataset.nbr()}")
            h5time = 0
            imgtime = 0
            batchtime = 0
            nbr = 0
            start_time = time.time()
            cum_time = 0
            for batch in pbar:
                end_time = time.time()
                nbr += 1
                batchtime += (end_time - start_time - batchtime)/nbr
                cum_time += end_time - start_time
                start_time = end_time
                h5time  += (batch['h5time'].mean().item()  - h5time)/nbr
                imgtime += (batch['imgtime'].mean().item() - imgtime)/nbr
                pbar.set_description(f"Speed test, h5={h5time:.3f}s, img={imgtime:.3f}, batch={batchtime:.3f}, wks={workers}, b={batch_size}, f={simple_dataset.nbr()}")
                if cum_time > 120:
                    break
        simple_dataset.set_parameters(max_files=old_max_files, return_metadata=False, project_radar=project_radar)


    simple_dataset.create_sample_video(
        vid_save_path=video_path,
        files_frac=1,#0.0025,
        fps=15,
        frames_per_file=max(1000, len(simple_dataset)),
        start=None,
        end=None,
        random=False,
        seed=None,
        max_sample_distance=8,
    )
    
    if check_for > 0:
        for extract in ["cv2"]:
            simple_dataset.set_parameters(extraction_function=extract)
            loader = DataLoader(simple_dataset, num_workers=0, batch_size=1, shuffle=True)
            first = next(iter(loader)) # making sure the dataloader is initialised
            cnt = 0
            t0 = perf_counter()
            for batch in tqdm(loader, desc=f"{check_for}s speed test"):
                #print(batch["jpg"].shape)
                cnt+=1
                duration = perf_counter() - t0
                if duration > check_for:
                    break
            print(f"{extract} Average {cnt/duration:.2} it/s")
