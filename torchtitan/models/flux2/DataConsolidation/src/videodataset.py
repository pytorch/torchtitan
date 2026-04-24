"""
A simple dataset class designed to enable easy access to random frames from a 
large collection of different video files.
:py:class:`VideoDataset` Can be used directly or taken as reference for custom
implementations.

This file can also be run as a script to create a sample video from all the
collected videos and to benchmark the speed of the different frame extraction
methods, such as via FFMPEG, OpenCV, torchvision or torchcodec. In our tests,
OpenCV (via :py:meth:`extract_frame_via_cv2`) was by far the fastest by a factor
of at least 2x.
"""

# TODO: ALso Enable Returning clips of videos instead of single frames

import os, sys
from typing import Callable, Tuple, Union, Optional, cast
from pathlib import Path
import subprocess

import numpy as np
import cv2
import torch
from fractions import Fraction
from torchvision.transforms import v2
from tqdm.auto import tqdm

try:
    from torchvision.io import read_video
    TORCHVISION_READ_VIDEO_AVAILABLE = True
except ImportError:
    read_video = None  # type: ignore[assignment]
    TORCHVISION_READ_VIDEO_AVAILABLE = False

try:
    from torchcodec.decoders import VideoDecoder, SimpleVideoDecoder # type: ignore
    TORCHCODEC_AVAILABLE = True
    import sys
    TORCHCODEC_VERSION = sys.modules[VideoDecoder.__module__.split('.')[0]].__version__
    TORCHCODEC_HASAPPROXIMATE = (TORCHCODEC_VERSION >= '0.2.0')
except ImportError:
    TORCHCODEC_AVAILABLE = False
    TORCHCODEC_VERSION = '-1'
    TORCHCODEC_HASAPPROXIMATE = False

try:
    subprocess.run(["ffmpeg", "-version"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL,
                   check=True)
    FFMPEG_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    FFMPEG_AVAILABLE = False

try:
    from av import open as av_open, time_base as av_time_base   # type: ignore
    from av.container import InputContainer                     # type: ignore
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

try:
    from decord import VideoReader as decord_VideoReader, cpu as decord_cpu  # type: ignore
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False



from .multifiledataset import MultiFileDataset



class VideoDataset(MultiFileDataset):
    """
    A dataset that wraps around multiple video files and provides random access
    to individual frames quickly.    
    """

    
    _file_metadatas : np.ndarray
    """Structured array storing per-file metadata returned by the 
       subclass hook :py:meth:`_get_samples_metadata_from_filepath`. 
       An ``N`` dimensional array of custom dtype 
       
        >>> np.dtype([('fps', np.float32), ('width', np.uint32), ('height', np.uint32)])

       containing the frame rate per second, frame width and height 
       of each video file in the dataset.
       
    """

    transform : Optional[v2.Transform]
    """Optional torchvision transform to apply to each frame."""

    metadtype : np.dtype
    """Structured dtype describing per-video metadata layout."""

    extract_function : Callable[[str,int,np.ndarray], torch.Tensor]
    """
    A function that extracts a frame from a video file.
        
        Parameters
        ----------
        video_path : str
            Path to the video file.
        frame_idx : int
            Frame index to extract.
        metadata : np.ndarray
            0-dimensional metainformation for the video as structured numpy array
            containing 'fps', 'height' and 'width'.
        
        Returns
        -------
        torch.Tensor
            The extracted RGB frame as a torch Tensor of shape (3, height, width).
    """
    
    _Conti_Continuous_Adaptation : bool
    """Specific to the ContiDriveDataset, as it contains videos with
       black bars that have to be removed manually."""
    _last_path : str
    """from which video the last image was returned"""
    _last_frame_idx : int
    """which frame was returned in the last video"""
    _lastmetadata : Optional[np.ndarray]
    """additional metadata of the last video from which an image was extracted"""
    outsize : Optional[tuple[int,int]]
    """If the dataloader resizes the image to a given output resolution and
       if so, then which height x width resolution it resizes to"""

    def __init__(
        self,
        source : Union[str, Path, list[str], list[Path], list[Union[str,Path]]],
        out_img_size : Optional[tuple[int,int]] = None,
        transform : Optional[v2.Transform] = None,
        extract_function : Union[Callable[[str,int,np.ndarray], torch.Tensor], str] = "cv2",
        *,
        source_is_summary_csv : bool = False,
        valid_file_ending : str = ".mp4",
        valid_file_must_contain : str = "",
        recursive : bool = True,
        max_files : Optional[int] = None,
        subsampling_step : Optional[int] = None,
        target_dataset_size : Optional[int] = None,
        random_subsampling : bool = True,
        seed : Optional[int] = None,
        strict : bool = True,
        display_stats : bool = False,
        quiet : bool = False,
        Conti_Continuous_Adaptation : bool = False,
    ):
        """
        Constructs a new VideoDataset, see :py:class:`MultiFileDataset` for the
        documentation of the inherited parameters.

        Parameters
        ----------
        source : str | Path | list[str | Path]
            Root directory that contains the data **or** path to a metadata CSV
            produced by :py:meth:`save` **or** a list of paths that should all
            be surveyed for valid source files.
        out_img_size : tuple[int,int]], optional
            If the images should get resized into a given height x width resolution.
        transform : v2.Transform, optional
            Any other transformation that should get applied to the images before
            potential resizing.
        extract_function: Callable[[str,int,np.ndarray],np.ndarray] | str
            The function that is used to extract the frames from the video files.
            This function expects the path to the video file, the index of the frame
            to extract and additional metadatas for the video in a numpy structured
            array, including 'fps', 'height' and 'width'.
            Alternatively, could be one of the following strings for existing
            extraction functions:

            - ``ffmpeg``: uses FFMPEG via subprocess. 
              In tests achieved about 1.52 frames per second
            - ``cv2``: uses the cv2 packages videocapture interfaces.
              In tests achieved about 3.05 frames per second
            - ``torchvision``: Uses the native torchvision readvideo function.
              In tests achieved about 1.10 frames per second
            - ``torchcodec``: Uses torchcodecs video interface.
              In tests achieved about 0.49 frames per second
            - ``torchcodec_approximate``: Uses torchcodecs video interface with
              approximate search mode. Could not be tested as it requires
              torchcodec version 0.2.0 or higher incompatible with pytorch 2.5.1
            - ``pyav`` uses pyav's interface to ffmpeg
            - ``decord`` uses the optimised framework decord

        Conti_Continuous_Adaptation : bool
            Specific to the ContiDriveDataset, as it contains videos with black
            bars that have to be removed by this adaptation.
            Leave at false for all other datasets.
        """
        # need to define the variables needed in _get_samples_metadata_from_filepath
        # before they are called by the constructer
        self.metadtype = np.dtype([('fps', np.float32),
                                   ('width', np.uint32),
                                   ('height', np.uint32)])
        super().__init__(
            source = source,
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
            display_stats=display_stats,
            quiet=quiet,
        )
        if callable(extract_function):
            self.extract_function = extract_function
        elif extract_function=="ffmpeg":
            if not FFMPEG_AVAILABLE:
                raise RuntimeError("FFMPEG is not available on the system")
            self.extract_function = extract_frame_via_pipe
        elif extract_function=="cv2":
            self.extract_function = extract_frame_via_cv2
        elif extract_function=="torchvision":
            if not TORCHVISION_READ_VIDEO_AVAILABLE:
                raise RuntimeError("torchvision.io.read_video is not available in this torchvision build")
            self.extract_function = extract_frame_via_torchvision
        elif extract_function=="torchcodec":
            if not TORCHCODEC_AVAILABLE:
                raise RuntimeError("Torchcodec is not installed")
            self.extract_function = extract_frame_via_torchcodec
        elif extract_function=="simpletorchcodec":
            if not TORCHCODEC_AVAILABLE:
                raise RuntimeError("Torchcodec is not installed")
            self.extract_function = extract_frame_via_simpletorchcodec
        elif extract_function=="torchcodec_approximate":
            if not TORCHCODEC_HASAPPROXIMATE:
                raise RuntimeError(f"Torchcodec version '{TORCHCODEC_VERSION}' "+
                    f"is older than '0.2.0' necessary for approximate seek mode")
            self.extract_function = (
                lambda video_path, frame_idx, metadata:
                    extract_frame_via_torchcodec(video_path=video_path,
                                                 frame_idx=frame_idx,
                                                 metadata=metadata,
                                                 approximate=True)
                )
        elif extract_function=="pyav":
            if not PYAV_AVAILABLE:
                raise RuntimeError("pyav is not installed")
            self.extract_function = extract_frame_via_pyav
        elif extract_function=="decord":
            if not DECORD_AVAILABLE:
                raise RuntimeError("decord is not installed")
            self.extract_function = extract_frame_via_decord
        else:
            raise ValueError(f"Could not parse extract function: {extract_function}"+
                             f"needs to be a function or one of ['ffmpeg', "+
                             f"'cv2', 'torchvision', 'torchcodec', "+
                             f"'torchcodec_approximate', 'pyav', 'decord']")


        self.transform = transform
        self.outsize = None
        if out_img_size is not None:
            assert (
                hasattr(out_img_size, "__len__") and len(out_img_size) == 2
            ), f"size needs to be a tupel, instead got: {out_img_size}"
            resize = v2.Resize((out_img_size[0], out_img_size[1]), antialias=True)
            self.outsize = out_img_size
            if self.transform is None:
                self.transform = resize
            else:
                self.transform = v2.Compose([resize, self.transform])
        
        # this is beacuse we have some uncleaned data in the datsets
        self._Conti_Continuous_Adaptation = Conti_Continuous_Adaptation
        self._last_path = ""
        self._last_frame_idx = -1
        self._lastmetadata = None


    def _get_samples_metadata_from_filepath(self, path: Union[str, Path]):
        """
        Examine ``path`` and return the number of valid samples *plus* an
        optional single-record structured ``np.ndarray`` with the FPS rate, the
        width of the images in the video and their height as metadata.

        Returns
        -------
        n_samples : int
        metadata : structured ndarray 
            metadata containing frame rate per second, frame width and
            height of the video, such as
            
            >>> np.array([(30.0, 1920, 1080)], dtype=[('fps', np.float32), 
            >>>                                       ('width', np.uint32), 
            >>>                                       ('height', np.uint32)])
        
        """
        
        # cv2 is about 2.5x faster than ffprobe for metadata extraction for some reason
        cap = cv2.VideoCapture(filename=str(path))
        if not cap.isOpened():
            raise ValueError(f"Video file '{path}' could not be opened!")
        n_samples = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        metadata = np.array(
            (cap.get(cv2.CAP_PROP_FPS),
             cap.get(cv2.CAP_PROP_FRAME_WIDTH),
             cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            dtype=self.metadtype
        )
        cap.release()
        return n_samples, metadata

    def _getitem(self, 
                 file_path: str,
                 sample_idx: int,
                 metadata: np.ndarray):
        """
        Loads the requested frame from the specified video file.

        Parameters
        ----------
        file_path : str
            Path of the file that contains the sample.
        sample_idx : int
            Zero-based index *inside* that file.
        metadata : structured ndarray
            Companion metadata produced by
            :py:meth:`_get_samples_metadata_from_filepath`.
        """
        self._last_path = file_path
        self._last_frame_idx = sample_idx
        self._lastmetadata = metadata
        img = self.extract_function(
            file_path,
            sample_idx,
            metadata,
        )
        # img is now a Tensor of shape channels x height x width
        
        if (self._Conti_Continuous_Adaptation 
            and metadata['width'] == 832 
            and metadata['height'] in [482, 630]
        ):
            img = img[:, :, :-156]

        aspect_ratio = img.shape[2] / img.shape[1]
        aspect_fraction = Fraction(aspect_ratio).limit_denominator(16)

        if self.transform is not None:
            img = self.transform(img)
        img = img.float() / 127.5 - 1
        img = img.squeeze(0)

        return {'jpg': img, 
                'txt': f'A photo from a centered front-facing camera', 
                'aspect': torch.tensor((aspect_fraction.numerator, aspect_fraction.denominator), dtype=torch.float32),
                'fovs': torch.tensor((-1, -1), dtype=torch.float32),
                'fps':metadata['fps']
                }

    def _get_image_representation(
        self, 
        file_path: str,
        sample_idx: int,
        metadata: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """
        Returns the extracted image from :py:meth:`_getitem` transformed back as
        a numpy array of uint8s and a string of the videoname where it came from
        and the corresponding timestamp of the frame in the video to be collected
        in a sample video for inspecting the dataset
        """
        batch = self._getitem(file_path=file_path, 
                              sample_idx=sample_idx, 
                              metadata=metadata)
        img = batch['jpg'].numpy().transpose(1, 2, 0)
        img = (127.5 * img + 127.5).astype(np.uint8)
        assert img.ndim == 3, "forgotten to unsqueeze the batch dim"

        timestamp = sample_idx / metadata['fps']
        mins = int(timestamp // 60)
        title = (f"{sample_idx:>5} = {mins:>02}:{timestamp-60*mins:05.2f} - "+
                 f"{os.path.basename(file_path)}\ntxt: \"{batch['txt']}\"\n"+
                 f"aspect: '{batch['aspect']}', fovs: {batch['fovs']}")

        return img, title











def extract_frame_via_pipe(video_path, frame_idx, metadata):
    """
    Extract a frame from a video at a specific timestamp using FFmpeg via a pipe.
    On tests could only extract about 1.52 random frames per second.

    Parameters
    ----------
    video_path : str
        Path to the video file from which to extract from.
    frame_idx : int
        0-based index of the frame to extract from this video.
    metadata : np.ndarray
        Additional metainformation for this process presented as a structured
        0-dimensional numpy array. Must contain Frames per second information
    
    Returns
    -------
    torch.Tensor
        The extracted RGB frame as a torch Tensor of shape (3, height, width).
    """
    cmd = [
        "ffmpeg",
        "-ss",       str(frame_idx / metadata['fps']),  # Seek to the timestamp
        "-i",        video_path,  # Input video
        "-frames:v", "1",  # Extract only one frame
        "-f",        "rawvideo",  # avoid encoding
        "-pix_fmt",  "rgb24",
        "-",                   # output the pimage on the stdandard output (pipe)
    ]

    # Run FFmpeg with a pipe
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # wait for ffmpeg to finish reading the frame and read the raw image data from stdout
    stdout_data, stderr_data = process.communicate()

    # Check for errors
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {stderr_data.decode()}")

    # Decode the raw image bytes into a NumPy array
    image = np.frombuffer(stdout_data, np.uint8).reshape(
        (metadata['height'], metadata['width'], 3)
    )
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return tensor


def extract_frame_via_cv2(video_path, frame_idx, metadata=None, order="CHW"):
    """
    Extracts a frame from a video using cv2 making use of efficient jump-ahead
    to the closest keyframe. In tests achieves about 3.05 images/second.
    
    Parameters
    ----------
    video_path : str
        Path to the video file from which to extract from.
    frame_idx : int
        0-based index of the frame to extract from this video.
    metadata : np.ndarray, optional
        Additional metainformation for this process presented as a structured
        0-dimensional numpy array (ignored here, included for interface compatibility).
    
    Returns
    -------
    torch.Tensor
        The extracted RGB frame as a torch Tensor of shape (3, height, width).
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        if cap is None:
            cap_reason = "video capture is None"
        else:
            cap_reason = f"has {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames"
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}\n"
                        +f"{cap_reason}")
    cap.release()
    # cv2 returns BGR so we need to flip the channel dimension to get RGB
    _order = str(order).lower()
    if _order == "hwc":
        tensor = torch.from_numpy(np.ascontiguousarray(frame[:,:,::-1]))
    elif _order == "chw":
        tensor = torch.from_numpy(np.ascontiguousarray(frame[:,:,::-1].transpose(2, 0, 1)))
    else:
        raise ValueError(f"Unkown order '{order}' is supposed to be 'CHW' or 'HWC'")
    return tensor


def extract_frame_via_torchvision(video_path, frame_idx, metadata):
    if not TORCHVISION_READ_VIDEO_AVAILABLE:
        raise RuntimeError("torchvision.io.read_video is not available in this torchvision build")
    """
    Extract a specific frame from a video using torchvision.io.
    In tests could extract about 1.10 frames per second.
    
    Parameters
    ----------
    video_path : str
        Path to the video file from which to extract from.
    frame_idx : int
        0-based index of the frame to extract from this video.
    metadata : np.ndarray, optional
        Additional metainformation (ignored here, included for interface compatibility).
    
    Returns
    -------
    torch.Tensor
        The extracted RGB frame as a torch Tensor of shape (3, height, width).
    """
    # Read video frames (tries to only extract the frame at the wanted timestep)
    video, _, info = read_video(
        video_path,
        start_pts=frame_idx / metadata["fps"],
        end_pts=(frame_idx + 1) / metadata["fps"],
        pts_unit="sec",
        output_format= "TCHW",
    )

    # Convert to NumPy array and return the desired frame
    return video[0].contiguous()


def extract_frame_via_torchcodec(
    video_path, 
    frame_idx, 
    metadata=None, 
    approximate:bool = False
) -> torch.Tensor:
    """
    Extract a specific frame from a video using torchcodec.
    In tests could extract about 0.49 frames per second.
    
    Parameters
    ----------
    video_path : str
        Path to the video file from which to extract from.
    frame_idx : int
        0-based index of the frame to extract from this video.
    metadata : np.ndarray, optional
        Additional metainformation (ignored here, included for interface compatibility).
    approximate : bool
        Whether to use the approximate seek mode (only available in torchcodec >= 0.2.0).
    
    Returns
    -------
    torch.Tensor
        The extracted RGB frame as a torch Tensor of shape (3, height, width).
    """
    # Read video frames (tries to only extract the frame at the wanted timestep)
    if approximate:
        try:
            decoder = VideoDecoder(video_path, seek_mode="approximate") # type: ignore
        except TypeError as e:
            print(f"The seek_mode keyword was only introduced in torchcodec "+
                  f"'0.2.0' you're using {TORCHCODEC_VERSION}")
            raise e
    else:
        decoder = VideoDecoder(video_path)
    return decoder.get_frame_at(frame_idx).data

def extract_frame_via_simpletorchcodec(
    video_path, 
    frame_idx, 
    metadata=None, 
    approximate:bool = False
) -> torch.Tensor:
    """
    Extract a specific frame from a video using torchcodec's SimpleVideoDecoder.
    In tests could extract about ? frames per second.
    
    Parameters
    ----------
    video_path : str
        Path to the video file from which to extract from.
    frame_idx : int
        0-based index of the frame to extract from this video.
    metadata : np.ndarray, optional
        Additional metainformation (ignored here, included for interface compatibility).
    approximate : bool
        Whether to use the approximate seek mode (only available in torchcodec >= 0.2.0).
    
    Returns
    -------
    torch.Tensor
        The extracted RGB frame as a torch Tensor of shape (3, height, width).
    """
    # Read video frames (tries to only extract the frame at the wanted timestep)
    if approximate:
        try:
            decoder = VideoDecoder(video_path, seek_mode="approximate") # type: ignore
        except TypeError as e:
            print(f"The seek_mode keyword was only introduced in torchcodec "+
                  f"'0.2.0' you're using {TORCHCODEC_VERSION}")
            raise e
    else:
        video = SimpleVideoDecoder(video_path)
    return video[frame_idx]

def extract_frame_via_pyav(
    video_path, 
    frame_idx, 
    metadata, 
) -> torch.Tensor:
    """
    Extract a specific frame from a video using PyAV.
    In tests could extract about 1.4 frames per second.
    
    Parameters
    ----------
    video_path : str
        Path to the video file from which to extract from.
    frame_idx : int
        0-based index of the frame to extract from this video.
    metadata : np.ndarray
        Additional metainformation including fps information.
    
    Returns
    -------
    torch.Tensor
        The extracted RGB frame as a torch Tensor of shape (3, height, width).
    """
    container = cast(InputContainer, av_open(file=video_path, mode='r'))
    stream = container.streams.video[0]
    time_base = stream.time_base or 1/av_time_base # how many seconds a pts is 
    frame_rate = float(stream.average_rate or metadata['fps'])
    target_pts = int(frame_idx / frame_rate / time_base)
    
    # Seek close to the target frame for efficiency
    container.seek(target_pts, stream=stream)

    for i, frame in enumerate(container.decode(video=0)):
        if frame.pts is None:
            continue
        if frame.pts >= target_pts:
            # Convert to RGB numpy array
            img = frame.to_ndarray(format='rgb24')
            return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    
    raise IndexError(f"Frame index {frame_idx} not found in '{video_path}'.")

def extract_frame_via_decord(
    video_path, 
    frame_idx, 
    metadata=None, 
) -> torch.Tensor:
    """
    Extract a specific frame from a video using PyAV.
    In tests could extract about 0.34 frames per second.
    
    Parameters
    ----------
    video_path : str
        Path to the video file from which to extract from.
    frame_idx : int
        0-based index of the frame to extract from this video.
    metadata : np.ndarray, optional
        Additional metainformation (ignored here, included for interface compatibility).
    
    Returns
    -------
    torch.Tensor
        The extracted RGB frame as a torch Tensor of shape (3, height, width).
    """
    vr = decord_VideoReader(video_path, ctx=decord_cpu(0))
    return torch.from_numpy(np.ascontiguousarray(vr[frame_idx].asnumpy().transpose(2, 0, 1)))






if __name__ == "__main__":

    from time import perf_counter
    from torch.utils.data import DataLoader
    import getpass

    data_source_id = 3

    username = os.environ.get("SLURM_JOB_USER") or getpass.getuser()
    print(f"detected username: '{username}'")
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    source_path = [
        "/p/data1/nxtaim/proprietary/continental/conti_drive_dataset/test_sequences",
        "/p/data1/nxtaim/proprietary/continental/conti_drive_dataset/training_sequences",
        "/p/data1/nxtaim/proprietary/continental/conti_drive_dataset/validation_sequences",
        "/p/data1/nxtaim/proprietary/continental/sys100",
        "/p/data1/nxtaim/proprietary/continental/sys100_preprocessed",
    ][data_source_id]
    save_path = [
        f"{basepath}/data/ContiDriveDataset_TestSequences.csv",
        f"{basepath}/data/ContiDriveDataset_TrainingSequences.csv",
        f"{basepath}/data/ContiDriveDataset_ValidationSequences.csv",
        f"{basepath}/data/SYS100_VideoDataset.csv",
        f"{basepath}/data/SYS100_VideoDataset_Preprocessed.csv",
    ][data_source_id]
    video_path = [
        f"/p/scratch/nxtaim-1/users/{username}/contidrive_test_vid",
        f"/p/scratch/nxtaim-1/users/{username}/contidrive_training_vid",
        f"/p/scratch/nxtaim-1/users/{username}/contidrive_validation_vid",
        f"/p/scratch/nxtaim-1/users/{username}/sys100_only_vid",
        f"/p/scratch/nxtaim-1/users/{username}/sys100_preprocessed_only_vid",
    ][data_source_id]

    use_summary = True
    extract_function="cv2"
    check_for = 150
    out_img_size=(512, 512)
    specific_files = [

    ]

    print(f"surveying files from '{save_path if use_summary else source_path}'")

    specific = False
    if specific_files is not None and len(specific_files) > 0:
        specific = True
        source_path = [
            os.path.join(source_path, p) for p in specific_files
        ]
        use_summary = False
        save_path = f"{save_path[:-4]}_specific.csv"
        video_path = f"{video_path}_specific"

    use_summary = use_summary and os.path.exists(save_path)
    simple_dataset = VideoDataset(
        source=save_path if use_summary else source_path,
        extract_function=extract_function,
        source_is_summary_csv=use_summary,
        out_img_size=out_img_size,
        subsampling_step=1,
        max_files=None,
        random_subsampling=False,
        Conti_Continuous_Adaptation=True,
        strict=False,
    )
    print(f"Found {len(simple_dataset)} frames.")
    if not use_summary:
        print(f"saving as '{save_path}'")
        simple_dataset.save(path=save_path)

    sizes = np.unique(simple_dataset._file_metadatas)
    print("Found the following configurations of fps x width x height:\n", sizes)

    assert False

    print("to video path: ", video_path)
    simple_dataset.create_sample_video(
        vid_save_path=f"{video_path}_{extract_function}",
        files_frac=0.1,
        fps=5,
        frames_per_file=1,
        start=None,
        end=None,
        random=True,
        seed=None
    )

    print(f"speed test for how fast frames can be randomly read in {check_for} seconds")
    for extract_method in [
        #"decord",
        "pyav",
        "ffmpeg",
        "cv2",
        #"torchvision",
        "simpletorchcodec"
        "torchcodec",
        #"torchcodec_approximate"
    ]:
        simple_dataset = VideoDataset(
            source=save_path,
            extract_function=extract_method,
            source_is_summary_csv=True,
            out_img_size=out_img_size,
            subsampling_step=1,
            max_files=None,
            random_subsampling=False,
            Conti_Continuous_Adaptation=True,
        )
        loader = DataLoader(simple_dataset, num_workers=0, batch_size=1, shuffle=True)
        cnt = 0
        t0 = perf_counter()
        for batch in tqdm(loader, desc=f"{extract_method}"):
            cnt+=1
            p_time = perf_counter() - t0 
            if p_time >= check_for:
                break
        print(f"Average {cnt/p_time:.2} it/s")

    # On login nodes with 2000 old ContiDrive videos:

    #speed test for how fast frames can be randomly read in 300 seconds
    #ffmpeg:   0%|        | 460/59418990 [05:03<10889:52:18,  1.52it/s]
    #Average 1.5 it/s
    #cv2:   0%|            | 925/59418990 [05:03<5411:26:47,  3.05it/s]
    #Average 3.1 it/s
    #torchvision:   0%|   | 332/59418990 [05:01<14974:44:33,  1.10it/s]
    #Average 1.1 it/s
    #torchcodec:   0%|    | 148/59418990 [05:02<33719:38:19,  2.04s/it]
    #Average 0.5 it/s