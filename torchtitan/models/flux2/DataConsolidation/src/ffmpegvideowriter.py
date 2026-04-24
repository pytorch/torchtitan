"""
FFMPEGVideoWriter -- Streamlined FFmpeg video writing via subprocess pipe.

Supports video frame writing from:

- NumPy arrays (HWC, RGB, dtype=uint8)
- PIL Images
- Matplotlib figures
- Raw RGB byte streams

Features:

- On-the-fly frame resizing and text overlays (via PIL)
- Lazy initialization (frame size can be inferred from first input)
- Multiple output formats (MP4, GIF, etc.)
- Configurable quality, compression, and looping options

Example usage, can be tested by executing this file as a script:
    
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> 
    >>> frames = 60
    >>> x = np.linspace(0,4*np.pi,1001)
    >>> f = np.linspace(1,10,frames)
    >>> fig, ax = plt.subplots(1,1, dpi=100, layout="tight")
    >>> sin_data = ax.plot([], [], color='yellow', label='sin')[0]
    >>> cos_data = ax.plot([], [], color='blue', label='cos')[0]
    >>> ax.set_xlim(-0.1, 12.6)
    >>> ax.set_ylim(-1.1, 1.1)
    >>> 
    >>> with FFMPEGVideoWriter(video_path="sin_test", fps=10) as videowriter:
    >>>     for i in range(frames):
    >>>         sin_data.set_data(x, np.sin(f[i]*x))
    >>>         cos_data.set_data(x, np.cos(f[i]*x))
    >>>         videowriter.write(fig, title=f"Freq: {f[i]:.2f} Hz")
    >>> 
    >>> plt.close(fig)

"""


import os
import numpy as np
from numpy.typing import NDArray
import subprocess
from typing import Optional, Union, Tuple, cast, BinaryIO
from cv2 import resize, INTER_LINEAR

try:
    from PIL import Image, ImageFont, ImageDraw
    try:
        from PIL.Image import Resampling
        LANCZOS = Resampling.LANCZOS
    except ImportError: # for PIL < 9.1.0
        LANCZOS = Image.LANCZOS # type: ignore[attr-defined]
    PIL_PRESENT = True
except ImportError:
    PIL_PRESENT = False

try:
    import matplotlib.figure as mpl_figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MLP_PRESENT = True
except ImportError:
    MLP_PRESENT = False

try:
    subprocess.run(["ffmpeg", "-version"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL,
                   check=True)
    FFMPEG_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    FFMPEG_AVAILABLE = False

#FFMPEG_VIDEO_WRITER_ACCEPTS = Union[NDArray[np.uint8], Image.Image, 
#                                    mpl_figure.Figure, bytes]


class FFMPEGVideoWriter(object):
    """
    A class for writing video files using FFmpeg by streaming raw frames to a subprocess.

    Supports frames as:
      - NumPy arrays (H x W x 3, dtype=np.uint8)
      - PIL Images
      - Matplotlib figures
      - Raw RGB bytes

    Optional text overlays can be added using PIL. The video resolution can either
    be inferred from the first frame or explicitly set during initialization.
    """

    @property
    def width(self) -> int:
        """The width all incoming frames will be resized to and the width of the
           resulting video in pixels. If not specified during initialisation will
           be inferred when the first image is being written"""
        if self._width is None:
            raise RuntimeError("Width is not set. It will be set after the first write() call.")
        return self._width
    @property
    def height(self) -> int:
        """The height all incoming frames will be resized to and the height of the
           resulting video in pixels. If not specified during initialisation will
           be inferred when the first image is being written"""
        if self._height is None:
            raise RuntimeError("Height is not set. It will be set after the first write() call.")
        return self._height
    @property
    def framerate(self) -> float:
        """The frames per second of the output video"""
        return self._fps
    @property
    def video_path(self) -> str:
        """The path to the output video file"""
        return self._output_file

    @property
    def _ffmpeg_pipe(self) -> subprocess.Popen[bytes]:
        if self.__ffmpeg_pipe is None:
            raise RuntimeError("._start_subprocess() has not been called yet and "+
                               "the ffmpeg process has not been started yet.")
        return self.__ffmpeg_pipe
        
    
    def __init__(
        self,
        video_path: Union[str, os.PathLike],
        *,
        file_extension: str = "mp4",
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: float = 30.0,
        quality: Optional[int] = 23,
        lossless: bool = False,
        repeat: Union[bool, int] = False,
        buffered_images: int = 4,
        font_size: int = 24,
        min_keyframe_interval: Optional[int] = None,
        max_keyframe_interval:  Optional[int] = None,
        scene_cut_threshold:  Optional[int] = None,
        codec: Optional[str] = 'AVC',
        preset: Optional[str] = None,
        target_bitrate: Optional[Union[str, int]] = None,
        max_bitrate: Optional[Union[str, int]] = None,
        bitrate_smoothing_buffer: Optional[Union[str, int]] = None,
    ):
        """
        A class for writing video files using FFmpeg via a subprocess pipe.

        Parameters
        ----------
        video_path : str
            Path (excluding suffix) to save the output video file.
        file_extension : str, default="mp4"
            Output video file extension: "mp4", "gif", etc.
        width : int, optional
            Width of the output video frames. If not provided, inferred from the
            first call of :py:meth:`write`. If provided but not height as well
            the first frame will be rescaled while preserving the aspect ratio
            to determine the height of all frames.
        height : int, optional
            Height of the video frames. If not provided, inferred from the first
            call of :py:meth:`write`. If provided but not width as well
            the first frame will be rescaled while preserving the aspect ratio
            to determine the width of all frames.
        fps : float, default=30
            Frames per second of the assumed input stream and the output video.
        quality : int, default=23
            Compression quality level (ignored if `lossless=True`).
            For AVC codec, 0 is the best possible achievable quality, 16 is 
            basically humanly indistinguishable to lossless compression, 20 is 
            good quality and 51 worst possible quality.
            For HEVC codec due to higher compression rates higher quality values
            (i.e. lower quality) are equivalent to the same AVC quality values 
            in terms of visual quality. If quality should not default to 23 but
            bitrates and compression quality be guided by other parameters, 
            set this to None.
        lossless : bool, default=False
            If True, enables lossless compression (`-qp 0`).
        repeat : Union[bool, int], default=False
            Repeat mode for the video. ``False`` = no repeat, ``True`` = infinite loop,
            ``int`` = specific number of repeats. Only works for gifs, or when
            the entire video already exists, e.g. not the application purpose of
            this class.
        buffered_images : int, default=4
            Approximate number of images buffered in FFmpeg's stdin.
        font_size : int, default=24
            Font size for title text overlays. Only impactful if the Helvetica
            font is present, otherwise will use the default font whose size
            cannot be changed.
        min_keyframe_interval : int | None, optional
            Minimum interval between keyframes (I-frames) in frames (`-keyint_min`).
            Defines the shortest allowed GOP (Group of Pictures) size. Larger values 
            prevent keyframes from occurring too close together, effectively limiting 
            overly frequent keyframes during rapid motion or scene changes. Smaller 
            values allow more flexibility for the encoder to insert keyframes closer 
            to each other if needed.
        max_keyframe_interval : int | None, optional
            Maximum interval between keyframes (`-g`), i.e., GOP size in frames.
            This sets the upper bound on how far apart keyframes can be. Larger values 
            increase compression efficiency by spacing keyframes farther apart but 
            reduce random access speed. Smaller values improve seekability at the cost 
            of larger file size.
        scene_cut_threshold : int | None, optional
            Sensitivity threshold for scene change detection (`-sc_threshold`).
            Lower values make the encoder more likely to insert a keyframe at scene cuts, 
            allowing variable GOP lengths. A value of 0 disables scene cut detection (fixed GOP). 
            Values typically range 0-100; the default is often 40. Higher values mean fewer scene cuts.
        codec : str, default 'AVC'
            Which codec should be used to write the video, can be
            'libx264', 'H.264', 'H264' or 'AVC' for AVC codec, or 
            'libx265', 'H.265', 'H265' or 'HEVC' for HEVC codec.
            HEVC has better compression rates but takes a bit longer to compute,
            but requires licenses (multiple) for commercial use.
            AVC is compatible with most (especially older) systems.
            (maybe also add AV1? royalty free equivalent to HEVC but takes a bit
            longer)
        preset: str
            Some preset configurations balancing coding time and compression
            efficiency. E.g. 'ultrafast' will write the video as fast as possible
            but have large file sizes, 'veryslow' will be rather slow while 
            writing but have small file sizes.
            Possible values include: ['ultrafast', 'superfast', 'veryfast',
            'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
            The default is 'medium' if not provided.
        target_bitrate : (int, str), optional
            The target/expected bitrate of the video, e.g. how many bits per 
            second the video can use, passed as `-b:v`.
            Higher bitrate results in better quality but larger files and more
            demanding bandwidth for transmission.
            BUT: This should not be set together with the quality and it needs 
            to be scaled to frame rate and resolution if those change.
            Can be given as a string such as "4M", "800k", or as an integer
            representing bits per second. If None, FFmpeg defaults to its
            internal rate control. Is ignored when losslessly compressing.
        max_bitrate : (int, str), optional
            Maximum allowed instantaneous bitrate, passed as `-maxrate`.
            This places a hard upper bound on how high the bitrate may spike.
            Useful for streaming or for producing files with predictable sizes.
            Same format rules as `target_bitrate`.
            Needs either `quality` or `target_bitrate` to be set, lossless 
            compression to not be active, and works only well if 
            `bitrate_smoothing_buffer` is also set to around 1-4x this value.
        bitrate_smoothing_buffer : (int, str), optional
            Rate-control buffer size (VBV buffer), passed as `-bufsize`.
            Determines how much bitrate fluctuation FFmpeg may use while 
            staying under `maxrate`. Larger values allow higher quality 
            but give less strict bandwidth control. Common values are
            1-4x `max_bitrate`. Same format rules as above.


        Raises
        ------
        EnvironmentError
            If ffmpeg is not found in the system.
        ValueError
            If dimensions cannot be determined and `image0` is not provided.
        """
        
        if not FFMPEG_AVAILABLE:
            raise EnvironmentError("ffmpeg is not available on this system.")
        
        assert fps > 0, "FPS must be a positive number"
        assert quality is None or 0 <= quality and quality <= 51, \
               "quality parameter must be between 0 and 51"
        self._file_extension = file_extension
        self._width = width
        self._height = height
        self._fps = fps
        self._quality = quality
        self._lossless = lossless
        self._repeat = repeat
        self._buffered_images = buffered_images
        self._pipebuffsize = None
        self._min_keyframe_interval = min_keyframe_interval
        self._max_keyframe_interval = max_keyframe_interval
        self._scene_cut_threshold = scene_cut_threshold
        self._codec = codec
        self._preset = preset
        self._target_bitrate = target_bitrate
        self._max_bitrate = max_bitrate
        self._bitrate_smoothing_buffer = bitrate_smoothing_buffer
        self._written_frames = 0

        self._output_file = self._build_output_filename( path=str(video_path) )
        if not os.path.exists(os.path.dirname(self.video_path)):
            print(f"WARNING! The directory '{os.path.dirname(self.video_path)}'"
                  +" doesn't exist yet. It will be created manually, but check"
                  +" that it is in the correct place")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)

        self._pending_ffmpeg_start = (width is None or height is None)
        self.__ffmpeg_pipe = None
        if not self._pending_ffmpeg_start:
            # If width and height are already known, start the ffmpeg subprocess 
            # immediately. Otherwise, wait for the first .write() call to infer them.
            self._start_subprocess()

        helvetica_path = os.path.join(os.path.dirname(__file__), "Helvetica.ttf")
        if PIL_PRESENT and os.path.exists(helvetica_path):
            self._font = ImageFont.truetype(helvetica_path, size=font_size)
        else:
            self._font = None

    
    def _build_output_filename(
        self, 
        path: str,
    ) -> str:
        """Build the output filename from the base path, quality, preset, and extension.

        Parameters
        ----------
        path : str
            Base path (without extension) for the output file.

        Returns
        -------
        str
            Complete output filename with quality/preset suffix and extension.
        """
        suffix = ""
        if self._lossless:
            suffix += "_lossless"
        elif self._quality is not None:
            suffix += f"_qt{self._quality:d}"
        if self._preset is not None:
            suffix += f"_{self._preset}"
        if self._file_extension != "gif":
            return f"{path}{suffix}.{self._file_extension}"
        else:
            return f"{path}.gif"

    def _start_subprocess(self) -> None:
        """
        Start the ffmpeg subprocess with a pipe for raw video input.

        This is called either immediately if frame dimensions are known at init,
        or lazily after the first call to `.write()` when frame dimensions can be inferred.

        Raises
        ------
        RuntimeError
            If the ffmpeg subprocess fails to open.
        """
        ffmpeg_command = self._build_ffmpeg_command()
        # use the closest even number of bits larger than needed to fit _buffered_images
        # into the buffer
        self._pipebuffsize = 2*int(np.log2(self._buffered_images*3*self.width*self.height)/2+1)
        
        self.__ffmpeg_pipe = subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            bufsize=2**self._pipebuffsize
        )
        self._pending_ffmpeg_start = False

        if not self.is_open():
            raise RuntimeError("Failed to open ffmpeg subprocess with command: " +
                            " ".join(ffmpeg_command))
    
    def _build_ffmpeg_command(self) -> list[str]:
        """
        Construct the full ffmpeg command used to launch the subprocess.

        Returns
        -------
        list of str
            Argument list suitable for `subprocess.Popen`.
        """
        input_args = [
            '-f', 'rawvideo',                    # input data format is raw data 
            '-vcodec', 'rawvideo',               # input data is not encoded
            '-pix_fmt', 'rgb24',                 # input images are rgb values with 3 8-bit values
            '-s', f'{self.width}x{self.height}', # input frame sizes
            '-r', str(self.framerate),           # input frame rate per second
            '-an', '-sn',                        # no audio or subtitles
            '-i', 'pipe:'                        # input frames come from the stdin
        ]

        loop_args = []
        if self._file_extension == "gif":
            output_args = ['-pix_fmt', 'yuv420p']
            if isinstance(self._repeat, bool):
                if self._repeat:
                    output_args.extend(['-loop', '-1'])
            elif isinstance(self._repeat, int): # for some reason, False is also an int
                output_args.extend(['-loop', str(self._repeat)])
        else:
            if self._repeat is not False:
                print("\nWARNING! repeat keyword doesn't work with raw video input and non-gif output.")
            output_args = []
            
            if self._codec is None or self._codec in ['libx264', 'H.264', 'H264', 'AVC']:
                output_args.extend(['-vcodec', 'libx264'])
            elif self._codec in ['libx265', 'H.265', 'H265', 'HEVC']:
                output_args.extend(['-vcodec', 'libx265'])
            else:
                print(f"\nWARNING: unknown interpret codec: '{self._codec}' "
                      +"trying to use it as is\n")
                output_args.extend(['-vcodec', self._codec])
            output_args.extend(['-pix_fmt', 'yuv420p'])
            if self._preset is not None:
                output_args.extend(['-preset', self._preset])
            if self._lossless:
                output_args.extend(['-qp', '0'])
            elif self._quality is not None:
                output_args.extend(['-crf', str(self._quality)])
            # Add keyframe options if provided
            if self._min_keyframe_interval is not None:
                output_args.extend(['-keyint_min', str(self._min_keyframe_interval)])
            if self._max_keyframe_interval is not None:
                output_args.extend(['-g', str(self._max_keyframe_interval)])
            if self._scene_cut_threshold is not None:
                output_args.extend(['-sc_threshold', str(self._scene_cut_threshold)])
            # Add bitrate information
            if self._target_bitrate is not None:
                output_args.extend(['-b:v', self._target_bitrate])
            if self._max_bitrate is not None:
                if self._target_bitrate is None and self._quality is None:
                    print("\nWARNING! max_bitrate is set but neither target"
                          +" bitrate nor quality is defined")
                output_args.extend(['-maxrate', self._max_bitrate])
                if self._bitrate_smoothing_buffer is not None:
                    output_args.extend(['-bufsize', self._bitrate_smoothing_buffer])

        return [
            'ffmpeg', 
            '-loglevel', 'error', # only show errors, no info or warnings
            '-y',                 # overwrite output file if it exists
            *loop_args, 
            *input_args,
            *output_args,
            self.video_path
        ]

    def write(
        self, 
        img, # : Union[NDArray[np.uint8], Image.Image, mpl_figure.Figure, bytes]
        title: Optional[str] = None,
        left_margin: int = 24,
        top_margin: int = 24,
    ) -> None:
        """
        Write a single frame to the video.

        Parameters
        ----------
        img : Union[NDArray[np.uint8], PIL.Image.Image, matplotlib.figure.Figure, bytes]
            Frame to write. Can be raw bytes, NumPy array, PIL image, or matplotlib figure.
            If a byte stream, then it must be in RGB format of total length 
            :py:attr:`width` ``*`` :py:attr:`height` ``* 3``.
        title : str, optional
            Optional title text to overlay on the frame.
        left_margin : int, default=24
            X-position of the title text (pixels).
        top_margin : int, default=24
            Y-position of the title text (pixels).

        Raises
        ------
        RuntimeError
            If the video writer has already been closed.
        ValueError
            If image type is unsupported, title overlay fails, or ``img`` is a
            bytestream when it's the first call to :py:meth:`write` with unknown
            :py:attr:`width` or :py:attr:`height` attributes during initialisation.
        TypeError
            If input type is unrecognized.
        """
        if self._pending_ffmpeg_start:
            if isinstance(img, np.ndarray):
                self._height, self._width = img.shape[:2]
            elif PIL_PRESENT and isinstance(img, Image.Image):
                self._width, self._height = img.size
            elif MLP_PRESENT and isinstance(img, mpl_figure.Figure):
                fig = img
                fig.canvas.draw()
                self._width, self._height = fig.canvas.get_width_height()
            elif isinstance(img, bytes):
                raise ValueError("As the FFMPEGVideoWriter was not initialised "+
                                 "with known width and height information, "+
                                 "the first input image in .write() must not "+
                                 "be a byte stream, but a NumPy array, PIL "+
                                 "Image, or matplotlib Figure!")
            else:
                raise ValueError(f"Unknown image type {type(img)} as input of .write()")

            self._start_subprocess()
        
        if not self.is_open():
            raise RuntimeError("Video writer has already been closed")
        # we know that the standard input is still open, but mypy/pylance does not
        ffmpeg_in = cast(BinaryIO, self._ffmpeg_pipe.stdin)
        
        if MLP_PRESENT and isinstance(img, mpl_figure.Figure):
            if not isinstance(img.canvas, FigureCanvasAgg):
                img.set_canvas(FigureCanvasAgg(img))
            # Telling interpreters that we know img.canvas is now a FigureCanvasAgg
            canvas = cast(FigureCanvasAgg, img.canvas)
            canvas.draw()
            w, h = canvas.get_width_height()
            img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        
        # resizing if necessary
        if PIL_PRESENT and isinstance(img, Image.Image):
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height), LANCZOS)
        elif isinstance(img, np.ndarray):
            if img.shape[:2] != (self.height, self.width):
                img = np.asarray(resize(img, (self.width, self.height), 
                                        interpolation=INTER_LINEAR), 
                                 dtype=np.uint8)
        elif isinstance(img, bytes) and len(img) != self.width*self.height*3:
            raise ValueError(f"expected {self.width}*{self.height}*3 = "+
                             f"{self.width*self.height*3} bytes, instead got "+
                             f"{len(img)} and unable to figure out original shape")

        # writing the title
        if PIL_PRESENT and title:
            if isinstance(img, bytes):
                img = np.frombuffer(img, dtype=np.uint8).reshape(self.height, self.width, 3)
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if isinstance(img, Image.Image):
                draw = ImageDraw.Draw(img)
                draw.text((left_margin, top_margin), title, fill="black",
                          stroke_width=1, stroke_fill="white", font=self._font)
                img = np.array(img)
            else:
                raise ValueError(f"Unsupported image type {type(img)} "+
                                 f"for title: '{title}'\n" +("" if MLP_PRESENT 
                                    else "Note that matplotlib is not present!")
                )
        
        # transforming into bytes and writing to the pipe
        if PIL_PRESENT and isinstance(img, Image.Image):
            img = np.array(img)
        if isinstance(img, np.ndarray):
            ffmpeg_in.write(img.tobytes())
        elif isinstance(img, bytes):
            ffmpeg_in.write(img)
        else:
            raise TypeError(f"Unsupported input type: {type(img)}. Expected a "
                            f"np.ndarray, PIL.Image, bytes, or matplotlib.figure.Figure.")
        
        self._written_frames += 1

    def is_open(self) -> bool:
        """
        Check if the FFmpeg pipe is still open.

        Returns
        -------
        bool
            True if the pipe is open, False otherwise.
        """
        return (self.__ffmpeg_pipe is not None
                and self.__ffmpeg_pipe.stdin is not None 
                and not self.__ffmpeg_pipe.stdin.closed)

    def close(self):
        """
        Finalize and close the video file.
        """
        self._pending_ffmpeg_start = False
        if self.is_open():
            # we know that the standard input is still open, but mypy/pylance does not
            ffmpeg_in = cast(BinaryIO, self._ffmpeg_pipe.stdin)
            ffmpeg_in.flush()
            ffmpeg_in.close()
            print(f"Waiting for ffmpeg to finish encoding '{self.video_path}'...")
            self._ffmpeg_pipe.wait()

    def __enter__(self):
        """
        Enable use of `with` context.

        Returns
        -------
        FFMPEGVideoWriter
            The current instance.
        """
        return self
    
    def __len__(self):
        """Returns the number of frames already written to the video"""
        return self._written_frames

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit context and close the video properly.
        """
        self.close()

    def __del__(self):
        """Destructor ensuring the FFmpeg process is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Avoid exceptions in destructor

    def __repr__(self):
        return (f"<FFMPEGVideoWriter output_file={self.video_path} "+
                f"fps={self.framerate}, size=({self._width}x{self._height}), "+
                f"frames={self._written_frames}>")
    


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    frames = 60
    x = np.linspace(0,4*np.pi,1001)
    f = np.linspace(1,10,frames)
    fig, ax = plt.subplots(1,1, dpi=100, layout="tight")
    sin_data = ax.plot([], [], color='yellow', label='sin')[0]
    cos_data = ax.plot([], [], color='blue', label='cos')[0]
    ax.set_xlim(-0.1, 12.6)
    ax.set_ylim(-1.1, 1.1)

    with FFMPEGVideoWriter(video_path="sin_test", fps=10) as videowriter:
        for i in range(frames):
            sin_data.set_data(x, np.sin(f[i]*x))
            cos_data.set_data(x, np.cos(f[i]*x))
            videowriter.write(fig, title=f"Freq: {f[i]:.2f} Hz")
    
    plt.close(fig)