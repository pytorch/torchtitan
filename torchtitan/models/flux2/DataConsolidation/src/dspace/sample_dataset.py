import os, sys
from typing import Union, Optional
from pathlib import Path

import numpy as np
import h5py
from PIL import Image
import io
import torch
from torchvision.transforms import v2

basepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if not basepath in sys.path:
    sys.path.insert(0, basepath)
from ..multifiledataset import MultiFileDataset



class DSpaceSampleDataset(MultiFileDataset):
    def __init__(
        self,
        source : Union[str, Path, list[Union[str,Path]], list[str], list[Path]],
        out_img_size : tuple[int,int] | None = None,
        out_dtype: torch.dtype = torch.float32,
        transform : v2.Transform | None = None,
        *,
        #source_is_summary_csv : bool = False,
        valid_file_ending : str = ".h5",
        #valid_file_must_contain : str = "",
        #recursive : bool = True,
        #max_files : Optional[int] = None,
        #subsampling_step : Optional[int] = None,
        #target_dataset_size : Optional[int] = None,
        #random_subsampling : bool = True,
        #seed : Optional[int] = None,
        #strict : bool = True,
        #display_stats : bool = False,
        #quiet : bool = False,
        **kwargs,
    ):
        """
        Dataset for DSpace sample data.

        Args:
            out_img_size: If given, resize output images to this size (height, width)
            out_dtype: Output dtype of images
            transform: Optional transform to apply to images. 
                (after ToImage, ToDtype and casting to [0,1] range but before resizing)
        """
        self.metadtype = np.dtype([('width', np.uint32), ('height', np.uint32)])
        super().__init__(
            source=source,
            valid_file_ending=valid_file_ending,
            **kwargs,
        )

        self.outsize = None
        self.outdtype = out_dtype
        transform_list = [
            v2.ToImage(),
            v2.ToDtype(out_dtype),
            v2.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])  
        ]
        if transform is not None:
            transform_list.append(transform)
        if out_img_size is not None:
            assert (
                hasattr(out_img_size, "__len__") and len(out_img_size) == 2
            ), f"size needs to be a tupel, instead got: {out_img_size}"
            resize = v2.Resize((out_img_size[0], out_img_size[1]), antialias=True)
            self.outsize = out_img_size
            transform_list.append(resize)
        self.transform = v2.Compose(transform_list)
    
    def _get_samples_metadata_from_filepath(self, path: str):
        """
        Abstract method, must be implemented by subclasses.

        Investigate the file at `path` and return the number of valid samples 
        and optionally a structured `np.ndarray` with additional metadata.
        For example, the video resolution or framerate for videos. The metadata
        needs to contain only basic numpy data types and no arrays - only scalar
        values.
        
        Parameters
        ----------
        file_path : str
            Path of the file that shall be inspected for the number of valid
            datapoints and any additional metadata

        Returns
        -------
        n_samples : int
            Number of valid samples.
        metadata : ndarray | None
            Structured numpy array with shape () but custom datatype containing
            additional metadata for this file. E.g.

            >>> np.array((1024,720), dtype=[('width', np.uint32), 
            >>>                             ('height',  np.uint32)])
        
        """
        with h5py.File(path, 'r') as h5_file:
            # KeysViewHDF5 ['captions', 'drive_sequence_id', 'filenames', 'image_bytes']
            # all as objcets
            valid = len(h5_file['drive_sequence_id']) # type: ignore
            # maybe extract height/width? Don't know if they are constant as 
            # multiple cameras are recorded in the same file.
            with io.BytesIO(h5_file["image_bytes"][0]) as byte_stream: # type: ignore
                with Image.open(byte_stream) as img:
                    img = img.convert("RGB")
            width, height = img.size
            metadata = np.array( (width,height), dtype=self.metadtype )

            return valid, metadata
        
    def _getitem(
        self, 
        file_path: str,
        sample_idx: int,
        metadata: np.ndarray
    ):
        """
        Abstract method, must be implemented by subclasses.

        Load the i-th valid datapoint in the file given by the path, can make
        use of additional metadata for the specific file computed by 
        :py:meth:`_get_samples_metadata_from_filepath`.

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
        with h5py.File(file_path, 'r') as h5_file:
            with io.BytesIO(h5_file["image_bytes"][sample_idx]) as byte_stream: # type: ignore
                with Image.open(byte_stream) as img:
                    img = img.convert("RGB")
                    assert img.size[0] == metadata['width']
                    assert img.size[1] == metadata['height']
                    img = self.transform(img)
            caption = h5_file["captions"][sample_idx].decode("utf-8") # type: ignore
            drive_sequence_id = h5_file["drive_sequence_id"][sample_idx].decode("utf-8") # type: ignore
            filename = h5_file["filenames"][sample_idx].decode("utf-8") # type: ignore
        return {
            "jpg": img,
            "txt": caption,
            "drive_sequence_id": int(drive_sequence_id),
            "filename": filename,
        }
    
    def _get_image_representation(
        self, 
        file_path: str,
        sample_idx: int,
        metadata: np.ndarray, 
        batch,
    ): # -> Tuple[FFMPEG_VIDEO_WRITER_ACCEPTS, str]:
        """
        Abstract method, must be implemented by subclasses.

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
          -  or an equivalent byte stream
          - A PIL Image object
          - A matplotlib Figure object
        the second element being a string containing text that shall be printed
        over the image.
        """
        img = batch["jpg"].cpu().numpy()
        caption = batch["txt"]
        filename = batch['filename']
        if img.ndim == 4:
            img = img[0]
            caption = caption[0]
            filename = filename[0]
        img = np.ascontiguousarray(127.5 * img.transpose(1,2,0) + 127.5, dtype=np.uint8)
        title = f"File: {filename}\n{caption}"
        return img, title