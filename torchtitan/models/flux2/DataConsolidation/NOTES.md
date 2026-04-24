
### DSpace
- Give write access, or at least fork access rights to the nxtAIM github team
- The dataloader in src/dspace/multiview_dataset.py is not compatible with the
uploaded proprietary sample. Please figure out what the state is
- The extracted Images are supposed to be rather low resolution or did I introduce
  an error in the code?

### Hella
- Saw nice looking implementation in Github
- No sample dataset provided yet to test on it

### IPG
- The dataloader works but is still a bit slower than the rest (about a factor 2)
- Note that a dictionary in a return dictionary 

### Mercedes Benz (MB)
- ~~Provided two dataloaders (ECP and TLD) but only samples for TLD?~~  
  Both samples uploaded now
- Loading the metadata is quite slow, could become a problem in upscaled training on
  the full data when parsing takes away multiple minutes off the training each time
  the dataset is initiated
- ~~The TLD dataloader needs on the proprietary samples about 70 seconds/iteration~~  
  Fixed, was just a small problem with the resizing transform being applied elementwise
- ~~Reading multiple frames can lead to errors if the the first frame of the multiple requested is already th last frame:~~
  ```python
    File "/p/project1/nxtaim-1/neuhoefer1/DataConsolidation/src/mb/traffic_light_dataset.py", line 57, in extract_frames_via_cv2
    raise ValueError(f"Could not read frame {start_frame_idx + idx} from {video_path}")
  ValueError: Could not read frame 200 from /p/data1/nxtaim/proprietary_samples/mbag/tld_samples/Koeln/Koeln4/rec_2015-04-23-14-30-50/cam_ampel_left_image_raw.mp4

    File "/p/project1/nxtaim-1/neuhoefer1/DataConsolidation/src/mb/traffic_light_dataset.py", line 301, in __getitem__
      f"Got error at getting video {idx} total frames {self.frames[idx]}",
                                                      ~~~~~~~~~~~^^^^^
  IndexError: index 401 is out of bounds for axis 0 with size 8
  ```
  Should be fixed now with the replacement of the idx_map. Please check.

### Valeo 
- The explaining files "/p/data1/nxtaim/proprietary_samples/valeo/.where.txt.swp"
  and "[...]/.where_is_it.txt.swp" lack permissions to be read
- "PermissionError: Permission denied (os error 13): ...p/data1/nxtaim/proprietary_samples/valeo/20210915_141538_rec_converted/IMAR/IMAR.parquet (set POLARS_VERBOSE=1 to see full path)"

### AVL
- The dataloader holds all h5 files open in the background. That's only possible
  for a very limited amount of files for each process (as dataloaders are cloned
  over processes) and also restricts the functionality of simultaneous sub_dataloaders
  of other partners.
- seems like the augmentations overwrite the given image transform?
- What is the original resolution in the real dataset? Is the solution in the sample
  dataset reduced this much on purpose? (at least that's what querrying a bigger resolution seems to suggest)