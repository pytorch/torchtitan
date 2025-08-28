# Vision Language Model training in `torchtitan`

**under active development**

This folder showcases how to train modern Vision Language Model (vlm) in torchtitan.


## Features:
- Native Aspect Ratio: not limited to square crops.
- Native Resolution: images in a batch can have different sizes, no more image tiles and thumbnails.
- Native Interleaved data: training samples can have variable number of images, interleaved with text at different position. You can train more than just a captioning model.


## Design
Distributed training usually does not play nice with input of varying shapes. To handle a varying number of images and image sizes, we requires two hyperparameters, image batch size `N` and image length `L` (in patches), and pad the actual image patches to this fixed size. 
Then we scatter the patch embeddings to their actual positions in the LLM input tokens.
This result in a very simple and general interface to train modern VLM with interleaved data and native resolution & aspect ratio.
By setting the appropriate dataloader hyperparameters, we can easily reduce the amount of padding tokens.
We leverage FlexAttention to efficiently handle varying number of patches per image.
