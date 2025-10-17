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

<img width="1398" height="840" alt="Screenshot 2025-08-21 at 16 21 57" src="https://github.com/user-attachments/assets/63fcbbc1-c587-4a63-8246-411cb72f5789" />

- After `tok_embedding`, we obtain tokens of shape `BxS`.
- After `encoder`, we obtain visual tokens of shape `NxL`.
- We extract the valid visual tokens only
- Then scatter those tokens to their actual positions in the LLM input tokens.


This results in a very simple and general interface to train modern VLM with interleaved data and native resolution & aspect ratio:
- Depending on data mixtures, we can set dataloader's hyperparameters `N, L` to have minimal empty image padding (in batch dimension).
- We use modern Pytorch features like FlexAttention and torch.compile to efficient efficiently handle variable sequence length.
- Interface nicely with TP, PP, etc


## Implementation

### Dataloader
This approach requires the dataloader to handle the following aspect:
- [x] Interleave the correct precise numbers of image tokens in the inputs token based on encoder's patch size and input images' size
- [x] Convert images/videos to 1D sequence of patches:
  - `rearrange(pixels, 'n (t pt) (h ph) (w pw) c -> n (t h w) (pt p pw c)', pt=temporal_ps, ph=patch_size, pw=patch_size)`
  - Pad all image patches sequence to a fixed length and return `pixel_values.shape == [N, L, D]`
- [x] Return a `grid_thw.shape == [N, L, 3]` to keep track of the location indices of each patches in the images. Padding image can be tracked in the same tensors with values `-1`.
- [x] LLM Sample / Document Packing.
- [x] Captioning dataset: CC12M
- [x] Interleaved dataset: Obelics



### Model
We also need a pretrained vision encoder with support for native resolution and aspect ratio. There is relatively few Vision Encoder that have this capability up until recently, including Siglip2, AimV2, and most recently DINOv3.
- [ ] Currently we support Siglip2 encoder using Positional Embedding interpolation approach.
    - [x] Base modelling code.
    - [ ] Weights conversion and loading from HF.
- [x] FSDP for both Encoder and Decoder
- [x] Context Parallel for LLM only, since we will use FlexAttention for Encoder.
- [ ] FlexAttention for with different seq len per image.
- [ ] Compile for Encoder + Deocoder
- [ ] Tensor Parallel
- [ ] Pipeline Parallel
