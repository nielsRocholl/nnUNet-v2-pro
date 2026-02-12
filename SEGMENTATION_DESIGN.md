# ROI-Prompted nnU-Net Lesion Segmentation — Implementation Instructions

## Goal
Implement a non-interactive 3D lesion segmentation model in nnU-Net v2 that:
- Accepts CT images + **optional point-prompt heatmap** channel.
- Segments **all lesions inside ROIs** around propagated coordinates.
- Treats prompts as **weak priors only** (never suppress lesions).
- Avoids full-volume inference but still finds **new lesions near tracked lesions**.

Do **not** invent new methods. Use **only** proven strategies and code from MangoTree primitives and their referenced repositories.

---

## Mandatory MangoTree Usage (must call these)
Call `mcp__mangotree__get_primitive_implementation` for each ID below.
If the snippet is insufficient, open the GitHub link in the citations and follow that implementation.

1) `nninteractive_early_prompting_channel_concatenation`
Sources:
- **Paper**: [Isensee, F., Rokuss, M., Kramer, L., et al. nnInteractive: Redefining 3D Promptable Segmentation. CVPR 2025.](https://arxiv.org/abs/2503.08373)
- **Code**: [Code](https://github.com/MIC-DKFZ/nnInteractive/blob/master/nnInteractive/trainer/nnInteractiveTrainer.py) (symbol: `nnInteractiveTrainer_stub.build_network_architecture`)
**License**: Apache-2.0 license

2) `lesionlocator_prompt_extraction_from_segmentation`
Sources:
- **Paper**: [Rokuss, M., et al. LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging. CVPR 2025.](https://arxiv.org/abs/2502.20985)
- **Code**: [Code](https://github.com/MIC-DKFZ/LesionLocator/blob/main/lesionlocator/utilities/prompt_handling/prompt_handler.py) (symbol: `get_centroids_from_inst_or_bin_seg, get_bboxes_from_inst_or_bin_seg`)
**License**: GPL-3.0 license

3) `lesionlocator_tracknet_architecture`
Sources:
- **Paper**: [Rokuss, M., et al. LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging. CVPR 2025.](https://arxiv.org/abs/2502.20985)
- **Code**: [Code](https://github.com/MIC-DKFZ/LesionLocator/blob/main/lesionlocator/modules/tracknet.py) (symbol: `TrackNet`)
**License**: GPL-3.0 license

4) `totalsegmentator_multipart_inference`
Sources:
- **Paper**: [Wasserthal, J., et al. TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT images. Radiology: Artificial Intelligence (2023). arXiv:2208.05868](https://arxiv.org/abs/2208.05868)
- **Code**: [Code](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/nnunet.py) (symbol: `nnUNet_predict_image`)
**License**: Apache-2.0 license

5) `nnunet_dice_ce_loss`
Sources:
- **Paper**: [Isensee, F., et al. nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation. arXiv:1809.10486 (2018).](https://arxiv.org/abs/1809.10486)
- **Code**: [Code](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/compound_losses.py) (symbol: `DC_and_CE_loss`)
**License**: Apache-2.0 license

---

## Concrete Model Design

### Inputs
- **Image**: standard nnU-Net preprocessed CT (unchanged).
- **Prompt heatmap**: single channel containing **all points in the ROI**.
  - Implement **early concatenation** exactly as nnInteractive.
  - Prompt encoding must follow the nnInteractive code; do not invent a new encoding.

### ROI Definition (Inference)
- For each propagated coordinate:
  - Build a cubic ROI with **50 mm radius** (100 mm edge length).
  - Convert mm → voxels using the **preprocessed spacing** from `data_properties`.
  - Include **all propagated points that fall inside the ROI** in the prompt heatmap.

### Target / Output
- Binary segmentation of **all lesions inside each ROI**.

---

## Training (single timepoint)

### Prompt synthesis
- Use **LesionLocator prompt extraction** to get lesion centroids from GT masks.
- For each training sample, choose an ROI center and include **all lesion centroids within that ROI**.

### Prompt robustness mix (weak prior only)
Apply these four cases (use these proportions exactly):

- **50%**: lesion-centered ROI, correct prompts only
- **20%**: lesion-centered ROI + **spurious points** (random background points)
- **15%**: lesion-centered ROI with **no prompt** (all zeros)
- **15%**: **negative ROI** (no lesions) with random points; target is all zeros

Prompts never suppress segmentation.

### Loss
- Use **Dice + CE** exactly as nnU-Net v2.

---

## Required Code Changes (nnU-Net v2)

### 1) Network Input Channels
- Add **1 extra channel** for the prompt heatmap.
- Implement early concatenation following nnInteractive.

### 2) Data Loader
- Add a new data loader variant (extend existing loader in
  `/Users/nielsrocholl/Documents/PhD DIAG - Local/Code/nnUNet-v2-pro/nnunetv2/training/dataloading/data_loader.py`).
- It must:
  - Sample ROIs as defined above.
  - Generate prompt heatmap for all points in ROI.
  - Apply the robustness mix exactly.

### 3) Trainer Variant
- Add a new trainer variant under
  `/Users/nielsrocholl/Documents/PhD DIAG - Local/Code/nnUNet-v2-pro/nnunetv2/training/nnUNetTrainer/variants/`
- Use the new dataloader and extra input channel configuration.

### 4) ROI Inference
- Add an inference entrypoint under
  `/Users/nielsrocholl/Documents/PhD DIAG - Local/Code/nnUNet-v2-pro/nnunetv2/inference/`
- It must:
  - Accept a list of propagated points.
  - Build ROIs and run **sliding-window inference** inside each ROI.
  - Merge overlapping logits into a full-volume canvas using **Gaussian weighting** (reuse nnU-Net sliding-window merge and TotalSegmentator multi-part inference logic).

---

## Acceptance Criteria
- Works with or without prompt channel.
- Segments **all lesions in ROI**, not just prompted lesion.
- Robust to wrong or missing prompts.
- No new algorithms: all logic must be based on MangoTree implementations and linked code.
