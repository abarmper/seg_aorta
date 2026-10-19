# Sample cases — licence note

The two sample cases in this folder (`K18`, `K19`; CT volume + ground-truth mask each) are
validation cases from the **AVT – Aortic Vessel Tree CTA dataset** (SEG.A 2023 challenge). They are
included only so the demo can be run end to end.

**They are NOT covered by the Apache-2.0 licence of this repository.**

- **CT volumes** (`volumes_full/`): taken from the **KiTS19** challenge data, licensed under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — attribution required,
  **non-commercial use only**, adaptations must be shared under the same licence.
- **Segmentation masks** (`labels_full/`): from the AVT dataset, licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Because the masks belong to the KiTS19 scans, treat each sample case as a whole as
**CC BY-NC-SA 4.0 (non-commercial)**. Files are provided here in NIfTI (`.nii.gz`) format.

## Sources and required attribution

- L. Radl, Y. Jin, A. Pepe, J. Li, C. Gsaxner, F. Zhao, J. Egger. "AVT: Multicenter Aortic Vessel Tree
  CTA Dataset Collection with Ground Truth Segmentation Masks". *Data in Brief* 40, 107801 (2022).
  https://doi.org/10.1016/j.dib.2022.107801 — dataset: https://doi.org/10.6084/m9.figshare.14806362
- N. Heller et al. "The KiTS19 Challenge Data: 300 Kidney Tumor Cases with Clinical Context, CT Semantic
  Segmentations, and Surgical Outcomes". arXiv:1904.00445 (2019). https://github.com/neheller/kits19
- N. Heller et al. "The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT
  imaging: Results of the KiTS19 Challenge". *Medical Image Analysis*, 101821 (2020).
