# SAMGA

Official PyTorch implementation of **Subject-Aware Multi-Granularity Alignment for Zero-Shot EEG-to-Image Retrieval**.

[Paper PDF](./2604.17782v1.pdf) | [arXiv:2604.17782](https://arxiv.org/abs/2604.17782)

SAMGA aligns visually evoked EEG signals with pretrained visual representations for zero-shot image retrieval. Instead of using a single fixed visual target, SAMGA constructs a subject-aware multi-granularity target from multiple intermediate visual layers and optimizes a coarse-to-fine shared embedding space.

<p align="center">
  <img src="./fig/fig1.png" alt="SAMGA Framework" width="95%">
</p>

## Main Contributions

- **Subject-aware multi-granularity alignment.** SAMGA explicitly addresses subject-dependent granularity mismatch in zero-shot EEG-to-image concept retrieval.
- **Adaptive visual target construction.** A subject-aware router aggregates multiple intermediate visual representations, providing visual supervision that better matches the multi-scale information preserved in EEG.
- **Coarse-to-fine cross-modal alignment.** The coarse stage stabilizes shared semantic geometry and reduces subject-induced distribution shift; the fine stage improves instance-level retrieval discrimination.
- **Strong THINGS-EEG results.** SAMGA reports **91.3% Top-1 / 98.8% Top-5** in the intra-subject setting and **34.4% Top-1 / 64.8% Top-5** in the inter-subject setting.

## Benchmark Results

| Dataset | Setting | Top-1 | Top-5 |
| --- | --- | ---: | ---: |
| THINGS-EEG | Intra-subject | 91.3 | 98.8 |
| THINGS-EEG | Inter-subject | 31.4 | 63.1 |
updated results using the final epoch

## Highlights

- THINGS-EEG preprocessing from raw `.npy` recordings to model-ready tensors.
- Optional THINGS-MEG preprocessing from preprocessed MNE epoch files.
- Intra- and inter-subject training with `train.py`.
- Subject-aware multi-layer routing over visual features such as layers `20 24 28 32 36`.
- Per-subject result aggregation with `compute_avg_results.py`.

## Repository Structure

```text
.
|-- train.py                   # intra- and inter-subject training
|-- preprocess_eeg.py          # THINGS-EEG preprocessing
|-- preprocess_meg.py          # optional THINGS-MEG preprocessing
|-- compute_avg_results.py     # merge per-subject result.csv files
|-- intra.sh                   # batch launcher for intra-subject runs
|-- inter.sh                   # batch launcher for inter-subject leave-one-subject-out runs
|-- requirements.txt
|-- fig/fig1.png               # framework figure
`-- module/
    |-- dataset.py             # EEG/image/text feature datasets
    |-- loss.py                # contrastive and MMD losses
    |-- projector.py           # projection heads and shared encoder
    |-- view_fusion.py         # visual feature fusion modules
    |-- eeg_augmentation.py    # EEG augmentations
    |-- image_augmentation.py  # image feature augmentation helpers
    |-- util.py                # retrieval metrics and JSON utilities
    `-- eeg_encoder/           # EEG encoders, including ATM-style encoder blocks
```



## Data Preparation

For reproduction, place the prepared THINGS-EEG files under `./data/things_eeg`. The provided `intra.sh` and `inter.sh` scripts already use this layout:

```text
data/things_eeg/
|-- preprocessed_eeg/
|   |-- info.json
|   |-- sub-01/
|   |   |-- train.npy   # [1654, 10, 4, 63, 250], float32
|   |   `-- test.npy    # [200, 1, 80, 63, 250], float32
|   `-- sub-02/ ... sub-10/
|-- image_feature/
|   |-- internvit_multilevel_20_24_28_32_36/
|   |   |-- image_train_layer20.npy   # [1654, 10, 3200], float16
|   |   |-- image_train_layer24.npy
|   |   |-- image_train_layer28.npy
|   |   |-- image_train_layer32.npy
|   |   |-- image_train_layer36.npy
|   |   |-- image_test_layer20.npy    # [200, 1, 3200], float16
|   |   |-- image_test_layer24.npy
|   |   |-- image_test_layer28.npy
|   |   |-- image_test_layer32.npy
|   |   `-- image_test_layer36.npy

`-- image_set/
    |-- train_images/
    `-- test_images/
```

The main experiments use `image_feature/internvit_multilevel_20_24_28_32_36` with visual layers `20 24 28 32 36`. Text features are optional; the provided launchers set `TEXT_FEATURE_DIR=""` to disable them.


## Common Pitfalls

- The multi-layer router requires files named exactly like `image_train_layer20.npy` and `image_test_layer20.npy` for every layer in `--layer_ids`.
- `--text_feature_dir ""` is the intended way to disable text features.
- `preprocess_eeg.py` asserts that exactly one of `--zscore` and `--mvnn` is enabled.
- Full launchers assume local GPU IDs and data paths; check the variables at the top of `intra.sh` and `inter.sh` before running.
- Training skips an existing completed run with the same `--output_name` suffix to avoid overwriting `result.csv`.
- Do not commit raw EEG/MEG data, generated `.npy` feature caches, TensorBoard logs, or `.pth` checkpoints.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{jiang2026samga,
  title={Subject-Aware Multi-Granularity Alignment for Zero-Shot EEG-to-Image Retrieval},
  author={Jiang, Lin and She, Qingshan and Xu, Jiale and Xu, Haiqi and Wu, Duanpo and Kuang, Zhenzhong},
  journal={arXiv preprint arXiv:2604.17782},
  year={2026}
}
```
