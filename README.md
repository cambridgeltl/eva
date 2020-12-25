# EVA: Entity Visual Alignment

![EVA logo](./eva_logo_banner.png)

This repo holds code for reproducing models presented in our paper: *Visual Pivoting for (Unsupervised) Entity Alignment* [\[arxiv\]](https://arxiv.org/pdf/2009.13603.pdf) at AAAI 2021.

## Citation
```bibtex
@inproceedings{liu2021eva,
	title={Visual Pivoting for (Unsupervised) Entity Alignment,
	author={Liu, Fangyu and Chen, Muhao and Roth, Dan and Collier, Nigel},
	booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	year={2021}
}
```

## Data

Download the used data (DBP15k, DWY15 along with precomputed features) from [dropbox](https://www.dropbox.com/sh/5jteio17gfzp3xc/AACeXmsMEYts0O5_0Cuva7lPa?dl=0) and place under `data/`.

The raw images are to be uploaded soon...

## Use EVA
Run the full model on DBP15k:
```bash
./run_dbp15k.sh 0 2020 ja_en
```
where `0` specifies the GPU device, `2020` is a random seed and `ja_en` sets the language pair.

Similarly, you can run the full model on DWY15k:
```bash
./run_dwy15k.sh 0 2020 1
```
where the first two args are the same as before, the third specifies where using the *normal* (`1`) or *dense* (`2`) split.

To run without iterative learning:
```bash
./run_dbp15k_no_il.sh 0 2020 zh_en
./run_dwy15k_no_il.sh 0 2020 1
```

To run the unsupervised setting on DBP15k:
```bash
./run_dbp15k_unsup.sh 0 2020 zh_en
```
