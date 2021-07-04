# EVA: Entity Visual Alignment

![EVA logo](misc/EVA_logo.png)

Entity Alignment is the task of linking entities with the same real-world identity from different knowledge graphs. EVA is a set of algorithms that leverage images in knowledge graphs for facilitating Entity Alignment.

This repo holds code for reproducing models presented in our paper: *Visual Pivoting for (Unsupervised) Entity Alignment* [\[arxiv\]](https://arxiv.org/pdf/2009.13603.pdf) at AAAI 2021.


## Data

Download the used data (DBP15k, DWY15 along with precomputed features) from [here (dropbox)](https://www.dropbox.com/sh/5jteio17gfzp3xc/AACeXmsMEYts0O5_0Cuva7lPa?dl=0) (1.3GB after unzipping) and place under `data/`. 

[optional] The raw images of entities appeared in DBP15k and DWY15k can be downloaded [here (dropbox)](https://www.dropbox.com/sh/rnvtnjhymbu8wh0/AACONryOmrNvoCkir2R8Dwxha?dl=0) (108GB after unzipping). All images are saved as title-image pairs in dictionaries and can be accessed with the following code:
```python
import pickle
zh_images = pickle.load(open("eva_image_resources/dbp15k/zh_dbp15k_link_img_dict_full.pkl",'rb'))
print(en_images["http://zh.dbpedia.org/resource/香港有線電視"].size)
``:

## Environment
The code is tested with python 3.7 and torch 1.7.0.

## Use EVA
Run the full model on DBP15k:
```console
./run_dbp15k.sh 0 2020 fr_en
```
where `0` specifies the GPU device, `2020` is a random seed and `fr_en` sets the language pair.

Similarly, you can run the full model on DWY15k:
```console
./run_dwy15k.sh 0 2020 1
```
where the first two args are the same as before, the third specifies where using the *normal* (`1`) or *dense* (`2`) split.

To run without iterative learning:
```console
./run_dbp15k_no_il.sh 0 2020 fr_en
./run_dwy15k_no_il.sh 0 2020 1
```

To run the unsupervised setting on DBP15k:
```console
./run_dbp15k_unsup.sh 0 2020 fr_en
```

## Acknowledgement
Our codes are modified from [KECG](https://github.com/THU-KEG/KECG). We appreciate the authors for making KECG open-sourced.

## Citation
```bibtex
@inproceedings{liu2021visual,
  title={Visual Pivoting for (Unsupervised) Entity Alignment},
  author={Liu, Fangyu and Chen, Muhao and Roth, Dan and Collier, Nigel},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4257--4266},
  year={2021}
}
```

## License
EVA is MIT licensed. See the [LICENSE](LICENSE) file for details.
