# EVA: Entity Visual Alignment

![EVA logo](misc/EVA_logo.png)

Entity Alignment is the task of linking entities with the same real-world identity from different knowledge graphs. EVA is a set of algorithms that leverage images in knowledge graphs for facilitating Entity Alignment.

This repo holds code for reproducing models presented in our paper: **_Visual Pivoting for (Unsupervised) Entity Alignment_** [\[arxiv\]](https://arxiv.org/pdf/2009.13603.pdf)[\[aaai\]](https://ojs.aaai.org/index.php/AAAI/article/view/16550) at AAAI 2021.


## Data

Download the used data (DBP15k, DWY15 along with precomputed features) from [dropbox](https://www.dropbox.com/sh/5jteio17gfzp3xc/AACeXmsMEYts0O5_0Cuva7lPa?dl=0) or [BaiduDisk](https://pan.baidu.com/s/1TnQMzKboMymvutc0hn8Y0Q) (code: dhya) (1.3GB after unzipping) and place under `data/`. 

Original sources of DBP15k and DWY15k:
- [DBP15k](http://ws.nju.edu.cn/jape/)
- [DWY15k](https://github.com/nju-websoft/RSN/blob/master/entity-alignment-full-data.7z)

[optional] The raw images of entities appeared in DBP15k and DWY15k can be downloaded from [dropbox](https://www.dropbox.com/sh/rnvtnjhymbu8wh0/AACONryOmrNvoCkir2R8Dwxha?dl=0) (108GB after unzipping). All images are saved as title-image pairs in dictionaries and can be accessed with the following code:
```python
import pickle
zh_images = pickle.load(open("eva_image_resources/dbp15k/zh_dbp15k_link_img_dict_full.pkl",'rb'))
print(en_images["http://zh.dbpedia.org/resource/香港有線電視"].size)
```

### Dataset Descriptions

We use the DWY15k dataset as an example (files not used in experiments are omitted).

```
data/DWY_data/
├── dwy15k_dense_sf_vec.npy: surface form vectors encoded by fastText (dense split)
├── dwy15k_norm_sf_vec.npy: surface form vectors encoded by fastText (normal split)
├── dbp_wd_15k_V1/: normal split
│   ├── mapping/
│   │   ├── 0_3/: the third split (used across all experiments)
│   │   │   ├── ent_ids_1: mapping between entity names and ids for graph 1
│   │   │   ├── ent_ids_2: mapping between entity names and ids for graph 2
│   │   │   ├── rel_ids_1: mapping between relation names and ids for graph 1
│   │   │   ├── rel_ids_2: mapping between relation names and ids for graph 2
│   │   │   ├── ill_ent_ids: inter-lingual links (specified by ids)
│   │   │   ├── triples_1: a list of tuples in the form of (head, relation, tail) for graph 1 (specified by ids)
│   │   │   ├── triples_2: a list of tuples in the form of (head, relation, tail) for graph 2 (specified by ids)
│   │   │   ├── ...
│   │   ├── ...
│   ├── ...
├── dbp_wd_15k_V2/: dense split
│   ├── ...
data/pkls/
├── dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl: mapping between entity names to image features for DWY15k (normal)
│   ├── ...
```

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
