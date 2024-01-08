<h1 align="center">
  kpgen-lowres-data-aug
</h1>

<h4 align="center">Data Augmentation for Low-Resource Keyphrase Generation</h4>

<p align="center">
  <a href="https://aclanthology.org/2023.findings-acl.534/"><img src="https://img.shields.io/badge/Findings%20of%20ACL-2023-red"></a>
  <a href="https://aclanthology.org/2023.findings-acl.534.pdf"><img src="https://img.shields.io/badge/Paper-PDF-yellow"></a>
  <a href="res/ACL_slides.pdf"><img src="https://img.shields.io/badge/Slides-PDF-blue"></a>
  <a href="https://github.com/kgarg8/kpgen-lowres-data-aug/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"></a>
  </a>
</p>

# kpgen-lowres-data-aug
Official code for Data Augmentation for Low-Resource Keyphrase Generation (ACL Findings 2023)

## Environment
Please refer `environment.yml`

## Preprocessing
Please refer `preprocess/` folder. Note that we create different versions of the original dataset apriori.

## Train & Test
```
# Train
python train.py

# Train on limited data
python train.py --limit=100

# Load Checkpoint
python train.py --checkpoint=True

# Train for multiple runs after the initial run(s)
python train.py --times=3 --initial_time=1

# Test (assuming that saved weights are present)
python train.py --test=True
```

## Citation
Please consider citing our paper if you find this work useful:

```
@inproceedings{garg-etal-2023-data,
    title = "Data Augmentation for Low-Resource Keyphrase Generation",
    author = "Garg, Krishna  and
      Ray Chowdhury, Jishnu  and
      Caragea, Cornelia",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.534",
    doi = "10.18653/v1/2023.findings-acl.534",
    pages = "8442--8455",
    abstract = "Keyphrase generation is the task of summarizing the contents of any given article into a few salient phrases (or keyphrases). Existing works for the task mostly rely on large-scale annotated datasets, which are not easy to acquire. Very few works address the problem of keyphrase generation in low-resource settings, but they still rely on a lot of additional unlabeled data for pretraining and on automatic methods for pseudo-annotations. In this paper, we present data augmentation strategies specifically to address keyphrase generation in purely resource-constrained domains. We design techniques that use the full text of the articles to improve both present and absent keyphrase generation. We test our approach comprehensively on three datasets and show that the data augmentation strategies consistently improve the state-of-the-art performance. We release our source code at \url{https://github.com/kgarg8/kpgen-lowres-data-aug}.",
}
```

## Questions
Please contact `kgarg8@uic.edu` for any questions related to this work.