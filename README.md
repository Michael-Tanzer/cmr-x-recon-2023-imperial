# CMRxRecon challenge - Imperial College London Team - 2023

## Method
We utilized Restormer (https://github.com/swz30/Restormer) as the backbone for our single-channel network and employed both Restormer and MoDL (https://github.com/hkaggarwal/modl) as backbones for our multi-channel network. 

Overview of Latent Transformer.
<div align="center">
    <img scr = "ModelFigure.png" width = "900"/>
</div>

We introduced a 'Latent Transformer' featuring an encoder-decoder architecture with shared blocks, designed to model time-dependent relationships. 

Each block of the Latent Transformer harnesses multi-layer and multi-head self-attention mechanisms to update the latent code through a weighted linear combination of itself and the latent codes from other time-frames, executed in a pixel-wise fashion. 

## Citation
```bibtex
    @inproceedings{tanzer2023t1,
    title={T1/T2 Relaxation Temporal Modelling from Accelerated Acquisitions Using a Latent Transformer},
    author={T{\"a}nzer, Michael and Wang, Fanwen and Qiao, Mengyun and Bai, Wenjia and Rueckert, Daniel and Yang, Guang and Nielles-Vallespin, Sonia},
    booktitle={International Workshop on Statistical Atlases and Computational Models of the Heart},
    pages={293--302},
    year={2023},
    organization={Springer}
    }
```