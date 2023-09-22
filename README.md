# CPCL
This repository will hold the PyTorch implementation of the JBHI'22 paper [All-around real label supervision: Cyclic prototype consistency learning for semi-supervised medical image segmentation](https://ieeexplore.ieee.org/abstract/document/9741294). Note that this is an extended implementation for the LA benchmark.

## Introduction
### Abstract
Semi-supervised learning has substantially advanced medical image segmentation since it alleviates the heavy burden of acquiring costly expert-examined annotations. Especially, the consistency-based approaches have attracted more attention for their superior performance, wherein the real labels are only utilized to supervise their paired images via supervised loss while the unlabeled images are exploited by enforcing the perturbation-based “unsupervised” consistency without explicit guidance from those real labels. However, intuitively, the expert-examined real labels contain more reliable supervision signals. Observing this, we ask an unexplored but interesting question: can we exploit the unlabeled data via explicit real-label supervision for semi-supervised training? To this end, we discard the previous perturbation-based consistency but absorb the essence of non-parametric prototype learning. Based on the prototypical networks, we then propose a novel cyclic prototype consistency learning (CPCL) framework, which is constructed by a labeled-to-unlabeled (L2U) prototypical forward process and an unlabeled-to-labeled (U2L) backward process. Such two processes synergistically enhance the segmentation network by encouraging more discriminative and compact features. In this way, our framework turns previous “unsupervised” consistency into new “supervised” consistency, obtaining the “all-around real label supervision” property of our method. 

### Highlights
- Utilize prototype-based label prediction to achieve "real-label learning" for unlabeled data.


## Requirements
Check requirements.txt.
* Pytorch version >=0.4.1.
* Python == 3.6 

## Usage

1. Clone the repo:
```
cd ./CPCL-SSL
```

2. Data Preparation
Refer to ./data for details


3. Train
```
cd ./code
python train_CPCL_general_3D.py --labeled_num 8 --model vnet_MTPD --gpu 0 
```

4. Test 
```
cd ./code
python test_3D.py --model vnet_MTPD
```


## Citation

If you find this paper useful, please cite as:
```
@article{xu2022all,
  title={All-around real label supervision: Cyclic prototype consistency learning for semi-supervised medical image segmentation},
  author={Xu, Zhe and Wang, Yixin and Lu, Donghuan and Yu, Lequan and Yan, Jiangpeng and Luo, Jie and Ma, Kai and Zheng, Yefeng and Tong, Raymond Kai-yu},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={26},
  number={7},
  pages={3174--3184},
  year={2022},
  publisher={IEEE}
}
```

