# ESResNeXt-fbsp
## Learning Robust Time-Frequency Transformation of Audio

This repository contains implementation of the models described in the paper [arXiv:2104.11587](https://arxiv.org/abs/2104.11587) (accepted for IJCNN 2021).
This work is an extension of our previous work [ESResNet: Environmental Sound Classification Based on Visual Domain Models](https://github.com/AndreyGuzhov/ESResNet).

### Abstract
Environmental Sound Classification (ESC) is a rapidly evolving field that recently demonstrated the advantages of application of visual domain techniques to the audio-related tasks. Previous studies indicate that the domain-specific modification of cross-domain approaches show a promise in pushing the whole area of ESC forward.

In this paper, we present a new time-frequency transformation layer that is based on complex frequency B-spline (fbsp) wavelets. Being used with a high-performance audio classification model, the proposed fbsp-layer provides an accuracy improvement over the previously used Short-Time Fourier Transform (STFT) on standard datasets. We also investigate the influence of different pre-training strategies, including the joint use of two large-scale datasets for weight initialization: ImageNet and AudioSet. Our proposed model out-performs other approaches by achieving accuracies of 95.20% on the ESC-50 and 89.14% on the UrbanSound8K datasets.

Additionally, we assess the increase of model robustness against additive white Gaussian noise and reduction of an effective sample rate introduced by the proposed layer and demonstrate that the fbsp-layer improves the model’s ability to withstand signal perturbations, in comparison to STFT-based training. For the sake of reproducibility, our code is made available.

### Downloading Pre-Trained Weights

The pre-trained model can be downloaded from the [releases](https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases).

    wget https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt

### How to Run the Model

The required Python version is >= 3.7.

#### ESResNeXt

##### On the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset
    python main.py --config protocols/esc50/esresnextfbsp-esc50-ptinas-cv1.json --Dataset.args.root /path/to/ESC50

##### On the [UrbanSound8K](https://urbansounddataset.weebly.com/) dataset
    python main.py --config protocols/us8k/esresnextfbsp-us8k-ptinas-cv1.json --Dataset.args.root /path/to/UrbanSound8K

### Cite Us

```
@misc{guzhov2021esresnextfbsp,
      title={ESResNe(X)t-fbsp: Learning Robust Time-Frequency Transformation of Audio}, 
      author={Andrey Guzhov and Federico Raue and Jörn Hees and Andreas Dengel},
      year={2021},
      eprint={2104.11587},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
