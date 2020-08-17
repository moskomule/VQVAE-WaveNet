# VQVAE-WaveNet
A PyTorch implementation of VQVAE and WaveNet, forked from https://github.com/dhgrs/chainer-VQ-VAE.

The status of this repository is work in progress.

## Requirements

* Python>=3.8
* PyTorch>=1.6.0
* torchaudio>=0.6.0
* homura>=2020.08

e.g.,

```commandline
conda create -n ${PROJECT_NAME} python=3.8
conda activate ${PROJECT_NAME}
conda install -c pytorch pytorch torchaudio cudatoolkit==10.2
pip install git+https://github.com/moskomule/homura@v2020.08
```

## Train and inference

### Training

```
python train.py
```

### Inference (generating)

```
python generate.py -i input_file -o output_file -m saved_model -s speaker
```