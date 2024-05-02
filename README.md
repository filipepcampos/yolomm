## You Only Look Once for Multimodal Multitasking

Note: This work builds upon [YOLOP](https://github.com/hustvl/YOLOP), please do check out that original repo.

### The Illustration of YOLOMM

<img src='pictures/yolomm.png' width='750'>

The changes, compared to YOLOP, are highlighted in blue. We add an extra input datum, LIDAR information and a new task, point-cloud segmentation.

## Requirements

This codebase has been developed with python version 3.7, PyTorch 1.7+ and torchvision 0.8+:
```
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
```

See `requirements.txt` for additional dependencies and version requirements.
```
pip install -r requirements.txt
```


## Training

You can set the training configuration in the `./lib/config/default.py`. (Including: the loading of preliminary model, loss, data augmentation, optimizer, warm-up and cosine annealing, auto-anchor, training epochs, batch_size).

After that, execute the scripts present in the `tools` directory.

## Publication

This work was published on CIARP 2023: [YOLOMM – You Only Look Once for Multi-modal Multi-tasking](https://link.springer.com/chapter/10.1007/978-3-031-49018-7_40).
```bibtex
@inproceedings{campos2023yolomm,
  title={{YOLOMM}--You Only Look Once for Multi-modal Multi-tasking},
  author={Campos, Filipe and Cerqueira, Francisco Gon{\c{c}}alves and Cruz, Ricardo PM and Cardoso, Jaime S},
  booktitle={Iberoamerican Congress on Pattern Recognition},
  pages={564--574},
  year={2023},
  organization={Springer}
}
```
