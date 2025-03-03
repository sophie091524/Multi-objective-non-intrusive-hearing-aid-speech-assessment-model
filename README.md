# Multi-Objective Non-Intrusive Hearing Aid Speech Assessment Model

This repository contains the official implementation of our paper:  
**[Multi-Objective Non-Intrusive Hearing Aid Speech Assessment Model](https://pubs.aip.org/asa/jasa/article/156/5/3574/3322471)**.

## Data Preparation

Please prepare the input data in a CSV file following the format below:

### CSV File Format:
The CSV file should contain the following columns:
| ref            | data            | HASQI | HASPI | HL                   | HLType |
|---------------|---------------|-------|----------|----------------------|--------|
| clean.wav | noisy.wav | 0.913 | 0.968    | [65 65 40 25 5 0]    | rising |

### Notes:
- **`ref`**: Path to the clean reference audio.
- **`data`**: Path to the processed version of the reference audio.
- **`HASQI`**, **`HASPI`**: The corresponding speech quality and intelligibility assessment metrics.
- **`HL`**: Hearing loss profile.
- **`HLType`**: Type of hearing loss.

Ensure your dataset is structured correctly before running experiments.

## Usage

Run the following command to train and test the model:

```
CUDA_VISIBLE_DEVICES=0 python main.py
```

### Notes:
Details of the hyperparameters are listed in config.yaml.

### Pre-Trained Model

Please download the model weights from [here](https://drive.google.com/file/d/1ntxOGY1cWqyxdeL-Kq37dgyczx1DTiOc/view?usp=sharing) and create a folder named `save_model`. Then, move the downloaded weight file into the folder.

## Citation

Please cite the following paper if you find this code useful in your research:

```
@article{chiang2024multi,
  title={Multi-objective non-intrusive hearing-aid speech assessment model},
  author={Chiang, Hsin-Tien and Fu, Szu-Wei and Wang, Hsin-Min and Tsao, Yu and Hansen, John HL},
  journal={The Journal of the Acoustical Society of America},
  volume={156},
  number={5},
  pages={3574--3587},
  year={2024},
  publisher={AIP Publishing}
}
```

Please cite the following paper if you use the following pretrained SSL model.
```
@article{chen2022wavlm,
  title={Wavlm: Large-scale self-supervised pre-training for full stack speech processing},
  author={Chen, Sanyuan and Wang, Chengyi and Chen, Zhengyang and Wu, Yu and Liu, Shujie and Chen, Zhuo and Li, Jinyu and Kanda, Naoyuki and Yoshioka, Takuya and Xiao, Xiong and others},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={16},
  number={6},
  pages={1505--1518},
  year={2022},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


