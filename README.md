# 15663-project
Course project of [CMU 15-663](http://graphics.cs.cmu.edu/courses/15-463/)

## Goal
In this project, we aim to implement a learning-based method for recovering images captured in the dark. Using a Unet-like architecture, we try to train our model to perform the image processing pipeline in an end ot end fashion. 

## environment
* Python 3.6.0
* rawpy
* PyTorch 1.0

## Training
```
python train_Sony.py --gpu 0
```

## Processing test images
```
python test_Sony.py --model ./model.pl --gpu 0
```

## Calculate metrics
```
python metrics.py --imgdir ./eval
```

## Benchmarks

|       | PSNR   | SSIM  |
|:------|:------:|:-----:|
| NLM   | 16.212 | 0.225 |
| LSID  | 28.575 | 0.815 |
| Ours  | 28.625 | 0.818 |


## Report
[pdf](https://drive.google.com/open?id=1A5Mr7G16Fw131A0hHq6molA9CZbYK5IZ)
## License
MIT license