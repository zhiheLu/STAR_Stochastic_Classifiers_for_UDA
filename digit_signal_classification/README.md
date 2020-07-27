## Digit and Sign classification

### Getting Started
#### Installation
- Install PyTorch (Works on Version 0.2.0_3) and dependencies from http://pytorch.org.
- Install Python 2.7.
- Install tensorboardX.

### Download Dataset
Download MNIST Dataset [here](https://drive.google.com/file/d/1cZ4vSIS-IKoyKWPfcgxFMugw0LtMiqPf/view?usp=sharing). Resized image dataset is contained in the file.
Place it in the directory ./data.
All other datasets should be placed in the directory too.

### Train
Here is an example for task: MNIST to USPS,

```
python main.py \
--source mnist \
--target usps \
--num_classifiers_train 2 \
--lr 0.0002 \
--max_epoch 300 \
--all_use yes \
--optimizer adam
```
