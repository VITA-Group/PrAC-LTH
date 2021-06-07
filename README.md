# Efficient Lottery Ticket Finding: Less Data is More





## Overview





## Prerequisites

Pytorch >= 1.4

torchvision

advertorch

## Usage

#### Vanilla Lottery Tickets

```
python -u main_imp.py \
	--data data/cifar10 \
	--dataset cifar10 \
	--arch res20s \
	--batch_size 128 \
	--lr 0.1 \
	--pruning_times 16 \
	--prune_type rewind_lt \
	--rewind_epoch 2 \
	--save_dir lt_cifar10_res20s
```

#### PrAC Lottery Tickets

```
python -u main_PrAC_imp.py \
	--data data/cifar10 \
	--dataset cifar10 \
	--arch res20s \
	--split_file npy_files/cifar10-train-val.npy \
	--batch_size 128 \
	--lr 0.1 \
	--pruning_times 16 \
	--eb_eps 0.08 \
	--prune_type rewind_lt \
	--rewind_epoch 2 \
	--threshold 0 \
	--save_dir PrAC_lt_cifar10_res20s
	
```

#### Train subnetworks 

```
python -u main_train.py \
	--data data/cifar10 \
	--dataset cifar10 \
	--arch res20s \
	--batch_size 128 \
	--lr 0.1 \
	--init_dir PrAC_lt_cifar10_res20s/1checkpoint.pth.tar \ 
	--mask_dir PrAC_lt_cifar10_res20s/1checkpoint.pth.tar \ # sparsity=20%
	--save_dir retrain_PrAC_lt_cifar10_res20s/1
```

