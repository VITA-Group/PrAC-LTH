CUDA_VISIBLE_DEVICES=1 python -u main_train.py \
	--data ../data \
	--dataset cifar10 \
	--arch res20s \
	--batch_size 128 \
	--lr 0.1 \
	--init_dir PrAC_lt_cifar10_res20s/1checkpoint.pth.tar \
	--mask_dir PrAC_lt_cifar10_res20s/1checkpoint.pth.tar \
	--save_dir retrain_PrAC_lt_cifar10_res20s/1