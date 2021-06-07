CUDA_VISIBLE_DEVICES=0 python -u main_imp.py \
	--data ../data \
	--dataset cifar10 \
	--arch res20s \
	--batch_size 128 \
	--lr 0.1 \
	--pruning_times 16 \
	--prune_type rewind_lt \
	--rewind_epoch 2 \
	--save_dir lt_cifar10_res20s
