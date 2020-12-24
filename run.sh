# running full model on FR-EN
CUDA_VISIBLE_DEVICES=$1 python3 run.py \
	--file_dir data/DBP15K/fr_en \
	--rate 0.3 \
	--lr .0005 \
	--epochs 1000 \
	--wo_NNS \
	--wo_K \
	--hidden_units "400,400,200" \
	--check_point 50  \
	--bsize 7500 \
	--il \
	--il_start 500 \
	--semi_learn_step 5 \
	--csls \
	--csls_k 3 \
	--seed $2 \
#	--unsup \
#	--unsup_k 3000
	
