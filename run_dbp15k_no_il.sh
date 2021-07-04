CUDA_VISIBLE_DEVICES=$1 python3 src/run_dbp15k.py \
	--file_dir data/DBP15K/$3 \
	--rate 0.3 \
	--lr .0005 \
	--epochs 500 \
	--hidden_units "400,400,200" \
	--check_point 50  \
	--bsize 7500 \
	--semi_learn_step 5 \
	--csls \
	--csls_k 3 \
	--seed $2 \
#	--unsup \
#	--unsup_k 3000
	
