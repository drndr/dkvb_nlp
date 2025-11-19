for seed in 1; do
    for DATASET in r8; do
	    python3 class_incremental.py "$DATASET" "dkvb" --epochs 5 --batch_size 16 --lr_global 0.01 --lr_values 0.01 --decoder "softmax" --pooling "mean" --key_init "full"
		python3 class_incremental.py "$DATASET" "dkvb" --epochs 5 --batch_size 16 --lr_global 0.01 --lr_values 0.01 --decoder "softmax" --pooling "mean" --key_init "wiki"
	    python3 class_incremental.py "$DATASET" "base" --epochs 5 --batch_size 16 --lr_global 0.01
		python3 class_incremental.py "$DATASET" "der" --epochs 5 --batch_size 16 --lr_global 0.01
    done
done