for seed in 1; do
    for DATASET in "R8"; do
		python3 finetune_eval.py "$DATASET" "dkvb" 10 32 0.1 --lr_values 0.1 --decoder "softmax" --pooling "mean" # non-parametric
		#python3 finetune_eval.py "$DATASET" "dkvb" 10 16 0.001 --lr_values 0.01 --decoder "mlp" --pooling "mean"  # parametric
	    #python3 finetune_eval.py "$DATASET" "dkvb" 10 32 0.1 --lr_values 0.1 --decoder "softmax" --pooling "cls" # non-parametric
		#python3 finetune_eval.py "$DATASET" "dkvb" 10 16 0.0001 --lr_values 0.01 --decoder "mlp" --pooling "cls"  # parametric
		#python3 finetune_eval.py "$DATASET" "base" 10 16 0.00005 # full
		#python3 finetune_eval.py "$DATASET" "base" 5 16 0.001 --is_frozen # frozen
		#python3 finetune_eval.py "$DATASET" "dkvb" 5 32 0.1 --lr_values 0.1 --decoder "softmax" --pooling "mean" --pool_before # non-parametric pool before
		#python3 finetune_eval.py "$DATASET" "dkvb" 5 32 0.0001 --lr_values 0.1 --decoder "mlp" --pooling "mean" --pool_before # parametric pool before	
		#python3 finetune_eval.py "$DATASET" "dkvb" 5 32 0.1 --lr_values 0.1 --decoder "softmax" --pooling "cls" --pool_before # non-parametric pool befo#re
		#python3 finetune_eval.py "$DATASET" "dkvb" 5 32 0.0001 --lr_values 0.01 --decoder "mlp" --pooling "cls" --pool_before # parametric pool before			
    done
done

for seed in 1; do
    for DATASET in "20ng"; do
		python3 finetune_eval.py "$DATASET" "dkvb" 10 32 0.1 --lr_values 0.1 --decoder "softmax" --pooling "mean" # non-parametric
		#python3 finetune_eval.py "$DATASET" "dkvb" 10 16 0.001 --lr_values 0.01 --decoder "mlp" --pooling "mean"  # parametric
	    #python3 finetune_eval.py "$DATASET" "dkvb" 5 16 0.1 --lr_values 0.1 --decoder "softmax" --pooling "cls" # non-parametr#ic
		#python3 finetune_eval.py "$DATASET" "dkvb" 10 16 0.0001 --lr_values 0.01 --decoder "mlp" --pooling "cls"  # parametric
		#python3 finetune_eval.py "$DATASET" "base" 10 16 0.00005 # full
		#python3 finetune_eval.py "$DATASET" "base" 5 16 0.001 --is_frozen # frozen
		#python3 finetune_eval.py "$DATASET" "dkvb" 10 32 0.1 --lr_values 0.1 --decoder "softmax" --pooling "mean" --pool_before # non-parametric pool before
		#python3 finetune_eval.py "$DATASET" "dkvb" 10 16 0.0001 --lr_values 0.01 --decoder "mlp" --pooling "mean" --pool_before # parametric pool before
		#python3 finetune_eval.py "$DATASET" "dkvb" 5 32 0.1 --lr_values 0.1 --decoder "softmax" --pooling "cls" --pool_before # non-parametric pool before
		#python3 finetune_eval.py "$DATASET" "dkvb" 10 16 0.001 --lr_values 0.1 --decoder "mlp" --pooling "cls" --pool_before # parametric pool before
    done
done