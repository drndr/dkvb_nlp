for seed in 1; do
    for DATASET in R8; do
	    #python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "mlp" "mean" --pool_before --wandb_enabled
		python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "mlp" "mean" --wandb_enabled
		python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.01 "mlp" "mean" --wandb_enabled
		python3 finetune_eval.py "$DATASET" 5 32 0.001 0.1 "mlp" "mean" --wandb_enabled
		python3 finetune_eval.py "$DATASET" 5 32 0.001 0.01 "mlp" "mean" --wandb_enabled
		python3 finetune_eval.py "$DATASET" 5 32 0.01 0.1 "mlp" "mean" --wandb_enabled
		python3 finetune_eval.py "$DATASET" 5 32 0.01 0.01 "mlp" "mean" --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "mlp" "cls" --pool_before --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "mlp" "cls" --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "softmax" "mean" --pool_before --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "softmax" "mean" --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "softmax" "cls" --pool_before --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "softmax" "cls" --wandb_enabled
    done
done