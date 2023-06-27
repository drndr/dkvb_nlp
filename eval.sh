for seed in 1; do
    for DATASET in "20ng"; do
	    #python3 finetune_eval.py "$DATASET" 5 32 0.00003 0.01 --pool_before --wandb_enabled
	    #python3 finetune_eval.py "$DATASET" 5 32 0.00003 0.01 --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 --pool_before --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "mean" --pool_before --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "cls" --wandb_enabled
		python3 finetune_eval.py "$DATASET" 5 32 0.0001 0.1 "mean" --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.001 0.3 --pool_before --wandb_enabled
		#python3 finetune_eval.py "$DATASET" 5 32 0.001 0.3 --wandb_enabled
    done
done