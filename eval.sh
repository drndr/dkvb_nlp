for seed in 1; do
    for DATASET in mrpc qqp mnli; do
	    python3 finetune_eval.py "$DATASET" 5 True 32 0.00003 0.01 True
	    python3 finetune_eval.py "$DATASET" 5 False 32 0.00003 0.01 True
    done
done