# LoRITa
This is the official repo for the TMLR paper "Structure-Preserving Network Compression Via Low-Rank Induced Training Through Linear Layers Composition"

The training code: train_CNN.py 
Important parameter:
  args.factor is the factorization parameter N.

The compression code: post_cnns.py
Important parameter:
  args.global_search = 'local' #uniform compression
  args.global_search = 'global' #global compression
  args.global_search = 'iter' #compress iteratively
  args.global_search = 'iter_fine' #compress iteratively with a smaller grid with finetuning
  args.global_search = 'iter_training' #compress iteratively with a smaller grid with retraining
