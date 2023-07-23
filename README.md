# The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter

https://arxiv.org/abs/2306.03805

### Abstract
Large pre-trained transformers are show-stealer in modern-day deep learning, and
it becomes crucial to comprehend the parsimonious patterns that exist within them
as they grow in scale. With exploding parameter counts, Lottery Ticket Hypothesis
(LTH) and its variants, have lost their pragmatism in sparsifying them due to high
computation and memory bottleneck of repetitive train-prune-retrain routine of
iterative magnitude pruning (IMP) which worsens with increasing model size. In
this paper, we comprehensively study induced sparse patterns across multiple
large pre-trained vision and language transformers. We propose the existence of
– “essential sparsity” defined with a sharp dropping point beyond which the
performance declines much faster w.r.t the rise of sparsity level, when we directly
remove weights with the smallest magnitudes in one-shot. We also present an
intriguing emerging phenomenon of abrupt sparsification during the pre-training
of BERT, i.e., BERT suddenly becomes heavily sparse in pre-training after certain
iterations. Moreover, our observations also indicate a counter-intuitive finding that
BERT trained with a larger amount of pre-training data tends to have a better ability
to condense knowledge in comparatively relatively fewer parameters. Lastly, we
investigate the effect of the pre-training loss on essential sparsity and discover that
self-supervised learning (SSL) objectives trigger stronger emergent sparsification
properties than supervised learning (SL). 

<img width="822" alt="image" src="https://github.com/VITA-Group/essential_sparsity/assets/6660499/64eeacd0-c360-403a-92cc-7281e22fc77e">

<img width="836" alt="image" src="https://github.com/VITA-Group/essential_sparsity/assets/6660499/ca3b1dbe-bc1c-45d9-a6ea-d1d0c991e997">

<img width="827" alt="image" src="https://github.com/VITA-Group/essential_sparsity/assets/6660499/a1e435e7-3082-42ec-9fc9-588a8084fa27">

# Installation

Our implementation is based on [Huggingface repo](https://github.com/huggingface/transformers). Details are referred to README [here](https://github.com/TAMU-VITA/BERT-Tickets/blob/master/transformers-master/README.md). 

### With pip

First you need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and/or [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When TensorFlow 2.0 and/or PyTorch has been installed, 🤗 Transformers can be installed using pip as follows:

```bash
pip install transformers
```


#### Glue task:

```shell
python -u bert_analysis.py
	   --output_dir tmp/mnli 
	   --logging_steps <ADD_VALUE> 
	   --task_name MNLI
     --do_lower_case
	   --data_dir glue_data/MNLI 
	   --model_type bert 
	   --model_name_or_path bert-base-uncased 
	   --max_seq_length  <ADD_VALUE> 
	   --learning_rate 2e-5 
	   --num_train_epochs  <ADD_VALUE>  
	   --overwrite_output_dir 
	   --evaluate_during_training 
	   --save_steps  <ADD_VALUE> 
	   --eval_all_checkpoints 
	   --seed  <ADD_VALUE> 
```

#### SQuAD task:

```shell
python -u squad_analysis.py 
	   --output_dir <ADD_VALUE> 
	   --model_type bert 
	   --model_name_or_path bert-base-uncased 
       --do_train 
       --do_eval 
       --do_lower_case 
       --train_file SQuAD/train-v1.1.json 
       --predict_file SQuAD/dev-v1.1.json 
       --per_gpu_train_batch_size <ADD_VALUE>  
       --learning_rate 3e-5 
       --num_train_epochs <ADD_VALUE>  
       --max_seq_length <ADD_VALUE>  
       --doc_stride 128 
       --evaluate_during_training 
       --eval_all_checkpoints 
       --overwrite_output_dir 
       --logging_steps <ADD_VALUE>  
       --save_steps <ADD_VALUE>  
       --seed <ADD_VALUE> 
```

## Citation

If you use this code for your research, please cite our paper:

```
@article{jaiswal2023emergence,
  title={The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter},
  author={Jaiswal, Ajay and Liu, Shiwei and Chen, Tianlong and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2306.03805},
  year={2023}
}
```







