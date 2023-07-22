import argparse
from transformers import glue_processors as processors
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


MODEL_CLASSES = {"bert": (BertConfig, BertForSequenceClassification, BertTokenizer)}

def debug_QNLI(args):
    args.do_lower_case = True
    return args

def base_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="/data/ajay_data/GLUE_DATA/MNLI", type=str,  help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument("--model_type", default="bert", type=str,  help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,)
    # parser.add_argument("--model_name_or_path", default="/data/ajay_data/bert_pretraining/pretrained-bert/checkpoint-11000", type=str,)
    parser.add_argument("--task_name", default="MNLI", type=str,  help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),)
    parser.add_argument("--output_dir", default="/data/ajay_data/sparseland/bert_sparseland", type=str,  help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--max_seq_length", default=256, type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.",)
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=0.0, type=float, help="Total number of training epochs to perform.",)
    parser.add_argument("--num_finetune_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",)

    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",)
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",)
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--before_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    args = parser.parse_args()
    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()
    args = debug_QNLI(args)

    return args, MODEL_CLASSES