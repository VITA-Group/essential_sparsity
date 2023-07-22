import os
import numpy as np
from utils.misc import *
from torch.utils.data import TensorDataset
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

def prepare_model(args, MODEL_CLASSES):
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir = None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        model_max_length=512,
        do_lower_case=args.do_lower_case,
        cache_dir=None,
    )
    model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=None,
    )
    return config, tokenizer, model

def load_examples(args, task, tokenizer, evaluate=False):
    # print_info(["Example Loading started ..."])
    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print_info("Loading features from cached file {}".format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        print_info("Creating features from dataset file at {}.".format(args.data_dir))

        label_list = processor.get_labels()
        examples = (processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir))
        features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                output_mode=output_mode,
            )
        print_info(["Saving features into cached file {}".format(cached_features_file)])
        torch.save(features, cached_features_file)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    
    if output_mode == "classification": all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":   all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    # print_info("Tensor dataset created.")
    return dataset