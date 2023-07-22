import os
import numpy as np
from utils.misc import *
from utils.glue_utils import *
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

def evaluate(args, model, tokenizer):
    eval_task_names, eval_outputs_dirs = (args.task_name, ), (args.output_dir,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_examples(args, eval_task, tokenizer, evaluate=True)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        
        # Eval!
        print_info("***** Running evaluation || Task : {} *****".format(eval_task))
        # print_info("Num examples = {}".format(len(eval_dataset)))
        # print_info("Batch size   = {}".format(args.eval_batch_size))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        for key in sorted(result.keys()):
            print_info("{} = {}".format(key, str(result[key])))
    return results
