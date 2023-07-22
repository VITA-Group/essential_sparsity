import torch
import random
import copy
from utils.evaluate import *
from utils.glue_utils import *
from utils.misc import *
from utils.optim import *
from pruner import Pruner
from torchsummary import summary

class Trainer(object):
    def __init__(self, args, MODEL_CLASSES):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.args = args
        self.output_mode = output_modes[self.task_name]

        self.config, self.tokenizer, self.model = prepare_model(self.args, MODEL_CLASSES)
        self.model.to(self.device)

        self.train_dataset = load_examples(args, self.task_name, self.tokenizer, evaluate=False)
        self.train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(self.train_dataset, sampler=self.train_sampler, batch_size=self.train_batch_size)

        self.t_total = len(self.train_dataloader) * self.num_train_epochs
        self.optimizer, self.scheduler = configure_optimizers(args, self.model, self.t_total)

        print("-*-*-*-*-*-*-*-*-*-*-*- Trainer Statistics -*-*-*-*-*-*-*-*-*-*-*-")
        print_info("Task Name    = {}".format(len(self.task_name)))
        print_info("Num Examples = {}".format(len(self.train_dataset)))
        print_info("Num Epochs   = {}".format(self.num_train_epochs))
        print_info("Total optimization steps    = {}".format(self.t_total))
        summary(self.model,input_size=(768,),depth=2,batch_dim=1, dtypes=['torch.IntTensor']) 
        print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

        #pruning related unilities 
        self.pruner = Pruner(self.model)
        self.global_step, self.epoch_trained = 0, 0
        self.training_loss = []
        self.evaluation_result = {}

    def paramter_distribution(self, flatten = True):
        parameters = {}
        for i in range(0, 12):
            parameters[i] = {}
            parameters[i]['bert.encoder.layer.{}.attention.self.query'.format(i)] = self.model.bert.encoder.layer[i].attention.self.query.weight.clone().flatten().cpu().detach().numpy()
            parameters[i]['bert.encoder.layer.{}.attention.self.key'.format(i)]   = self.model.bert.encoder.layer[i].attention.self.key.weight.clone().flatten().cpu().detach().numpy()
            parameters[i]['bert.encoder.layer.{}.attention.self.value'.format(i)] = self.model.bert.encoder.layer[i].attention.self.value.weight.clone().flatten().cpu().detach().numpy()
            parameters[i]['bert.encoder.layer.{}.attention.output.dense'.format(i)] = self.model.bert.encoder.layer[i].attention.output.dense.weight.clone().flatten().cpu().detach().numpy()
            parameters[i]['bert.encoder.layer.{}.intermediate.dense'.format(i)] = self.model.bert.encoder.layer[i].intermediate.dense.weight.clone().flatten().cpu().detach().numpy()
            parameters[i]['bert.encoder.layer.{}.output.dense'.format(i)] = self.model.bert.encoder.layer[i].output.dense.weight.clone().flatten().cpu().detach().numpy()
        return parameters

    def save_model(self):
        output_dir = os.path.join(self.output_dir, "checkpoint_bert-base_{}_{}".format(self.task_name, self.epoch_trained))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model, os.path.join(output_dir, "model.pt"))
        torch.save(self.optimizer, os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.pruner.get_prune_mask(), os.path.join(output_dir, "mask.pt"))
        self.tokenizer.save_pretrained(output_dir)
        
    def evaluate_model(self):
        eval_task_names, eval_outputs_dirs = (self.task_name, ), (self.output_dir,)
        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = load_examples(self.args, eval_task, self.tokenizer, evaluate=True)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)
            
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    outputs = self.model(**inputs)
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
            
            if self.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif self.output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)
            for key in sorted(result.keys()):
                print_info("{} = {}".format(key, str(result[key])))
        
        self.evaluation_result["epoch"] = results
        return results

    def train_epoch(self, prob = 1.0, seed = 99):
        print_info("Training Epoch => {} || Learning Rate => {}".format(self.epoch_trained + 1, self.scheduler.get_lr()[0]))
        
        epoch_loss = 0.0
        random.seed(seed)
        epoch_iterator = tqdm(self.train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = self.model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            self.optimizer.zero_grad()

            loss.backward()
            epoch_loss += loss.item()

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.global_step += 1

        epoch_iterator.close()
        self.training_loss.append(epoch_loss/step)
        self.epoch_trained += 1
        print_info("After Training epoch {}, Current Loss : {:.3f}, Current LR : {}".format(self.epoch_trained, self.training_loss[self.epoch_trained - 1], self.scheduler.get_lr()[0]))
        

    def get_state_dict(self):
        return self.model.state_dict(), self.optimizer.state_dict(), [self.scheduler._last_lr, self.scheduler.last_epoch]

    def set_state_dict(self, model_state, optimizer_state = None, scheduler_state = None):
        self.model.load_state_dict(model_state)
        if optimizer_state != None:
            self.optimizer.load_state_dict(optimizer_state)
            self.scheduler =  get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps = self.t_total - scheduler_state[1]
            )
        print("Model and Optimizer State re-initialized.")
