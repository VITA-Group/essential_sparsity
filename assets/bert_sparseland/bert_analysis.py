from datetime import datetime
import numpy as np
import torch

from utils.evaluate import *
from utils.glue_utils import *
from utils.misc import *
from utils.optim import *
from argument import base_arguments
from trainer import Trainer


def main():
    args, MODEL_CLASSES = base_arguments()
    print(args)
    device = torch.device("cuda:5")
    args.device = device
    set_seed(args)

    fopen = open("logs/MNLI_train_log.txt", "a")
    prune_ratio = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for p in prune_ratio:
        main_trainer = Trainer(args, MODEL_CLASSES)
        fprint_info(f"----------------- Prune Ratio : {p} --------------------", fopen)
        main_trainer.pruner.prune_model(p)
        fprint_info(f"Sparsity : {main_trainer.pruner.get_sparsity_ratio()} %", fopen)
        for i in range(0, int(args.num_train_epochs)):
            main_trainer.train_epoch()
            fprint_info(f"Epoch {i + 1} >> {main_trainer.evaluate_model()}", fopen)

if __name__ == "__main__":
    main()