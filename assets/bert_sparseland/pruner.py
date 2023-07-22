import copy
from unittest import makeSuite
import torch
import copy
import torch.nn.utils.prune as prune

class Pruner(object):
    def __init__(self, model):
        self.model = model
        self.mask_parameters = []
        self.isPruned = False
        for ii in range(12):
            self.mask_parameters.append((self.model.bert.encoder.layer[ii].attention.self.query, 'weight'))
            self.mask_parameters.append((self.model.bert.encoder.layer[ii].attention.self.key, 'weight'))
            self.mask_parameters.append((self.model.bert.encoder.layer[ii].attention.self.value, 'weight'))
            self.mask_parameters.append((self.model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
            self.mask_parameters.append((self.model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
            self.mask_parameters.append((self.model.bert.encoder.layer[ii].output.dense, 'weight'))
       

    def get_sparsity_ratio(self):
        print("Is Model Pruned : {}".format(self.isPruned))
        sum_list = 0
        zero_sum = 0
        for module, _ in self.mask_parameters:
            sum_list += float(module.weight.nelement())
            zero_sum += float(torch.sum(module.weight == 0))
            print(f"{module} > {float(torch.sum(module.weight == 0))/float(module.weight.nelement())} %")
        return 100*zero_sum/sum_list

    def prune_model(self, per_zero, isRandom = False):
        if isRandom == False:
            prune.global_unstructured(
                tuple(self.mask_parameters),
                pruning_method=prune.L1Unstructured,
                amount=per_zero,
            )
        else:
            print("Random pruning Model")
            prune.global_unstructured(
                tuple(self.mask_parameters),
                pruning_method=prune.RandomUnstructured,
                amount=per_zero,
            )
        self.isPruned = True

    def get_prune_mask(self):
        mask_dict = {}
        for i in range(0, 12):
            mask_dict[i] = {}
            mask_dict[i]['bert.encoder.layer.{}.attention.self.query'.format(i)] = copy.deepcopy(self.model.bert.encoder.layer[i].attention.self.query.weight_mask)
            mask_dict[i]['bert.encoder.layer.{}.attention.self.key'.format(i)]   = copy.deepcopy(self.model.bert.encoder.layer[i].attention.self.key.weight_mask)
            mask_dict[i]['bert.encoder.layer.{}.attention.self.value'.format(i)] = copy.deepcopy(self.model.bert.encoder.layer[i].attention.self.value.weight_mask)
            mask_dict[i]['bert.encoder.layer.{}.attention.output.dense'.format(i)] = copy.deepcopy(self.model.bert.encoder.layer[i].attention.output.dense.weight_mask)
            mask_dict[i]['bert.encoder.layer.{}.intermediate.dense'.format(i)] = copy.deepcopy(self.model.bert.encoder.layer[i].intermediate.dense.weight_mask)
            mask_dict[i]['bert.encoder.layer.{}.output.dense'.format(i)] = copy.deepcopy(self.model.bert.encoder.layer[i].output.dense.weight_mask)
        return mask_dict

    def prune_model_custom(self, mask_dict):
        print("Pruning with custom mask!")
        for ii in range(0, 12):
            prune.CustomFromMask.apply(self.model.bert.encoder.layer[ii].attention.self.query, 'weight', mask=mask_dict[ii]['bert.encoder.layer.{}.attention.self.query'.format(ii)])
            prune.CustomFromMask.apply(self.model.bert.encoder.layer[ii].attention.self.key, 'weight', mask=mask_dict[ii]['bert.encoder.layer.{}.attention.self.key'.format(ii)])
            prune.CustomFromMask.apply(self.model.bert.encoder.layer[ii].attention.self.value, 'weight', mask=mask_dict[ii]['bert.encoder.layer.{}.attention.self.value'.format(ii)])
            prune.CustomFromMask.apply(self.model.bert.encoder.layer[ii].attention.output.dense, 'weight', mask=mask_dict[ii]['bert.encoder.layer.{}.attention.output.dense'.format(ii)])
            prune.CustomFromMask.apply(self.model.bert.encoder.layer[ii].intermediate.dense, 'weight', mask=mask_dict[ii]['bert.encoder.layer.{}.intermediate.dense'.format(ii)])
            prune.CustomFromMask.apply(self.model.bert.encoder.layer[ii].output.dense, 'weight', mask=mask_dict[ii]['bert.encoder.layer.{}.output.dense'.format(ii)])
        self.isPruned = True

    def remove_prune(self):
        print("Removing Prune!")
        for ii in range(0, 12):
            prune.remove(self.model.bert.encoder.layer[ii].attention.self.query, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].attention.self.key, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].attention.self.value, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].attention.output.dense, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].intermediate.dense, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].output.dense, 'weight')
        self.isPruned = True