import copy
from unittest import makeSuite
import torch
import copy
import torch.nn.utils.prune as prune
import numpy as np

def generate_mask_parameters(model, mask_parameters, exception=None):
    for name, module in model.named_modules():
        if exception is not None and exception in name:
            continue
        if module in mask_parameters:
            mask = torch.ones_like(module.weight)
            prune.CustomFromMask.apply(module, "weight", mask)
            yield module.weight_mask, module.weight_orig

class Pruner(object):
    def __init__(self, model):
        self.model = model
        self.mask_parameters = []
        self.isPruned = False
        for ii in range(24):
            self.mask_parameters.append((self.model.model.decoder.layers[ii].self_attn.k_proj, 'weight'))
            self.mask_parameters.append((self.model.model.decoder.layers[ii].self_attn.v_proj, 'weight'))
            self.mask_parameters.append((self.model.model.decoder.layers[ii].self_attn.q_proj, 'weight'))
            self.mask_parameters.append((self.model.model.decoder.layers[ii].self_attn.out_proj, 'weight'))
            self.mask_parameters.append((self.model.model.decoder.layers[ii].fc1, 'weight'))
            self.mask_parameters.append((self.model.model.decoder.layers[ii].fc2, 'weight'))
       

    def get_sparsity_ratio(self):
        print("Is Model Pruned : {}".format(self.isPruned))
        sum_list = 0
        zero_sum = 0
        for module, _ in self.mask_parameters:
            sum_list += float(module.weight.nelement())
            zero_sum += float(torch.sum(module.weight == 0))
        return 100*zero_sum/sum_list

    
    def prune_model_structured(self, per_zero, isRandom = False):
        for l in self.mask_parameters:
            prune.global_unstructured(
                tuple([l]),
                pruning_method=prune.L1Unstructured,
                amount=per_zero,
            )
        self.isPruned = True
        
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

    def generate_mask_parameters(self, exception=None):
        masked_parameters = []
        for module, _ in self.mask_parameters:
            mask = torch.ones_like(module.weight)
            prune.CustomFromMask.apply(module, "weight", mask)
            masked_parameters.append((module.weight_mask, module.weight))
        return masked_parameters

    def prune_model_random_erk(self, sparsity, scope = None):
        self.masked_parameters = self.generate_mask_parameters()
        total_params = 0
        for (mask, weight) in self.masked_parameters:
            total_params += weight.numel()
        is_epsilon_valid = False
        erk_power_scale = 1.0
        dense_layers = set()
        density = 1 - sparsity

        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, (mask, params) in enumerate(self.masked_parameters):
                n_param = np.prod(params.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density
                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                        np.sum(mask.shape) / np.prod(mask.shape)
                    ) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True
        self.density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, (mask, params) in enumerate(self.masked_parameters):
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                self.density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                self.density_dict[name] = probability_one
            mask.data.copy_((torch.rand(mask.shape) < self.density_dict[name]).float())
            total_nonzero += self.density_dict[name] * mask.numel()
            params.mul_(mask)
        
        self.isPruned = True
        print("ERK Pruning done !!!")


    def prune_n_m_sparsity(self, prune_n, prune_m):
        """
        Code reference : https://github.com/locuslab/wanda/blob/main/lib/prune.py#L105
        """
        self.masked_parameters = self.generate_mask_parameters()
        for name, (mask, params) in enumerate(self.masked_parameters):
            W_metric = torch.abs(params)
            W_mask = (torch.zeros_like(W_metric)==1)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            mask[W_mask] = 0
            params.mul_(mask)
        print(f"N:M ({prune_n}:{prune_m}) Pruning Done!")
    
    def get_prune_mask(self):
        mask_dict = {}
        for i in range(0, 24):
            mask_dict[i] = {}
            mask_dict[i]['decoder.layers.{}.self_attn.k_proj'.format(i)] = copy.deepcopy(self.model.model.decoder.layers[i].self_attn.k_proj.weight_mask)
            mask_dict[i]['decoder.layers.{}.self_attn.v_proj'.format(i)]   = copy.deepcopy(self.model.model.decoder.layers[i].self_attn.v_proj.weight_mask)
            mask_dict[i]['decoder.layers.{}.self_attn.q_proj'.format(i)] = copy.deepcopy(self.model.model.decoder.layers[i].self_attn.q_proj.weight_mask)
            mask_dict[i]['decoder.layers.{}.self_attn.out_proj'.format(i)] = copy.deepcopy(self.model.model.decoder.layers[i].self_attn.out_proj.weight_mask)
            mask_dict[i]['decoder.layers.{}.fc1'.format(i)] = copy.deepcopy(self.model.model.decoder.layers[i].fc1.weight_mask)
            mask_dict[i]['decoder.layers.{}.fc2'.format(i)] = copy.deepcopy(self.model.model.decoder.layers[i].fc2.weight_mask)
        return mask_dict

    def prune_model_custom(self, mask_dict):
        print("Pruning with custom mask!")
        for ii in range(0, 24):
            prune.CustomFromMask.apply(self.model.model.decoder.layers[ii].self_attn.k_proj, 'weight', mask=mask_dict[ii]['decoder.layers.{}.self_attn.k_proj'.format(ii)])
            prune.CustomFromMask.apply(self.model.model.decoder.layers[ii].self_attn.v_proj, 'weight', mask=mask_dict[ii]['decoder.layers.{}.self_attn.v_proj'.format(ii)])
            prune.CustomFromMask.apply(self.model.model.decoder.layers[ii].self_attn.q_proj, 'weight', mask=mask_dict[ii]['decoder.layers.{}.self_attn.q_proj'.format(ii)])
            prune.CustomFromMask.apply(self.model.model.decoder.layers[ii].self_attn.out_proj, 'weight', mask=mask_dict[ii]['decoder.layers.{}.self_attn.out_proj'.format(ii)])
            prune.CustomFromMask.apply(self.model.model.decoder.layers[ii].fc1, 'weight', mask=mask_dict[ii]['decoder.layers.{}.fc1'.format(ii)])
            prune.CustomFromMask.apply(self.model.model.decoder.layers[ii].fc2, 'weight', mask=mask_dict[ii]['decoder.layers.{}.fc2'.format(ii)])
        self.isPruned = True

    def remove_prune(self):
        print("Removing Prune!")
        for ii in range(0, 24):
            prune.remove(self.model.bert.encoder.layer[ii].attention.self.query, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].attention.self.key, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].attention.self.value, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].attention.output.dense, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].intermediate.dense, 'weight')
            prune.remove(self.model.bert.encoder.layer[ii].output.dense, 'weight')
        self.isPruned = True