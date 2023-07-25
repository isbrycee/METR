# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def build_label_classifier(args):
    if args.classifier_type=="multi":
        return DetrClassifier(args)
    elif args.classifier_type=="linear":
        return LinearClassifier(args)
    elif args.classifier_type=="dict":
        return DictClassifier(args)
    else:
        raise KeyError
        
        
class AbstractClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.classifier_num_layers
        if self.num_layers > 0:
            self.feature_linear = nn.ModuleList([nn.Linear(args.hidden_dim, args.hidden_dim) for i in range(self.num_layers)])
            self.skip_and_init = args.classifier_skip_and_init
            if self.skip_and_init:
                nn.init.constant_(self.feature_linear[-1].weight, 0.)
                nn.init.constant_(self.feature_linear[-1].bias, 0.)
        else:
            self.feature_linear = None
        if True:
            self.bias = True
            self.b = nn.Parameter(torch.Tensor(1))
        self.reset_parameters(args.classifier_init_prob)

    def reset_parameters(self, init_prob):
        if True:
            prior_prob = init_prob
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.b.data, bias_value)

    def forward(self, x, class_vector=None, cls_idx=None):
        if self.feature_linear is not None:
            skip = x
            for i in range(self.num_layers):
                x = F.relu(self.feature_linear[i](x)) if i < self.num_layers - 1 else self.feature_linear[i](x)
            if self.skip_and_init:
                x = skip + x
        new_feat = x
        assert x.dim() == 3
        W = self.getClassifierWeight(class_vector, cls_idx)
        sim = (x * W).sum(-1)
        if True:
            sim = sim + self.b
        return sim


class DetrClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.classifier_num_layers
        if self.num_layers > 0:
            self.feature_linear = nn.ModuleList([nn.Linear(args.hidden_dim, args.hidden_dim) for i in range(self.num_layers)])
        else:
            self.feature_linear = None
        self.classifier = nn.Linear(args.hidden_dim, 80)
        self.reset_parameters(args.classifier_init_prob)

    def reset_parameters(self, init_prob):
        prior_prob = init_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.classifier.bias.data, bias_value)

    def forward(self, x, class_vector=None, cls_idx=None):
        if self.feature_linear is not None:
            for i in range(self.num_layers):
                x = F.relu(self.feature_linear[i](x)) if i < self.num_layers - 1 else self.feature_linear[i](x)
        return self.classifier(x)


class LinearClassifier(AbstractClassifier):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, args):
        super().__init__(args)

        self.hidden_dim = args.hidden_dim
        self.W = nn.Parameter(torch.Tensor(self.hidden_dim))
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)

    def getClassifierWeight(self, class_vector=None, cls_idx=None):
        return self.W


class DictClassifier(AbstractClassifier):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, args):
        super().__init__(args)
        self.scale = args.hidden_dim ** -0.5

    def getClassifierWeight(self, class_vector=None, cls_idx=None):
        W = class_vector * self.scale
        return W