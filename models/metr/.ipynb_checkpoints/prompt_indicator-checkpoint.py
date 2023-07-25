# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn
import numpy as np
import math

from .prompt_transformer import prompt_TransformerDecoderLayer, _get_clones
from .prompt_classifier import build_label_classifier
from .class_criterion import ClassDecoderCriterion

class RetentionPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        """ args: MODEL.PROMPT_INDICATOR.RETENTION_POLICY"""
        # select some class
        self.train_min_classes=args.retention_policy_train_min_classes
        self.train_max_classes=args.retention_policy_train_max_classes
        self.train_class_thr=args.retention_policy_train_class_thr
        # for eval
        self.eval_min_classes = args.retention_policy_eval_min_classes
        self.eval_max_classes = args.retention_policy_eval_max_classes
        self.eval_class_thr = args.retention_policy_eval_class_thr
    
    @torch.no_grad()
    def forward(self, label_logits, force_sample_probs=None, num_classes=None):
        """ label_logits: bs * K  """
        """ Return:       bs * K' """
        label_prob = label_logits.sigmoid()
        if self.training:
            if force_sample_probs is not None:
                label_prob = torch.where(force_sample_probs >= 0., force_sample_probs, label_prob)
            min_classes = num_classes.clamp(max=self.train_min_classes) if num_classes is not None else self.train_min_classes
            max_classes = num_classes.clamp(max=self.train_max_classes) if num_classes is not None else self.train_max_classes
            class_thr = self.train_class_thr
        else:
            min_classes = num_classes.clamp(max=self.eval_min_classes) if num_classes is not None else self.eval_min_classes
            max_classes = num_classes.clamp(max=self.eval_max_classes) if num_classes is not None else self.eval_max_classes
            class_thr = self.eval_class_thr
            
        num_above_thr = (label_prob >= class_thr).sum(dim=1)
        if isinstance(min_classes, torch.Tensor):
            num_train = num_above_thr.where(num_above_thr > min_classes, min_classes).where(num_above_thr < max_classes, max_classes)
        else:
            num_train = num_above_thr.clamp(min=min_classes, max=max_classes)
        sorted_idxs = label_prob.argsort(dim=1, descending=True)
        
        bs_idx, cls_idx = [], []
        for id_b, (sorted_idx) in enumerate(sorted_idxs):
            n_train = num_train[id_b]
            cls_idx.append(sorted_idx[:n_train].sort().values)
            bs_idx.append(torch.full_like(cls_idx[-1], id_b))
        
        return torch.cat(bs_idx), torch.cat(cls_idx)


class PromptIndicator(nn.Module):
    def __init__(self, args): # MODEL.PROMPT_INDICATOR
        super().__init__()
        
        # class prompts
        self.d_model = args.hidden_dim
        self.args = args
        self._init_class_prompts(args)
        # prompt blocks
        self.num_blocks = args.prompt_indicator_num_blocks
        self.level_preserve = args.prompt_indicator_level_preserve # only work for DeformableDETR
        
        prompt_block = prompt_TransformerDecoderLayer(args)

        self.prompt_blocks = _get_clones(prompt_block, self.num_blocks)

        # For classification
        self.classifier_label = build_label_classifier(args)
        self.classifier_label = nn.ModuleList([self.classifier_label for _ in range(self.num_blocks)])
        self.criterion = ClassDecoderCriterion(args)

        # For filter
        if args.retain_categories:
            self.retention_policy = RetentionPolicy(args)
        else:
            self.retention_policy = None
            
    def _postmap(self, data, MIN, MAX):
        d_min=np.max(data, axis=-1)
        d_max=np.min(data, axis=-1)
        return MIN + (MAX-MIN) / (d_max-d_min) * (data - d_min)
        
    def _init_class_prompts(self, args): # MODEL.PROMPT_INDICATOR.CLASS_PROMPTS
        # load given vectors
        if args.init_vectors:
            if args.init_vectors[-3:] == "pth":
                class_prompts = torch.load(args.init_vectors)
            elif args.init_vectors[-3:] == "npy": # here
                class_prompts = torch.tensor(np.load(args.init_vectors), dtype=torch.float32)
            else:
                raise KeyError
            if args.fix_class_prompts:
                self.register_buffer("class_prompts", class_prompts)
            else:
                self.register_parameter("class_prompts", nn.Parameter(class_prompts))
        # rand init
        else:
            class_prompts = torch.zeros(self.args.num_class, self.d_model)
            assert args.fix_class_prompts == False
            self.register_parameter("class_prompts", nn.Parameter(class_prompts))
            nn.init.normal_(self.class_prompts.data)
        
        # if the dimensiton does not match.
        if class_prompts.shape[1] != self.d_model:
            self.convert_vector = nn.Linear(class_prompts.shape[1], self.d_model)
            self.vector_ln = nn.LayerNorm(self.d_model)
        else:
            self.convert_vector = None

    def forward(self, srcs, mask, targets=None, kwargs={}):
        """
        srcs: bs, l, c
        mask:
        """
        
        bs = srcs.shape[1]

        # get class prompts
        if self.convert_vector is not None:
            class_prompts = self.vector_ln(self.convert_vector(self.class_prompts))
        else:
            class_prompts = self.class_prompts
        tgt_class = class_prompts.unsqueeze(0).repeat(bs, 1, 1)
        origin_class_vector = tgt_class

        output_label_logits = []
        output_feats = []
        
        #all_srcs = []
        for lid, layer in enumerate(self.prompt_blocks):
            tgt_class = layer(tgt_class, None, None, srcs=srcs, src_padding_masks=mask, **kwargs)
            label_logits = self.classifier_label[lid](tgt_class, class_vector=origin_class_vector)
            label_logits = label_logits.view(bs, -1)
            output_label_logits.append(label_logits)
            output_feats.append(tgt_class)

        # organize outputs
        outputs = {
            'tgt_class': tgt_class,               # bs, k, d
            'cls_label_logits': label_logits,     # bs, k
            'cls_output_feats': tgt_class, # bs, k, d
        }

        # select some classes
        # with here
        if self.retention_policy is not None:
            force_sample_probs = torch.stack([t["force_sample_probs"] for t in targets]) if self.training else None
            num_classes = torch.stack([torch.tensor(self.args.num_classes).cuda() for t in targets])
            bs_idxs, cls_idxs = self.retention_policy(label_logits, force_sample_probs, num_classes)
            return_tgts = tgt_class[bs_idxs, cls_idxs]
            outputs.update({
                'bs_idx': bs_idxs,
                'cls_idx': cls_idxs,
                'original_tgt_class': tgt_class,
                'tgt_class': return_tgts,
            })

        if len(output_label_logits) > 1:
            aux_outputs = [{'cls_label_logits': a}
                           for a in output_label_logits[:-1]]
        else:
            aux_outputs = []

        # organize losses
        assert targets is not None
        
        loss_dict = self.criterion(outputs, aux_outputs, targets)
        return outputs, loss_dict
