# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import copy

class PromptIndicator_plain(nn.Module):
    def __init__(self, args): # MODEL.PROMPT_INDICATOR
        super().__init__()
        
        self.d_model = args.hidden_dim
        self.args = args
        self._init_class_prompts(args)
        self.num_classes_for_CEM = args.num_classes_for_CEM
        self.train_topk = args.train_topk

    def _init_class_prompts(self, args): # MODEL.PROMPT_INDICATOR.CLASS_PROMPTS
        # load given vectors
        if args.init_vectors:
            if args.init_vectors[-3:] == "pth":
                class_prompts = torch.load(args.init_vectors)
            elif args.init_vectors[-3:] == "npy":
                class_prompts = torch.tensor(np.load(args.init_vectors), dtype=torch.float32)
            else:
                raise KeyError
            if args.fix_class_prompts:
                self.register_buffer("class_prompts", class_prompts)
            else:
                self.register_parameter("class_prompts", nn.Parameter(class_prompts))
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

    def forward(self, srcs, targets=None, kwargs={}):
        """
        srcs: bs, l, c
        """
        bs = srcs.shape[1]
        # get class prompts
        if self.convert_vector is not None:
            class_prompts = self.vector_ln(self.convert_vector(self.class_prompts))
        else:
            class_prompts = self.class_prompts
            
        uniq_labels = torch.tensor([t for t in targets[0].keys() if isinstance(t, int)])
        uniq_labels = torch.unique(uniq_labels).to("cpu")
        max_len = self.train_topk
        max_pad_len = self.train_topk
        all_ids = torch.tensor(range(self.num_classes_for_CEM))

        uniq_labels = uniq_labels[torch.randperm(len(uniq_labels))][:max_len]
        select_id = uniq_labels.tolist()
        if len(select_id) < max_pad_len: 
            pad_len = max_pad_len - len(uniq_labels)
            extra_list = torch.tensor([i for i in all_ids if i not in uniq_labels])
            extra_labels = extra_list[torch.randperm(len(extra_list))][:pad_len]
            select_id += extra_labels.tolist()
            select_id.sort()
            
        return class_prompts, select_id