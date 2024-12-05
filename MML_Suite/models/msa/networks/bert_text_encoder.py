## This file was taken and modified from https://github.com/declare-lab/MSA-Robustness/blob/main/Self-MM/models/subNets/BertTextEncoder.py
from os import PathLike
from typing import Literal

import torch
import torch.nn as nn
from experiment_utils import format_path_with_env
from transformers import BertModel, BertTokenizer

__all__ = ["BertTextEncoder"]


class BertTextEncoder(nn.Module):
    def __init__(
        self,
        language: Literal["en", "ch"] = "en",
        use_finetune: bool = False,
        pretrained_path: PathLike = "pretrained_model/bert_en",
    ):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()
        pretrained_path = format_path_with_env(pretrained_path)

        assert language in ["en", "cn"], "Language must be either 'en' or 'cn'"
        self.language = language
        tokenizer_class = BertTokenizer
        model_class = BertModel
        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_path, do_lower_case=True)
        self.model = model_class.from_pretrained(pretrained_path)

        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = (
            text[:, 0, :].long(),
            text[:, 1, :].float(),
            text[:, 2, :].long(),
        )
        if self.use_finetune:
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
            )[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                )[0]  # Models outputs are now tuples
        return last_hidden_states


if __name__ == "__main__":
    bert_normal = BertTextEncoder()
