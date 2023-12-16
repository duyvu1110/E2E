import torch.nn as nn
import torch
from transformers import AutoModel


class SeqEncoder(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained('vinai/phobert-base-v2',vocal_size = 60000)
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask) # input_ids: bs,hidden
        last_hidden_state, pooler_output = out.last_hidden_state, out.pooler_output
        return last_hidden_state, pooler_output

class SeqEncoder_last(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.bert_directory)
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state, pooler_output = out.last_hidden_state, out.pooler_output

        if self.args.use_last_hidden_state=="True":
            return last_hidden_state, pooler_output
        elif self.args.use_last_hidden_state == "False": # use last four hidden state concat
            hidden_state = out.hidden_states[-4:]
            hidden_state = torch.stack(hidden_state, dim=-1)
            batch_size, length, _, _ = hidden_state.shape 
            hidden_state = hidden_state.reshape(batch_size, length, -1)
            # return hidden_state, pooler_output
        
        # return hidden_state, pooler_output # 4*bs,se,hi
        return last_hidden_state, pooler_output # bs,se,hi
