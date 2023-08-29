import torch.nn as nn
import torch
from models.set_decoder_absa import SetDecoder_absa
from models.set_criterion_absa import SetCriterion_absa
from models.seq_encoder import SeqEncoder
from utils.functions_absa import generate_triple_absa
import copy
import torch.nn.functional as F
from pdb import set_trace as stop


class SetPred4RE_absa(nn.Module):

    def __init__(self, args, num_classes):
        super(SetPred4RE_absa, self).__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.linear = nn.Linear(config.hidden_size, self.args.max_text_length, bias=False) # add 对应论文的公式（8），线性层没有偏置
        self.num_classes = num_classes
        self.decoder = SetDecoder_absa(args, config, args.num_generated_triples, args.num_decoder_layers, num_classes, return_intermediate=False)
        # self.criterion = SetCriterion(num_classes, na_coef=args.na_rel_coef, losses=["entity", "relation", "quintuple_relation"], matcher=args.matcher) 
        self.criterion = SetCriterion_absa(num_classes, na_coef=args.na_rel_coef, losses=["entity", "relation"], matcher=args.matcher) # quintuple_relation
        # '--matcher', type=str, default="avg", choices=['avg', 'min']

        self.kl_loss = nn.KLDivLoss()  # add


    def forward(self, input_ids, attention_mask, targets=None):
        last_hidden_state, pooler_output = self.encoder(input_ids, attention_mask) # hidden state, cls
        _, pooler_output2 = self.encoder(input_ids, attention_mask) # pooler_output= cls, dim = bsz, hidden_size
       
        hidden_states, class_logits, aspect_start_logits, aspect_end_logits, opinion_start_logits, opinion_end_logits = self.decoder(encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
        # sub_start_logits = sub_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        # sub_end_logits = sub_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        # obj_start_logits = obj_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        # obj_end_logits = obj_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        aspect_start_logits = aspect_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        aspect_end_logits = aspect_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        opinion_start_logits = opinion_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        opinion_end_logits = opinion_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        
        outputs = { 
            'pred_rel_logits': class_logits, # bsz, q_num, num_class
            # 'sub_start_logits': sub_start_logits,  # bsz, q_num, seq_len
            # 'sub_end_logits': sub_end_logits,
            # 'obj_start_logits': obj_start_logits, 
            # 'obj_end_logits': obj_end_logits,
            'aspect_start_logits': aspect_start_logits, 
            'aspect_end_logits': aspect_end_logits,
            'opinion_start_logits': opinion_start_logits, 
            'opinion_end_logits': opinion_end_logits,
            'v_logits': hidden_states, # 直接将中间结果作为V_logits
        }

        if targets is not None:
            loss = self.criterion(outputs, targets) 
            # targets是一个list, len(targets)=bsz, targets[i]是dict，
            # dict 顺序是sub_s,sub_e, obj_s,obj_e,asp_s, asp_e, op_s,op_e, r,于outputs的顺序并不相同

            # 注意此处一个是log_softmax, 一个是softmax
            # kl_loss1 = self.kl_loss(F.log_softmax(pooler_output, dim=-1), F.softmax(pooler_output2, dim=-1)).sum(-1) # add KL loss有方向性
            # kl_loss2 = self.kl_loss(F.log_softmax(pooler_output2, dim=-1), F.softmax(pooler_output, dim=-1)).sum(-1) # add
            # kl_loss = (kl_loss1 + kl_loss2) / 2 # add 
            # loss = loss + 2 * kl_loss # add

            return loss, outputs
        else:
            return outputs


    def gen_triples_absa(self, input_ids, attention_mask, info):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            # print(outputs)
            pred_triple = generate_triple_absa(outputs, info, self.args, self.num_classes)
            # print(pred_triple)
        return pred_triple

    '''
    def batchify(self, batch_list):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)
        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
        if self.args.use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False).cuda() for k, v in t.items()} for t in targets]
        else:
            targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False) for k, v in t.items()} for t in targets]
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}
        return input_ids, attention_mask, targets, info
    '''

    @staticmethod
    def get_loss_weight(args):
        return {"relation": args.rel_loss_weight, "head_entity": args.head_ent_loss_weight, "tail_entity": args.tail_ent_loss_weight}





