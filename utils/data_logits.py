import os
import torch
from torch import nn
import json
from tqdm import tqdm, trange
from typing import List
from transformers import AutoTokenizer
from collections import defaultdict
import re

from pdb import set_trace as stop


EMO_MAP = {
    -1: 1,
    0: 2,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 8,
}

def pass_offset(data_path, offset):
    """
    Set the offset in english dataset(Camera-COQE) starts from 0.
    """
    if 'smartphone' in data_path:
        return offset - 1
    else:
        return pass_offset

def proc_raw_offset(offset_spans: str, text, data_path):
    if offset_spans == '':
        # use 1 to denotes the empty span
        return (0, 0)
    # 7&&all 8&&of 9&&the 10&&Nikon 11&&DLSR 12&&models
    if 'smartphone' in data_path:
        offsets = re.findall(r'([0-9]+)&&(\S+)', offset_spans)
    else:
        offsets = re.findall(r'([0-9]+)&(\S+)', offset_spans) # type(offset_spans):str
    # [7&&all, 8&&of, 9&&the, 10&&Nikon, 11&&DLSR, 12&&models]

    return int(offsets[0][0]), int(offsets[-1][0]) # obtain start token and end token for each span, [('5', '幸'), ('6', '福'), ('7', '使'), ('8', '者')]--> (5,8)


def process_line(args, text_line, label_line, tokenizer: AutoTokenizer, sample_id, logits):
    text = text_line.split('\t')[0].strip() # text_line:当前行， text：sentence
    # text_prompt = " subject is [MASK], object is [MASK]."
    # text = text + text_prompt

    have_triples = int(text_line.split('\t')[1]) # obtain the label is comparative (1) or no-comparative (0)

    re_result = re.findall(r'\[\[(.*?)\];\[(.*?)\];\[(.*?)\];\[(.*?)\];\[(.*?)\]\]', label_line)
    # label_line--> re_result:去除原始数据中的[]，以及;
    raw_labels: List = [[x for x in y] for y in re_result] #一个样本存放在一个list中 
    # List of triples for a sentence
     
    tokens_output = tokenizer(text, max_length=args.max_text_length - 1, pad_to_max_length=True) # input_ids, token_type_ids, attention_mask
    # unused1:表示据识（拒绝识别），即query可能未是被出五元组， unused1表示0，
    token_ids = [tokenizer.convert_tokens_to_ids('[unused1]')] + tokens_output['input_ids']
    sample = {'token_ids': token_ids, 'labels': [], 'sample_id': sample_id}


    if have_triples == 0:
        cur_logits = logits[str(token_ids)]# 将保存的logits加载进sample
        sample['logits'] = cur_logits
        return sample

    for tri in raw_labels:
        # tri: [sub, obj, aspect, opinion, sentiment]
        # data_path:主要是为了区分中英文数据集，原始数据中，存放格式不一致
        sub_offset = proc_raw_offset(tri[0], text, args.data_path) # pro_raw_offset: obtain the start offset and end offset for each element
        obj_offset = proc_raw_offset(tri[1], text, args.data_path)
        aspect_offset = proc_raw_offset(tri[2], text, args.data_path)
        view_offset = proc_raw_offset(tri[3], text, args.data_path)

        sentiment_label = have_triples * EMO_MAP[int(tri[4])]

        if 'smartphone' in args.data_path:
            sample['labels'].append({
                'sub_start_index': tokens_output.word_to_tokens(sub_offset[0]).start,
                'sub_end_index': tokens_output.word_to_tokens(sub_offset[1]).end - 1,
                'obj_start_index': tokens_output.word_to_tokens(obj_offset[0]).start,
                'obj_end_index': tokens_output.word_to_tokens(obj_offset[1]).end - 1,
                'aspect_start_index': tokens_output.word_to_tokens(aspect_offset[0]).start,
                'aspect_end_index': tokens_output.word_to_tokens(aspect_offset[1]).end - 1,
                'opinion_start_index': tokens_output.word_to_tokens(view_offset[0]).start,
                'opinion_end_index': tokens_output.word_to_tokens(view_offset[1]).end - 1,
                'relation': sentiment_label
            })
            cur_logits = logits[str(token_ids)]# 将保存的logits加载进sample
            # stop()
            sample['logits'] = cur_logits

        else:
            sample['labels'].append({
                'sub_start_index': tokens_output.char_to_token(sub_offset[0]) + 1,
                'sub_end_index': tokens_output.char_to_token(sub_offset[1]) + 1,
                'obj_start_index': tokens_output.char_to_token(obj_offset[0]) + 1,
                'obj_end_index': tokens_output.char_to_token(obj_offset[1]) + 1,
                'aspect_start_index': tokens_output.char_to_token(aspect_offset[0]) + 1,
                'aspect_end_index': tokens_output.char_to_token(aspect_offset[1]) + 1,
                'opinion_start_index': tokens_output.char_to_token(view_offset[0]) + 1,
                'opinion_end_index': tokens_output.char_to_token(view_offset[1]) + 1,
                'relation': sentiment_label
            })

            cur_logits = logits[str(token_ids)]# 将保存的logits加载进sample
            sample['logits'] = cur_logits
    return sample
        
def get_logits(args):
    # 加载存储好的logits文件,并获取text对应的logits
    if args.data_type == "tuple":
        name_str = "tuple_logits.json"
    elif args.data_type == "quintuple":
        name_str = "quintuple_logits.json"
    path1 = args.output_path
    path2 = "ckpt-five_nodrop"
    fr_path = os.path.join(path1, path2, name_str)

    file = open(fr_path,'r')
    cur_dict = json.loads(file.read())

    return cur_dict


def load_data_logits(args, mode: str):
    logits_dump = get_logits(args)
    raw_data = []
    with open(os.path.join(args.data_path, f'{mode}.txt'), 'r') as f:
        for line in f:
            raw_data.append(line)
    all_samples = []
    line_id, i = 0, 0
    text_line, label_line = '', ''
    for line_id in trange(len(raw_data), desc=f'processing data for mode {mode}'):
        cur_line = raw_data[line_id]
        if len(cur_line.split('\t')) != 2:
            label_line += '\n' + cur_line
        else:
            # a new text line, so push the last text and update text_line
            if text_line != '':
                all_samples.append(process_line(args, text_line, label_line, args.tokenizer, i, logits_dump))
                i += 1
            text_line = cur_line
            label_line = ''
    
    all_samples.append(process_line(args, text_line, label_line, args.tokenizer, i, logits_dump))
    return all_samples

def build_collate_fn_logits(args):
    def collate_fn(batch):
        input_ids = torch.tensor([sample['token_ids'] for sample in batch], device=args.device, dtype=torch.long) # input_ids: bsz, max_len
        seq_ids = [sample['sample_id'] for sample in batch]
        # stop()
        batch_rel_logits = torch.tensor([sample['logits']['pred_rel_logits'] for sample in batch], device=args.device, dtype=torch.float32) 
        batch_sub_start_logits = torch.tensor([sample['logits']['sub_start_logits'] for sample in batch], device=args.device, dtype=torch.float32) 
        batch_sub_end_logits = torch.tensor([sample['logits']['sub_end_logits'] for sample in batch], device=args.device, dtype=torch.float32) 
        batch_obj_start_logits = torch.tensor([sample['logits']['obj_start_logits'] for sample in batch], device=args.device, dtype=torch.float32) 
        batch_obj_end_logits = torch.tensor([sample['logits']['obj_end_logits'] for sample in batch], device=args.device, dtype=torch.float32) 
        batch_asp_start_logits = torch.tensor([sample['logits']['aspect_start_logits'] for sample in batch], device=args.device, dtype=torch.float32) 
        batch_asp_end_logits = torch.tensor([sample['logits']['aspect_end_logits'] for sample in batch], device=args.device, dtype=torch.float32) 
        batch_op_start_logits = torch.tensor([sample['logits']['opinion_start_logits'] for sample in batch], device=args.device, dtype=torch.float32) 
        batch_op_end_logits = torch.tensor([sample['logits']['opinion_end_logits'] for sample in batch], device=args.device, dtype=torch.float32) 

        batch_logits = { 
            'pred_rel_logits': batch_rel_logits, # bsz, q_num, num_class: (4,60,5)
            'sub_start_logits': batch_sub_start_logits,  # bsz, q_num, seq_len: (4,60,512)
            'sub_end_logits': batch_sub_end_logits,
            'obj_start_logits': batch_obj_start_logits, 
            'obj_end_logits': batch_obj_end_logits,
            'aspect_start_logits': batch_asp_start_logits, 
            'aspect_end_logits': batch_asp_end_logits,
            'opinion_start_logits': batch_op_start_logits, 
            'opinion_end_logits': batch_op_end_logits,
        }
        labels = []
        for sample in batch:
            target = {
                'sub_start_index': [],
                'sub_end_index': [],
                'obj_start_index': [],
                'obj_end_index': [],
                'aspect_start_index': [],
                'aspect_end_index': [],
                'opinion_start_index': [],
                'opinion_end_index': [],
                'relation': [],
            }
            for tri in sample['labels']:
                for k in tri:
                    target[k].append(tri[k])

            for k in target:
                assert len(target[k]) <= args.num_generated_triples  # num_generated_triples: default=10
                target[k] = torch.tensor(target[k], device=args.device, dtype=torch.long)
            labels.append(target)

        return input_ids, labels, seq_ids, batch_logits
    return collate_fn