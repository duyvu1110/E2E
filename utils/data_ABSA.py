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

# dic = {
#    -1: 'worse', 1
#    0: 'equal',  2
#    1: 'better',  3
#    2: 'different'  4
# }
EMO_MAP = {
    'NEG': 1, 
    'NEU': 2, 
    'POS': 3 
}

def proc_raw_offset(offset_spans: list, dataset_name):
    if offset_spans == '':
        return (0, 0)
    if 'lap' in dataset_name or 'res' in dataset_name:
        if len(offset_spans) == 1: # 判断list中有几个数字
            start_position = offset_spans[0]
            end_position = offset_spans[0]
        else: # len(span)>1
            start_position = offset_spans[0]
            end_position = offset_spans[-1] # 取结束位置,可以取到

    elif 'zhijiang' in dataset_name:
        start_position = offset_spans[0]
        end_position = offset_spans[-1] - 1


    return  start_position, end_position# 返回的是一个span的start_position,和end_position


def process_line_absa(args, text_line, tokenizer:AutoTokenizer, sample_id):
    text = text_line.split('####')[0].strip()
    try:
        all_labels = text_line.split('####')[1] # e.g., [([1], [4], 'POS'), ([1], [6], 'POS')]
    except:
        print(text_line)
        stop()
        
    raw_labels = eval(all_labels) # 一个样本的all labels存放在一个list中
    tokens_output = tokenizer(text, max_length=args.max_text_length - 1, pad_to_max_length= True)
    token_ids = [tokenizer.convert_tokens_to_ids('[unused1]')] + tokens_output['input_ids']
    sample = {'token_ids':token_ids, 'labels':[],'sample_id':sample_id}

    for tri in raw_labels: # raw_labels:表示的是一个text对应的所有三元组,tri表示的是其中的一个
        aspect_offset = proc_raw_offset(tri[0], args.data_path)
        opinion_offset = proc_raw_offset(tri[1], args.data_path)
        sentiment_label = EMO_MAP[tri[2]]

        if 'lap' in args.data_path or 'res' in args.data_path: # English
            sample['labels'].append({
                'aspect_start_index':tokens_output.word_to_tokens(aspect_offset[0]).start + 1,
                'aspect_end_index':tokens_output.word_to_tokens(aspect_offset[1]).end,
                'opinion_start_index':tokens_output.word_to_tokens(opinion_offset[0]).start + 1,
                'opinion_end_index':tokens_output.word_to_tokens(opinion_offset[1]).end,
                'relation':sentiment_label
            })

        elif 'zhijiang' in args.data_path: # Chinese
            sample['labels'].append({
                'aspect_start_index': tokens_output.char_to_token(aspect_offset[0]) + 1,
                'aspect_end_index': tokens_output.char_to_token(aspect_offset[1]),
                'opinion_start_index': tokens_output.char_to_token(opinion_offset[0]) + 1,
                'opinion_end_index': tokens_output.char_to_token(opinion_offset[1]),
                'relation': sentiment_label
            })

    return sample

def load_data_absa(args, mode: str):
    raw_data = []
    with open(os.path.join(args.data_path, f'{mode}.txt'), 'r') as f:
        for line in f:
            raw_data.append(line)
    all_samples = []
    line_id, i = 0, 0
    for line_id in trange(len(raw_data), desc=f'processing data for mode {mode}'):
        cur_line = raw_data[line_id]
        if len(cur_line) != 0:
            all_samples.append(process_line_absa(args, cur_line, args.tokenizer, i))
            i += 1
    all_samples.append(process_line_absa(args, cur_line, args.tokenizer, i))
    return all_samples

def build_collate_fn_absa(args):
    def collate_fn(batch):
        input_ids = torch.tensor([sample['token_ids'] for sample in batch], device=args.device, dtype=torch.long)
        seq_ids = [sample['sample_id'] for sample in batch]
        labels = []
        for sample in batch:
            target = {
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
        return input_ids, labels, seq_ids
    return collate_fn