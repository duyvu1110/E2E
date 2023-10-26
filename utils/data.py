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
    2: 4
}

def pass_offset(data_path, offset):
    """
    Set the offset in english dataset(Camera-COQE) starts from 0.
    """
    if 'Camera' in data_path:
        return offset - 1
    else:
        return pass_offset

def get_token_span(word_idx, tok_to_orig_index):
    #  word_idx: 是text按空格划分之后的word,分词之后对应的index, 传入的是offset的数字， 加1？
    """
    功能: 根据word索引得到token的span (start, end), 若先将'[unused1]'加入text,此处可以直接用sub_offset
    """
    start , end = -1, -1
    for idx, i in enumerate(tok_to_orig_index):
        if i == word_idx:
            if start == -1:
                start = idx
            end = idx
    return start , end # 

def words_to_tokens(tokenizer, doc_tokens, max_text_len): # doc_tokens,表示text按空格划分之后的list
    """
    功能: 将word list 变成 token list, word_list表示
    tok_to_orig_index: token到原来word位置的映射
    all_doc_tokens: tokenizer之后的tokens
    """
    tok_to_orig_index = []
    all_doc_tokens = ['[unused1]', '[CLS]']
    for (i, token) in enumerate(doc_tokens): # token-->word
        # orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            # pass
            tok_to_orig_index.append(i) # i 实际word的index
            all_doc_tokens.append(sub_token) # sub_token-->token

    tok_to_orig_index = [0, 1]+[num + 2 for num in tok_to_orig_index] # 添加cls
    tok_to_orig_index.append(tok_to_orig_index[-1] + 1) # 添加sep
    all_doc_tokens.append('[SEP]')
    assert len(tok_to_orig_index) == len(all_doc_tokens), "length is not equal" # 断言判断，是否相等，不相等，则报错
    # pad到最大长度
    for i in range(max_text_len - len(tok_to_orig_index)):
        tok_to_orig_index.append(0) # 补齐到最大长度
        all_doc_tokens.append('[PAD]')
    # stop()
    return tok_to_orig_index, all_doc_tokens # tok_to_orig_index: word_list按空格划分的序号，0， 1，2，3，3……
# all_doc_tokens： 表示的是对text进行tokenizer的结果


def proc_raw_offset(offset_spans: str, text, data_path):
    if offset_spans == '':
        # use 1 to denotes the empty span
        return (0, 0)
    # 7&&all 8&&of 9&&the 10&&Nikon 11&&DLSR 12&&models
    if 'Camera' in data_path:
        offsets = re.findall(r'([0-9]+)&&(\S+)', offset_spans)
    else:
        offsets = re.findall(r'([0-9]+)&(\S+)', offset_spans) # type(offset_spans):str
    # [7&&all, 8&&of, 9&&the, 10&&Nikon, 11&&DLSR, 12&&models]

    return int(offsets[0][0]), int(offsets[-1][0]) # obtain start token and end token for each span, [('5', '幸'), ('6', '福'), ('7', '使'), ('8', '者')]--> (5,8)
    # return int(offsets[0][0]) -1, int(offsets[-1][0]) -1 # 等同于Camera从0开始计数


def process_line(args, text_line, label_line, tokenizer: AutoTokenizer, sample_id):
    text = text_line.split('\t')[0].strip() # text_line:当前行, text：sentence
    have_triples = int(text_line.split('\t')[1]) # obtain the label is comparative (1) or no-comparative (0)
    re_result = re.findall(r'\[\[(.*?)\];\[(.*?)\];\[(.*?)\];\[(.*?)\];\[(.*?)\]\]', label_line)
    # label_line--> re_result:去除原始数据中的[]，以及;
    raw_labels: List = [[x for x in y] for y in re_result] #一个样本存放在一个list中 
    # List of triples for a sentence
    if 'Camera' in args.data_path:
        token_offset, tokens = words_to_tokens(tokenizer, text.split(" "), args.max_text_length)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
    ######## xu
        tokens_output = tokenizer(text, max_length=args.max_text_length - 1, pad_to_max_length=True) # input_ids, token_type_ids, attention_mask
        # unused1:表示据识（拒绝识别），即query可能未是被出五元组， unused1表示0，
        token_ids = [tokenizer.convert_tokens_to_ids('[unused1]')] + tokens_output['input_ids']
        tokens_output['input_ids'] = token_ids
    # stop()
    sample = {'token_ids': token_ids, 'labels': [], 'sample_id': sample_id}
    # stop()
    if have_triples == 0:
        return sample

    for tri in raw_labels:
        sub_offset = proc_raw_offset(tri[0], text, args.data_path) # pro_raw_offset: obtain the start offset and end offset for each element
        obj_offset = proc_raw_offset(tri[1], text, args.data_path)
        aspect_offset = proc_raw_offset(tri[2], text, args.data_path)
        view_offset = proc_raw_offset(tri[3], text, args.data_path)
        sentiment_label = have_triples * EMO_MAP[int(tri[4])]

        if 'smartphone' in args.data_path:
            sample['labels'].append({
                # 'sub_start_index': tokens_output.word_to_tokens(sub_offset[0]).start,
                'sub_start_index': get_token_span(sub_offset[0]+1, token_offset)[0],
                'sub_end_index': get_token_span(sub_offset[1] +1, token_offset)[1],
                'obj_start_index': get_token_span(obj_offset[0] +1, token_offset)[0],
                'obj_end_index': get_token_span(obj_offset[1] +1, token_offset)[1],
                'aspect_start_index': get_token_span(aspect_offset[0] +1, token_offset)[0],
                'aspect_end_index': get_token_span(aspect_offset[1] +1, token_offset)[1],
                'opinion_start_index': get_token_span(view_offset[0] +1, token_offset)[0],
                'opinion_end_index': get_token_span(view_offset[1] +1, token_offset)[1],
                'relation': sentiment_label,

            })
            # word_to_tokens()得到的是start,end位置， char_to_token得到的是一个固定的值
        else:
            sub_offset = proc_raw_offset(tri[0], text, args.data_path) # pro_raw_offset: obtain the start offset and end offset for each element
            obj_offset = proc_raw_offset(tri[1], text, args.data_path)
            aspect_offset = proc_raw_offset(tri[2], text, args.data_path)
            view_offset = proc_raw_offset(tri[3], text, args.data_path)
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
            # stop()
    return sample
        

def load_data(args, mode: str):
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
                all_samples.append(process_line(args, text_line, label_line, args.tokenizer, i))
                i += 1
            text_line = cur_line
            label_line = ''
    
    all_samples.append(process_line(args, text_line, label_line, args.tokenizer, i))
    # stop()
    return all_samples

def build_collate_fn(args):
    def collate_fn(batch):
        input_ids = torch.tensor([sample['token_ids'] for sample in batch], device=args.device, dtype=torch.long)
        seq_ids = [sample['sample_id'] for sample in batch]
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
        return input_ids, labels, seq_ids
    return collate_fn