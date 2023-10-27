import argparse, os, torch
import random
from datetime import datetime
import numpy as np
from utils.data import load_data, build_collate_fn
from utils.data_ABSA import load_data_absa, build_collate_fn_absa
from utils.data_logits import load_data_logits, build_collate_fn_logits # 只有train dataset需要 return logits
from trainer.trainer import Trainer
from trainer.trainer_logits import Trainer_logits
from models.setpred4RE import SetPred4RE
from models.setpred4RE_absa import SetPred4RE_absa
# from models.setpred4RE_ZL import SetPred4RE_ZL
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pdb import set_trace as stop


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def convert_data(data_type,file_name = ''):
    folder_path = ''
    if data_type == 'train':
        folder_path = './data/smartphone/VLSP2023_ComOM_training_v2'
        des_file = './data/smartphone/train.txt'
    if data_type == 'test':
        folder_path = './data/smartphone/VLSP2023_ComOM_testing_v2'
        des_file = './data/smartphone/test.txt'
    if data_type == 'dev':
        folder_path = './data/smartphone/VLSP2023_ComOM_dev_v2'
        des_file = './data/smartphone/dev.txt'

    
    files = os.listdir(folder_path)
    # if (data_type == 'test'):
    #     print(file_name)
    #     files.clear()
    #     files.append(file_name)
    sentences_and_content = []
    for file_name in files:
        # if file_name.startswith('train_') and file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            sections = file.read().split('\n\n')
            for section in sections:
                parts = section.split('\n')
                json_format = ""
                if len(parts) >= 2:
                    sentence = parts[0].strip()
                    sentence = sentence.split('\t')[1]
                    sentence = " ".join(sentence.split())
                    sentence += '\t' + '1'

                    contents = parts[1].strip()
                    json_contents = contents.strip().split('\n')
                    combined_format = sentence + '\n'
                    for i in range(1, len(parts)):
                        for json_content in json_contents:
                            idx_s, idx_e = (-1, -1)
                            tuples = ""
                            while True:
                                idx_s = json_content.find('[', idx_s + 1)
                                idx_e = json_content.find(']', idx_e + 1)

                                if idx_s == -1:
                                    break

                                for i in json_content[idx_s:idx_e + 1]:
                                    if i != ',':
                                        tuples += i

                                tuples += ';'

                            labels = ["DIF", "EQL", "SUP+", "SUP-", "SUP", "COM+", "COM-", "COM"]

                            for label in labels:
                                if json_content.find(label) != -1:
                                    if label == "DIF":
                                        tuples += '[' + str(-1) + ']'
                                    elif label == "EQL":
                                        tuples += '[' + str(0) + ']'
                                    elif label == "SUP+":
                                        tuples += '[' + str(1) + ']'
                                    elif label == "SUP-":
                                        tuples += '[' + str(2) + ']'
                                    elif label == "SUP":
                                        tuples += '[' + str(3) + ']'
                                    elif label == "COM+":
                                        tuples += '[' + str(4) + ']'
                                    elif label == "COM-":
                                        tuples += '[' + str(5) + ']'
                                    elif label == "COM":
                                        tuples += '[' + str(6) + ']'
                                    break

                            json_format += '[' + tuples + ']' + '\n'
                            json_format = json_format.replace('"', '')
                    combined_format += json_format[:-1]
                    sentences_and_content.append(combined_format)

                else:
                    sentence = parts[0].split('\t')
                    if len(sentence) == 1:
                        continue
                    else:
                        sentence = sentence[1]
                    sentence = " ".join(sentence.split())
                    sentence += '\t' + '0'
                    json_format = "[[];[];[];[];[]]"
                    combined_format = f"{sentence}\n{json_format}"
                    sentences_and_content.append(combined_format)
    
    with open(des_file, 'w', encoding='utf-8') as output_file:
        for item in sentences_and_content:
            output_file.write(str(item) + '\n')

if __name__ == '__main__':
    convert_data('train')    
    convert_data('dev')
    convert_data('test')
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='')
    parser.add_argument('--output_path', default='log/')
    parser.add_argument('--bert_directory', type=str, default="vinai/phobert-base-v2")
    parser.add_argument('--model_name', type=str, default="SPN")
    parser.add_argument('--num_generated_triples', type=int, default=10)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
    parser.add_argument('--na_rel_coef', type=float, default=1)
    parser.add_argument('--rel_loss_weight', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--decoder_lr', type=float, default=2e-5)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--n_best_size', type=int, default=5)
    parser.add_argument('--max_text_length', type=int, default=258)
    parser.add_argument('--max_span_length', type=int, default=10)
    parser.add_argument('--wirte_param_path', type=str, default='')
    # add
    parser.add_argument("--use_last_hidden_state", type=str, default="True")
    parser.add_argument("--encoder_kind", type=str, default='Bert') # Bert or Bilstm
    parser.add_argument('--use_fp16', type=str, default='False')
    parser.add_argument('--bilstm_input_size', help='the size of encoder embedding', type=int, default=300)
    parser.add_argument('--bilstm_hidden_size', help='the size of hidden embedding', type=int, default=512)
    parser.add_argument('--bilstm_num_layers', help='the number of layer', type=int, default=2)
    parser.add_argument('--stage', type=str, default='one')
    parser.add_argument('--method_stage', help='rhe method of the decoder to using different load', type=str, default='method_one')
    parser.add_argument('--multi_heads', type=int, help='the heads of classification', default=5) # 5 or 3
    parser.add_argument('--data_type', type=str, default="quintuple")
    args = parser.parse_args()
    # args.output_path = os.path.join(args.output_path, datetime.today().strftime("%Y-%m-%d-%H-%M-%S")+"-"+args.data_path.split('/')[1])
    args.output_path = os.path.join(args.output_path, args.model_name +"-"+ str(args.random_seed)) # three and five have the same folder name

    tokenizer = AutoTokenizer.from_pretrained(args.bert_directory, padding=True)
    args.tokenizer = tokenizer
    
    if os.path.exists(args.output_path):
        print("该文件夹已存在，请勿重复创建！")
    else:
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path,"params.txt"),"w") as fw:
        print("Start time is {}".format(datetime.today().strftime("%Y-%m-%d-%H-%M-%S")), file=fw)
        for arg in vars(args):
            print(arg, ":",  getattr(args, arg)) # show directly
            print(arg, ":",  getattr(args, arg), file=fw) # write to params_txt
    set_seed(args.random_seed)

    collate_fn = build_collate_fn(args)
    data = {
        'train': DataLoader(load_data(args, 'train'), args.batch_size, True, collate_fn=collate_fn),
        'dev': DataLoader(load_data(args, 'dev'), args.batch_size, False, collate_fn=collate_fn),
        'test': DataLoader(load_data(args, 'test'), args.batch_size, False, collate_fn=collate_fn),
    }
    model = SetPred4RE(args, 5).to(args.device)
    trainer = Trainer(model, data, args)
    trainer.train_model()

    with open(os.path.join(args.output_path,'params.txt'),"a") as f:
        print("=============================================", file=f)
        print("End time is {}".format(datetime.today().strftime("%Y-%m-%d-%H-%M-%S")), file=f)
