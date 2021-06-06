import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as ppb
import warnings
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings('ignore')
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='0',
                    help='device to run the program')
parser.add_argument('--lr', type=float, default=1e-6,
                    help='learning rate')
parser.add_argument('--batch_size', type=int, default=4,
                    help='learning rate')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#data read
train_data = pd.read_csv('./train.tsv',delimiter='\t',header=None)
test_data = pd.read_csv('./test.tsv',delimiter='\t',header=None)[:1820]



train_data_input = []
train_data_mask = []
train_padded = train_data[0].apply(
    (lambda x: tokenizer.encode_plus(x, add_special_tokens=True,max_length=512, pad_to_max_length=True)))

for i in range(train_padded.shape[0]):
    train_data_input.append(train_padded[i]['input_ids'])
    train_data_mask.append(train_padded[i]['attention_mask'])
train_labels = train_data[1]

test_padded = test_data[0].apply(
    (lambda x: tokenizer.encode_plus(x, add_special_tokens=True,max_length=512, pad_to_max_length=True)))

test_data_input = []
test_data_mask = []
for i in range(test_padded.shape[0]):
    test_data_input.append(test_padded[i]['input_ids'])
    test_data_mask.append(test_padded[i]['attention_mask'])
test_labels = test_data[1]



#load model
model_path = './model_every_epoch.pkl'
if os.path.exists(model_path):
    model = torch.load(model_path)
    print('model loaded')
else:
    model = RobertaForSequenceClassification.from_pretrained('roberta-large')
    print('model created')
model.to(device)

def train(model,epoch,train_data_loader,test_data_loader):
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=epoch*len(train_data_loader))
    model.zero_grad()
    for _ in range(epoch):
        print('epoch:',_)
        model.train()
        correct = 0
        running_loss = 0.
        for idx,(data,mask,label) in enumerate(train_data_loader):

            data,mask ,label = torch.tensor(data).to(device),torch.tensor(mask).to(device),torch.tensor(label).to(device)
            print(data.shape)
            print(mask.shape)
            outputs = model(input_ids=data,attention_mask = mask,labels=label)

            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

            scheduler.step()

            model.zero_grad()

            _,predicted = torch.max(logits,1)
            correct += (predicted == label).sum().item()            

        print('ACC:',correct/(len(train_data_loader)*args.batch_size))
        print('loss:',running_loss)
        acc = test(model,test_data_loader)
        # # torch.save(model,'./model_every_epoch.pkl')
        # print('model saved')


def test(model,test_data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        loss = 0
        for idx,(data,mask,label) in enumerate(test_data_loader):
            data,mask ,label = torch.tensor(data).to(device),torch.tensor(mask).to(device),torch.tensor(label).to(device)

            outputs = model(input_ids=data,attention_mask = mask,labels=label)

            loss += outputs.loss.item()
            logits = outputs.logits
            _,predicted = torch.max(logits,1)
            correct += (predicted == label).sum().item()
        print('ACC:', correct /(len(test_data_loader)*args.batch_size), 'Running_loss:',loss)
        model.zero_grad()
        
    return correct /(len(test_data_loader)*args.batch_size)

from torch.utils.data import DataLoader

train_data = torch.utils.data.dataset.TensorDataset(torch.tensor(train_data_input),torch.tensor(train_data_mask),torch.tensor(train_labels))

test_data = torch.utils.data.dataset.TensorDataset(torch.tensor(test_data_input),torch.tensor(test_data_mask),torch.tensor(test_labels))


train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

train(model,epoch = 10,train_data_loader=train_data_loader,test_data_loader=test_data_loader)