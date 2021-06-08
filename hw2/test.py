import numpy as np
from numpy.testing._private.utils import import_nose
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
from transformers import RobertaTokenizer, RobertaModel
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import tqdm
import os
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='0',
                    help='device to run the program')
parser.add_argument('--lr', type=float, default=1e-6,
                    help='learning rate')
parser.add_argument('--batch_size', type=int, default=16,
                    help='learning rate')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

extractor = RobertaModel.from_pretrained('roberta-large')

class mybert(nn.Module):
    def __init__(self,extractor,num_labels) -> None:
        super().__init__()
        self.extractor = extractor
        self.fc1 = nn.Linear(128*1024,128)
        self.fc2 = nn.Linear(128,num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids,attention_mask):
        try:
            output = self.extractor(input_ids=input_ids,attention_mask = attention_mask)
            x = output.last_hidden_state

            output = output.last_hidden_state.reshape(x.shape[0],128*1024)
            # print(output.shape)
            output = self.dropout(F.relu(self.fc1(output)))

            return F.softmax(self.fc2(output))
        except:
            print(x.shape)

    def num_features(self,x):
        size = x.size()[1:] #计算除了batch大小的所有特征数用于fully connect
        num_fea = 1
        for s in size:
            num_fea *= s
        return num_fea

model = mybert(extractor=extractor,num_labels=5)
model.to(device)

train_data_input = []
train_data_mask = []
train_label = []
with open('./train.txt','r',encoding='utf-8')as f:
    lines = f.readlines()
name2label = {'OBJECTIVE':0,'METHODS':1,'RESULTS':2,'CONCLUSIONS':3,'BACKGROUND':4}

for i,line in enumerate(lines):
    if len(line.split('\t'))<2:
        continue
    label,data = line.split('\t')[0],line.split('\t')[1].replace('\n','')

    temp=tokenizer.encode_plus(data,add_special_tokens=True,max_length=128,pad_to_max_length=True)

    train_data_input.append(temp['input_ids'])
    train_data_mask.append(temp['attention_mask'])
    train_label.append(name2label[label])

test_data_input = []
test_data_mask = []
test_label = []
with open('./test.txt','r',encoding='utf-8')as f:
    lines = f.readlines()

for i,line in enumerate(lines):
    if len(line.split('\t'))<2:
        continue
    label,data = line.split('\t')[0],line.split('\t')[1].replace('\n','')

    temp=tokenizer.encode_plus(data,add_special_tokens=True,max_length=128,pad_to_max_length=True)

    test_data_input.append(temp['input_ids'])
    test_data_mask.append(temp['attention_mask'])
    test_label.append(name2label[label])

from torch.utils.data import DataLoader
# print(train_data_input)

# print(train_data_mask)
train_data = torch.utils.data.dataset.TensorDataset(torch.tensor(train_data_input),torch.tensor(train_data_mask),torch.tensor(train_label))

test_data = torch.utils.data.dataset.TensorDataset(torch.tensor(test_data_input),torch.tensor(test_data_mask),torch.tensor(test_label))

train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size)


criterion = nn.CrossEntropyLoss()

def train(model,epoch,train_data_loader,test_data_loader):
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=epoch*len(train_data_loader))
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    for _ in range(epoch):
        print('epoch:',_)
        model.train()
        correct = 0
        running_loss = 0.
        for (data,mask,label) in tqdm.tqdm(train_data_loader):

            data,mask ,label = torch.tensor(data).to(device),torch.tensor(mask).to(device),torch.tensor(label).to(device)

            outputs = model(input_ids=data,attention_mask = mask)
            loss = criterion(outputs,label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            model.zero_grad()
            running_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            correct += (predicted == label).sum().item()            

        print('train ACC:',correct/(len(train_data_loader)*args.batch_size))
        print('train loss:',running_loss)
        acc,predict = test(model,test_data_loader)
        print(classification_report(test_label,predict,digits=6))
        # # # torch.save(model,'./model_every_epoch.pkl')
        # # print('model saved')


def test(model,test_data_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    predict = []
    with torch.no_grad():
        correct = 0
        loss = 0
        for idx,(data,mask,label) in enumerate(test_data_loader):
            data,mask ,label = torch.tensor(data).to(device),torch.tensor(mask).to(device),torch.tensor(label).to(device)

            outputs = model(input_ids=data,attention_mask = mask)

            loss += criterion(outputs,label).item()

            _,predicted = torch.max(outputs,1)
            predict.extend(predicted.cpu().numpy().tolist())
            correct += (predicted == label).sum().item()
        print('test ACC:', correct /(len(test_data_loader)*args.batch_size))
        print('test loss:',loss)
        model.zero_grad()
        
    return correct /(len(test_data_loader)*args.batch_size),predict

train(model,epoch = 15,train_data_loader=train_data_loader,test_data_loader=test_data_loader)