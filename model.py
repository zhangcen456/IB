import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch
from torch.optim import Adam
import random
import os
import numpy as np
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader
from data.reader import ClassificationReader
from utils.fn import collate_fn
import argparse
from utils.attack import build_attacker,CustomModelWrapper
from textattack import Attacker,AttackArgs
from textattack.datasets import Dataset

class IB(nn.Module):
    def __init__(self,i_dim,h_dim):
        super(IB,self).__init__()
        self.encoder=nn.Linear(i_dim,2*h_dim)
        self.h_dim=h_dim

    def forward(self,embeds):
        enc=self.encoder(embeds)#[batch_size,2*h_dim]
        mu = enc[:, :self.h_dim]
        std = F.softplus(enc[:, self.h_dim:])
        sample=self.reparameter(mu,std)
        return mu,std,sample

    def reparameter(self,mu,std):
        distribution = Normal(mu, std)
        return distribution.rsample()

class ModelH(nn.Module):
    def __init__(self,h_dim,class_num,dropout,bert,device):
        super(ModelH,self).__init__()
        self.h_dim = h_dim#information bottleneck的维度
        self.bert = bert
        self.h_ib=IB(bert.config.hidden_size,h_dim)
        self.linear1=nn.Linear(bert.config.hidden_size,bert.config.hidden_size)
        self.linear2=nn.Linear(h_dim,class_num)
        self.r_mu = torch.randn(h_dim)
        self.r_mu = self.r_mu.to(device)
        self.r_mu.requires_grad = True
        self.r_std = torch.rand(h_dim)
        self.r_std = self.r_std.to(device)
        self.r_std.requires_grad = True
        self.dropout=nn.Dropout(dropout)
        self.embedding = bert.embeddings.word_embeddings.to(device)

    def forward(self,input_ids=None,token_type_ids=None,attention_mask=None,inputs_embeds=None):
        embeds=self.bert(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,inputs_embeds=inputs_embeds).last_hidden_state
        embeds=embeds[:,0,:]#embeds:[batch_size,emb_dim]
        hidden_state=self.linear1(embeds)#hidden_state:[batch_size,emb_dim]
        mu,std,sample=self.h_ib(hidden_state)
        kl=self.kl_divergence(mu,std,self.r_mu,self.r_std) #KL(p||r)
        sample=self.dropout(sample)
        logit=self.linear2(sample)
        return kl,logit

    def kl_divergence(self, mu1, std1, mu2, std2): #KL(p1||p2)
        mu2 = mu2.to(mu1.device)
        std2 = std2.to(std1.device)
        std1_2 = torch.pow(std1, 2)
        std2_2 = torch.pow(std2, 2)
        mu_2 = torch.pow((mu1 - mu2), 2)
        return torch.log(std2 / std1) - 0.5 + (mu_2 + std1_2) / std2_2 / 2

class Trainer():
    def __init__(self,h_dim,class_num,beta,dropout,bert,tokenizer,device,device_ids,lr,loadPath=None):
        self.model=ModelH(h_dim,class_num,dropout,bert,device).to(device)
        if (loadPath != None):
            self.model.load_state_dict(torch.load(loadPath + ".pkl"))
            self.model.r_mu = torch.load(loadPath + '_mu.pt')
            self.model.r_std = torch.load(loadPath + '_std.pt')
        # self.model = nn.DataParallel(self.model, device_ids)
        self.criterion=nn.CrossEntropyLoss()

        #bert参数
        self.b_optim=Adam(bert.parameters(),lr=1e-5)
        #r
        # self.r_optim=Adam([{"params":self.model.module.r_mu},{"params":self.model.module.r_std}],lr=2e-5)
        self.r_optim = Adam([{"params": self.model.r_mu}, {"params": self.model.r_std}], lr=2e-5)
        #线性层+information bottleneck
        # self.l_optim=Adam([{'params':self.model.module.h_ib.parameters()},{"params":self.model.module.linear1.parameters()},{"params":self.model.module.linear2.parameters()}],lr=2e-5)
        self.l_optim = Adam(
            [{'params': self.model.h_ib.parameters()}, {"params": self.model.linear1.parameters()},
             {"params": self.model.linear2.parameters()}], lr=2e-5)

        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta

    def acc(self,test,name=''):
        self.model.eval()
        accurate = 0
        num = 0
        for batch in test:
            batch = (
                batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device))
            kl,logit = self.model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])
            result = logit.argmax(dim=-1)  # [batch_size]
            accurate += sum(result.eq(batch[3])).item()
            num += batch[0].shape[0]
        print("{} set,classification accuracy {}".format(name,accurate / num))
        return accurate/num

    def save(self,name):
        if (isinstance(self.model, nn.DataParallel)):
            torch.save(self.model.module.state_dict(), name + ".pkl")
            torch.save(self.model.module.r_mu, name + '_mu.pt')
            torch.save(self.model.module.r_std, name + '_std.pt')
        else:
            torch.save(self.model.state_dict(), name + ".pkl")
            torch.save(self.model.r_mu, name + '_mu.pt')
            torch.save(self.model.r_std, name + '_std.pt')

    def attack(self,instances,args,h_dim,class_num,dropout,bert,device,loadPath):
        self.model=self.model=ModelH(h_dim,class_num,dropout,bert,device).to(device)
        self.model.load_state_dict(torch.load(loadPath + ".pkl"))
        self.model.r_mu = torch.load(loadPath + '_mu.pt')
        self.model.r_std = torch.load(loadPath + '_std.pt')
        self.model.eval()
        wrapper = CustomModelWrapper(self.model, self.tokenizer)
        attack = build_attacker(wrapper, args)
        attack_args = AttackArgs(num_examples=args['attack_examples'], log_to_txt=args['log_path'], csv_coloring_style="file")
        test_instances=instances
        for i in range(args['attack_times']):
            print("Attack time {}".format(i))
            test_dataset=[]
            for instance in test_instances:
                test_dataset.append((instance.text_a,int(instance.label)))
            dataset=Dataset(test_dataset,shuffle=True)
            attacker=Attacker(attack,dataset,attack_args)
            attacker.attack_dataset()

    def freelb(self,train,b,r,l,adv_init_mag=0.08,adv_steps=3,adv_lr=0.04,adv_max_norm=0.0,max_grad_norm=1.0):
        self.model.train()
        total_loss = 0
        for batch in train:
            batch = (
                batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device))
            if isinstance(self.model, torch.nn.DataParallel):
                embeds_init = self.model.module.embedding(batch[0])
            else:
                embeds_init = self.model.embedding(batch[0])  # [batch_size,input_length,embed_size]
            if adv_init_mag > 0:

                input_mask = batch[1].to(embeds_init)  # attention_mask:[batch_size,input_length]?
                input_lengths = torch.sum(input_mask, 1)  # [batch_size]
                # check the shape of the mask here..

                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(
                    2)  # [batch_size,input_length,embed_size]
                dims = input_lengths * embeds_init.size(-1)  # [batch_size]
                mag = adv_init_mag / torch.sqrt(dims)  # mag:[batch_size]
                delta = (delta * mag.view(-1, 1, 1)).detach()  # mag.view():[batch_size,1,1]
            else:
                delta = torch.zeros_like(embeds_init)

                # the main loop
            # dp_masks = None
            for astep in range(adv_steps):
                # (0) forward
                delta.requires_grad_()
                inputs_embeds = delta + embeds_init
                # inputs['dp_masks'] = dp_masks

                kl,logit = self.model(attention_mask=batch[1],token_type_ids=batch[2],inputs_embeds=inputs_embeds)
                loss = self.criterion(logit, batch[3])  # model outputs are always tuple in transformers (see doc)
                loss+=self.beta*kl.mean()
                # (1) backward
                # if isinstance(model, torch.nn.DataParallel):
                #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

                loss = loss / adv_steps

                total_loss += loss.item()

                loss.backward()

                if astep == adv_steps - 1:
                    # further updates on delta
                    break

                # (2) get gradient on delta
                delta_grad = delta.grad.clone().detach()

                # (3) update and clip
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1,
                                                                                         1)  # denorm:[batch_size,1,1]
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr * delta_grad / denorm).detach()
                if adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2,
                                            dim=1).detach()  # delta_norm:[batch_size,1,1]
                    exceed_mask = (delta_norm > adv_max_norm).to(embeds_init)  # exceed_mask:[batch_size](0 or 1)
                    reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                   1)  # reweights:[batch_size,1,1]
                    delta = (delta * reweights).detach()

                if isinstance(self.model, torch.nn.DataParallel):
                    embeds_init = self.model.module.embedding(batch[0])
                else:
                    embeds_init = self.model.embedding(batch[0])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            if (r):
                self.r_optim.step()
            if (b):
                self.b_optim.step()
            if (l):
                self.l_optim.step()
            self.r_optim.zero_grad()
            self.b_optim.zero_grad()
            self.l_optim.zero_grad()
        return total_loss

    def mask_sig(self,train,save_path):#将embedding置0衡量重要性
        self.model.eval()
        all_sentence = []
        for batch in train:
            batch = (
                batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device))
            embeds_init = self.model.embedding(batch[0])
            kl, logit, sample = self.model(attention_mask=batch[1], token_type_ids=batch[2],
                                           inputs_embeds=embeds_init)
            prob = self.softmax(logit)
            origin_class_prob=prob[0][batch[3].item()].item()
            l=[]
            for idx in range(batch[0].shape[1]):#batch[0]:[1,seq_len]
                new_embeds=embeds_init.clone()#embeds_init:[1,seq_len,emb_dim]
                zeros=torch.zeros([new_embeds.shape[-1]])
                new_embeds[0][idx]=zeros
                _, new_logit, _ = self.model(attention_mask=batch[1], token_type_ids=batch[2],
                                               inputs_embeds=new_embeds)
                new_prob=self.softmax(new_logit)
                new_class_prob=new_prob[0][batch[3].item()].item()
                token = self.tokenizer._convert_id_to_token(batch[0][0][idx].item())
                l.append("{}|{:.2e}".format(token,origin_class_prob-new_class_prob))
            all_sentence.append(" ".join(l))
        with open(save_path,'w') as f:
            for sent in all_sentence:
                f.write(sent+"\n")

def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

def run_dataset(args):
    h_dim = int(args.hidden_size)
    beta = float(args.beta)
    batch_size = int(args.batch_size)
    gpu_num = int(args.gpu_num)
    dropout = float(args.dropout)
    adv_init_mag=float(args.adv_init_mag)
    adv_steps = int(args.adv_steps)
    adv_lr = float(args.adv_lr)
    if(args.adv_max_norm==None):
        adv_max_norm=adv_init_mag
    else:
        adv_max_norm=float(args.adv_max_norm)

    model = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [i for i in range(gpu_num)]
    loadPath=args.load_path
    trainer = Trainer(h_dim, int(args.class_num), beta, dropout, model, tokenizer, device, device_ids, 2e-5, loadPath)
    reader = ClassificationReader(max_seq_len=128)

    if args.attack:
        attackMethod=args.attack_method
        if (attackMethod == None):
            attackMethod = 'textfooler'
        attack_args = {'attack_method': attackMethod, 'attack_times': 1,'attack_examples':int(args.attack_example),'modify_ratio':float(args.modify_ratio),
                'log_path': '{}_{}_{}_{}.txt'.format(loadPath, attackMethod,args.attack_dataset,args.modify_ratio)}
        attack_ins=reader.read_from_file(args.dataset_path,args.attack_dataset)
        trainer.attack(attack_ins, attack_args, h_dim, int(args.class_num), dropout, model, device, loadPath)
    elif args.test:
        acc_ins=reader.read_from_file(args.dataset_path,args.attack_dataset)
        acc=reader.get_dataset(acc_ins,tokenizer)
        acc = DataLoader(acc, batch_size=batch_size, collate_fn=collate_fn)
        trainer.acc(acc,args.attack_dataset)

    else:
        instances = reader.read_from_file(args.dataset_path,'train')
        train = reader.get_dataset(instances, tokenizer)
        train = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_ins = reader.read_from_file(args.dataset_path, 'dev')
        valid = reader.get_dataset(valid_ins, tokenizer)
        valid = DataLoader(valid, batch_size=batch_size, collate_fn=collate_fn)
        test_ins = reader.read_from_file(args.dataset_path, 'test')
        test = reader.get_dataset(test_ins, tokenizer)
        test = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn)
        max_acc = trainer.acc(valid, 'valid')
        b_start = False
        for i in range(10):
            l = i % 2 == 0
            r = i % 2 != 0
            b = i % 2 == 0 and b_start
            trainer.freelb(train, b, r, l,adv_init_mag=adv_init_mag,adv_steps=adv_steps,adv_lr=adv_lr,adv_max_norm=adv_max_norm,max_grad_norm=1.0)
            valid_acc = trainer.acc(valid, 'valid')
            if (valid_acc > max_acc):
                max_acc = valid_acc
                print("save model_acc")
                trainer.save(args.save_path+"_acc")
            trainer.acc(test, 'test')
            if (valid_acc >= 0.75):
                b_start = True
            if(args.save_epoch):
                trainer.save(args.save_path+'_epo{}'.format(i))

def str2bool(strIn):
    if strIn.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif strIn.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(strIn)
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="parameter")
    parser.add_argument('-hd','--hidden_size',default=500)
    parser.add_argument('-be','--beta',default='0.1')
    parser.add_argument('-b','--batch_size',default=64)
    parser.add_argument('-s','--seed',default=0)
    parser.add_argument('-d','--dropout',default=0.1)
    parser.add_argument('-g','--gpu_num',default=2)
    parser.add_argument('-a','--attack',type=str2bool,nargs='?',const=False)
    parser.add_argument('-l','--load',type=str2bool,nargs='?',const=False)
    parser.add_argument('-t', '--test', type=str2bool, nargs='?', const=False)
    parser.add_argument('-lp', '--load_path', default=None)
    parser.add_argument('-am', '--attack_method', default=None)
    parser.add_argument('-ai', '--adv_init_mag', default=0.08)
    parser.add_argument('-as', '--adv_steps', default=3)
    parser.add_argument('-al', '--adv_lr', default=0.04)
    parser.add_argument('-amn','--adv_max_norm',default=None)
    parser.add_argument('-ad','--attack_dataset',default='test')#attack dataset & accuracy dataset
    parser.add_argument('-ae','--attack_examples',default=1000)
    parser.add_argument('-mr','--modify_ratio',default=0.15)
    parser.add_argument('-e', '--epoch', default=10)
    parser.add_argument('-se','--save_epoch',type=str2bool,nargs='?',const=False)
    parser.add_argument('-c','--class_num',default=2)
    parser.add_argument('-dp','--dataset_path',default='dataset/sst2')
    parser.add_argument('-sp','--save_path',default='')
    args = parser.parse_args()
    seed=int(args.seed)
    set_seed(seed)

    print(args)
    run_dataset(args)
