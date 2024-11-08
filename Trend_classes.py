def import_libraries():
    import pandas as pd, random,json
    import datetime,subprocess,scipy, os, argparse,sys, torch,re 
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from scipy.spatial import distance
    import seaborn as sns
    from alive_progress import alive_bar
    from torchmetrics.functional import pearson_corrcoef
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.nn.utils import prune
    from torch import optim
    '''
        pip install dash_bio
    '''
    return [pd,datetime,subprocess,scipy,os,argparse,sys,torch,np,plt,train_test_split,distance,sns,alive_bar,pearson_corrcoef,re,nn,F,DataLoader,ReduceLROnPlateau,prune,random,json]

pd,datetime,subprocess,scipy,os,argparse,sys,torch,np,plt,train_test_split,distance,sns,alive_bar,pearson_corrcoef,re,nn,F,DataLoader,ReduceLROnPlateau,prune,random,json=import_libraries()


from sklearn.model_selection import GridSearchCV
import re,json,argparse,csv,os,sys,pickle
import pandas as pd, numpy as np
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sns.set_context('paper')




class Time_mapper():
    def __init__(self,args) -> None:
        self.args=args
        ## Set output directory
        self.outdir=args.dir if args.dir[-1]!='/' else args.dir[:-1]
        try:
            os.mkdir(self.outdir)
        except Exception:
            pass
        ## Sniff Bigtable.csv for separator ,/\t/; etc
        with open(args.table, 'r') as csvfile:
            sample = csvfile.read(128)  
            # Use the Sniffer to Detect the separator
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            # Retrieve the Detected Separator
            separator = dialect.delimiter
        self.bigtable=pd.read_csv(args.table,sep=separator,header=0,index_col=0)
        # Normalize bigtable
        self.bigtable=self.bigtable.apply(lambda x:x/x.sum(),axis=0)



        ## Sniff Mapping.csv for separator ,/\t/; etc
        with open(args.mapping, 'r') as csvfile:
            sample = csvfile.read(512)  
            # Use the Sniffer to Detect the separator
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            # Retrieve the Detected Separator
            separator = dialect.delimiter
        self.maptable=pd.read_csv(args.mapping,sep=separator,header=0,index_col=0)
        found_samples=set(self.maptable.index).intersection(set(self.bigtable.columns))
        self.maptable=self.maptable.loc[[i for i in found_samples]]
        self.bigtable=self.bigtable[[i for i in found_samples]]
        self.timelist=args.Timepoint_series.split(',') if args.Timepoint_series else sorted(list(self.maptable['Timepoint'].unique()))
        
        self.maptable['Timepoint_listed']= pd.Categorical(self.maptable['Timepoint'], categories=self.timelist, ordered=True)
        self.maptable=self.maptable.sort_values('Timepoint_listed')
        self.taxalist=self.bigtable.index
        print (self.maptable)
        self.Per_sample()
        self.Per_Timepoint()
        self.Directions()
        self.Delta_samples()

    def Per_sample(self,):
        self.sample_otus,self.sample_data={},{}
        for sample in self.maptable['Sample'].unique():
            self.sample_data[sample]=self.maptable[self.maptable['Sample']==sample].sort_values('Timepoint_listed')
            self.sample_otus[sample]=self.bigtable[self.maptable[self.maptable['Sample']==sample].sort_values('Timepoint_listed').index]
            self.sample_data[sample].to_csv(f'{self.outdir}/{sample}_metadata.tsv',header=True,index=True,sep='\t')
            self.sample_otus[sample].to_csv(f'{self.outdir}/{sample}_OTUs.tsv',header=True,index=True,sep='\t')

    def Per_Timepoint(self,):
        self.timepoint_otus,self.timepoint_data={},{}
        for tt in self.maptable['Timepoint'].unique():
            try:
                tp=tt.item()
            except Exception:
                tp=tt
            self.timepoint_data[tp]=self.maptable[self.maptable['Timepoint']==tp]
            self.timepoint_otus[tp]=self.bigtable[self.maptable[self.maptable['Timepoint']==tp].index]
            self.timepoint_data[tp].to_csv(f'{self.outdir}/Timepoint_{tp}_metadata.tsv',header=True,index=True,sep='\t')
            self.timepoint_otus[tp].to_csv(f'{self.outdir}/Timepoint_{tp}_OTUs.tsv',header=True,index=True,sep='\t')
            
    def Directions(self,):
        self.directed_metadata,self.directed_otus={},{}
        
        for sample in self.maptable['Sample'].unique():
            
            self.directed_metadata[sample]={'Initial':{},'Target':{}}
            self.directed_otus[sample]={'Initial':{},'Target':{}}

            self.sample_data[sample]=self.maptable[self.maptable['Sample']==sample].sort_values('Timepoint_listed')

            self.directed_metadata[sample]['Initial']=self.maptable[self.maptable['Sample']==sample].iloc[0,:]
            self.directed_metadata[sample]['Target']=self.maptable[self.maptable['Sample']==sample].iloc[-1,:]
            self.directed_otus[sample]['Initial']=self.bigtable[self.maptable[self.maptable['Sample']==sample].iloc[0,:].name]
            self.directed_otus[sample]['Target']=self.bigtable[self.maptable[self.maptable['Sample']==sample].iloc[-1,:].name]
            
            if self.sample_data[sample].shape[0]>2:
                self.directed_metadata[sample]['Intermediate']=self.maptable[self.maptable['Sample']==sample].iloc[1:-1,:]
                self.directed_otus[sample]['Intermediate']=self.bigtable[self.maptable[self.maptable['Sample']==sample].iloc[1:-1,:].index] 
        #self.paired_data={sample:pd.concat((self.sample_data[sample].index[indexaki],self.sample_data[sample].index[indexaki+1]),axis=1) for indexaki in range(len(self.sample_data[sample].index)-1)] for sample in self.sample_data.keys()}
        self.input_dimensions=self.directed_otus[sample]['Initial'].shape[0]-1
        
    def Delta_samples(self,):
        self.paired_samples={}
        self.later_points=set([])
        for sample in self.sample_otus:       
            for tp in range(self.sample_data[sample].shape[0]-1):
                self.paired_samples[f"{sample}_DTP_{self.sample_data[sample]['Timepoint'][tp]}_{self.sample_data[sample]['Timepoint'][tp+1]}"]={'Initial':self.bigtable[self.sample_data[sample].iloc[tp].name],'Target':self.bigtable[self.sample_data[sample].iloc[tp+1].name]}
                if self.sample_data[sample]['Timepoint'][tp]!=self.timelist[0]:
                    self.later_points.add(f"{sample}_DTP_{self.sample_data[sample]['Timepoint'][tp]}_{self.sample_data[sample]['Timepoint'][tp+1]}")

class Microbe_trend_predictor(nn.Module):
    def __init__(self, num_layers,activation,input_dim):
        super(Microbe_trend_predictor, self).__init__()
        self.num_layers=num_layers
        self.activation=activation if activation else 'ReLU'

        self.model = nn.ModuleList()

        
        for i in range(num_layers-1):
            size = round(input_dim/2)
            self.model.append(nn.Linear(input_dim, size,bias=False))
            input_dim = size
        
        self.model.append(nn.Linear(input_dim,3,bias=False))
        #self.model.append(nn.Sigmoid())
    
    def activation_function(self,x):
        if self.activation_f=='sigmoid':
            return torch.sigmoid(x)
        elif self.activation_f=='tanh':
            return torch.tanh(x)
        elif self.activation_f=='Softmax':
            return torch.softmax(x,dim=1)
        elif self.activation_f=='ReLU':
            return torch.relu(x)
        elif self.activation_f==None:
            return x
        elif self.activation_f=='Shifted_relu':
            return self.shifted_activation(x)
        


    def forward(self, x):       
        for layer in self.model:
            x=layer(x)

        return x


class Delta_Trend_Dataset(Dataset):
    def __init__(self, directed,microbe):
        self.microbe=microbe
        self.samples=list(directed.keys())
        self.data=[]
        self.consider_it=sum([sum((directed_df['Target'].loc[microbe]>0,directed_df['Initial'].loc[microbe]>0)) for directed_df in directed.values()])
       
        self.actual_values,self.fold_change={},{}
        for sample,directed_df in directed.items():
            target_df=directed_df['Target'].drop(index=microbe).T.values
            target_df=target_df/target_df.sum()
            
            initial_df=directed_df['Initial'].drop(index=microbe).T.values
            initial_df=initial_df/initial_df.sum()
            target=np.array([directed_df['Target'].loc[microbe]<directed_df['Initial'].loc[microbe],directed_df['Target'].loc[microbe]==directed_df['Initial'].loc[microbe],directed_df['Target'].loc[microbe]>directed_df['Initial'].loc[microbe]])
            self.data.append((torch.tensor(target_df-initial_df,dtype=torch.float32),torch.tensor(target,dtype=torch.float32)))
            self.actual_values[sample]={'Initial':directed_df['Initial'].loc[microbe].item(),'Target':directed_df['Target'].loc[microbe].item()}
            try:
                self.fold_change[sample]=((directed_df['Target'].loc[microbe].item()-directed_df['Initial'].loc[microbe].item())/directed_df['Initial'].loc[microbe].item())
            except Exception:
                self.fold_change[sample]=1

    def __len__(self):
        # Returns the size of the dataset
        return len(self.samples)

    def __getitem__(self, idx):

        input_vector,target=self.data[idx][0],self.data[idx][1]
        sample=self.samples[idx]
        return input_vector, target,sample
    
    def prevelance(self,):
        return (self.targets>0).sum()/self.targets.shape[0]
    def mean_abundance(self,):
        return self.targets.mean()


def Training_models(Delta_loader,optimizer,model,scheduler,criterion,epochs):
    Microbe_trends={}

    Microbe_trends['Losstrack']={}
    for epoch in range(epochs):
        for inputs,targets,sample in Delta_loader:
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        #print(f"Epoch {epoch+1}, Loss: {loss.item():.7f}")
        Microbe_trends['Losstrack'][epoch]=loss.item()
    return Microbe_trends,model

def Validation(model,Microbe_trends,Validation_loader,Validation_Delta_set,settype):
    
    model.eval()

    Microbe_trends[f'{settype}_Prediction']=[]
    Microbe_trends[f'{settype}_Targets']=[]
    Microbe_trends[f'{settype}_Accuracy']={}
    
    with torch.no_grad():
        for valinput,valtarget,valsample in Validation_loader:
            valoutputs=model(valinput)
            
            Microbe_trends[f"{settype}_Prediction"].append(valoutputs.tolist())
            Microbe_trends[f'{settype}_Targets'].append(valtarget.tolist())
            Microbe_trends[f'{settype}_Accuracy'][valsample[0]]=(torch.argmax(valoutputs)==torch.argmax(valtarget)).item()
    
    Microbe_trends[f'{settype}_Accuracy']=np.mean(list(Microbe_trends[f'{settype}_Accuracy'].values())).item()
    Microbe_trends[f'{settype}_Actual_Target_Values']=Validation_Delta_set.actual_values
    Microbe_trends[f'{settype}_Target_Fold_Change']=Validation_Delta_set.fold_change
    return Microbe_trends