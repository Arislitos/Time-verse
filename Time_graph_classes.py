import argparse,subprocess,sys,os,torch,csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from alive_progress import alive_bar
from torch.utils.data import Dataset, DataLoader
import warnings,pickle
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# Suppress only UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
## Check that the flags are there


def Node_correspondance(biomeset,graph_data,biome,outdir):
    if os.path.isfile(f'{outdir}/{biome}_mapped.tsv'):
        return pd.read_csv(f'{outdir}/{biome}_mapped.tsv',sep='\t',header=0,index_col=0)

    biome_mapped=pd.DataFrame(columns=biomeset.columns)
    with alive_bar(len(graph_data.node_names),title='Reindexing the dataframe') as bar:
        for i,node in enumerate(graph_data.node_names):
            if 'Unknown' in node:
                tosearch=node.split('__')[-1]+f";{node[0]}__"
            else:
                tosearch=node    
            tosearch=tosearch.replace("sk__","k__")
            biome_mapped.loc[node]=biomeset.iloc[biomeset.index.str.contains(tosearch)].sum(axis=0)
            bar()
    biome_mapped=biome_mapped.apply(lambda x: x/x.max())
    biome_mapped.to_csv(f'{outdir}/{biome}_mapped.tsv',sep='\t',header=True,index=True)
    return biome_mapped



class Prokaryotic_graph():
    def __init__(self,taxalist,outdir) -> None:
        try:
            os.mkdir(outdir)
        except Exception:
            pass
        self.outdir=outdir
        if os.path.isfile(outdir+'/Graph_Prokaryotes.pt') and not args.runmeagainGRaph:
            self.graph_data=torch.load(outdir+'/Graph_Prokaryotes.pt')
        else:
            diraki=os.path.dirname(os.path.realpath(__file__))
 
            with open(diraki+'/Essentials/'+taxalist,'r') as f:
                self.alltaxa=[i.replace('k__;','').strip() if ';p' in i else i.replace("k__; ","").replace("; ","").strip() for i in f.readlines()]
            self.Create_graph()
    
    
    def Create_graph(self,):
        self.unique_nodes={}
        self.unused_lines=[]
        for line in self.alltaxa:
            for node in line.split(';'):
                if len(node)>3:
                    kleidi=node
                elif len(line.split(';')[line.split(';').index(node)-1])<=3:
                    self.unused_lines.append(line)
                else:
                    kleidi=node+'Unknown_of_'+line.split(';')[line.split(';').index(node)-1] 
                self.unique_nodes[kleidi]=line
        self.node_index_map = {node: i for i, node in enumerate(self.unique_nodes.keys())}

        self.filtered_data=[i for i in self.alltaxa if i not in self.unused_lines]

        edges = []
        with alive_bar(len(self.filtered_data), title='Connecting taxa') as bar:
            for line in self.filtered_data:
                lineage=line.split(';')
                for i in range(1, len(lineage)):
                    nodeto=self.node_index_map[lineage[i-1]] if len(lineage[i-1])>3 else self.node_index_map[lineage[i-1]+'Unknown_of_'+lineage[i-2]]
                    nodefrom=self.node_index_map[lineage[i]] if len(lineage[i])>3 else self.node_index_map[lineage[i]+'Unknown_of_'+lineage[i-1]]
                    u, v = nodefrom,nodeto  # Connect only adjacent nodes
                    edges.append([u, v])
                    edges.append([v, u])
                bar()

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Step 3: Create a PyTorch Geometric graph data object
        self.graph_data = Data(edge_index=edge_index,node_names=list(self.unique_nodes.keys()))
        self.graph_data.x=torch.zeros(len(self.unique_nodes))

        # Now let's save the graph data with node labels
        torch.save(self.graph_data, f'{outdir}/Graph_Prokaryotes.pt')

    
  
class Biome_graphs(Prokaryotic_graph):
    def __init__(self,biome,biomeset,taxfile,outdir):
        super().__init__(taxfile,outdir)
        self.biome=biome
        self.biomeset=biomeset    
    
    
    def Weighting_graph(self,):
        if os.path.isfile(f'{outdir}/{self.biome}_weighted_graph.pt') and not args.runmeagain:
            self.graph_data=torch.load(f'{outdir}/{self.biome}_weighted_graph.pt')
            return self.graph_data
        if os.path.isfile(f'{outdir}/{self.biome}_mapped.tsv') and not args.runmeagain:

            self.biome_mapped=pd.read_csv(f'{outdir}/{self.biome}_mapped.tsv',sep='\t',header=0,index_col=0)
        else:
            self.biome_mapped=Node_correspondance(self.biomeset,self.graph_data,self.biome)
        
        coocurance=np.zeros((len(self.graph_data.node_names),len(self.graph_data.node_names)))
        weight_index=self.biome_mapped>0

        with alive_bar(self.biome_mapped.shape[0], title=f'Weighting on {self.biome}') as bar:
            for n,a in enumerate(self.biome_mapped.index):
                

                athetiko=weight_index[weight_index.columns[weight_index.loc[a]>0]].sum(axis=1)
                coocurance[n,]=athetiko.values
                coocurance[:,n]=athetiko.values
   
                bar()

        np.fill_diagonal(coocurance,np.max(coocurance))
        self.coocurance=coocurance/np.max(coocurance)
        self.edge_weights = [coocurance[u, v].item() for u, v in self.graph_data.edge_index.t()]
        self.graph_data.edge_attr = torch.tensor(self.edge_weights, dtype=torch.float)
    
        torch.save(self.graph_data,f'{outdir}/{self.biome}_weighted_graph.pt')
     

def node_attributes(graph_data,sample):
    graph_data.x = graph_data.x + torch.tensor(sample)
    return graph_data

   

class Time_mapper(Prokaryotic_graph):

    def __init__(self,otu_file_location,meta_file_location, outdir,taxfile) -> None:
        super().__init__(taxfile,outdir)
    ### Create output directory
        self.outdir=outdir
        
    ### Read otu table
        ## Sniff Bigtable.csv for separator ,/\t/; etc
        try:
            with open(otu_file_location, 'r') as csvfile:
                sample = csvfile.read(512)  
                # Use the Sniffer to Detect the separator
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                # Retrieve the Detected Separator
                separator = dialect.delimiter
            self.bigtable=pd.read_csv(otu_file_location,sep=separator,header=0,index_col=0)
        except Exception:
            self.bigtable=pd.read_csv(otu_file_location,sep='\t',header=0,index_col=0)
        self.bigtable.index=[i.replace(';_','; ') for i in self.bigtable.index]
        
        # Normalize bigtable
        self.bigtable=self.bigtable.apply(lambda x:x/x.sum(),axis=0)

    ### Read metadata table
        with open(meta_file_location, 'r') as csvfile:
            sample = csvfile.read(512)  
            # Use the Sniffer to Detect the separator
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            # Retrieve the Detected Separator
            separator = dialect.delimiter
        self.maptable=pd.read_csv(meta_file_location,sep=separator,header=0,index_col=0)
        self.maptable=self.maptable.drop_duplicates(keep='first')

    ### Check the correspondance on the names (first column)
        unionaki=set(self.bigtable.columns.tolist()).intersection(set(self.maptable.index.tolist()))
        self.maptable=self.maptable.loc[list(unionaki)]
        self.bigtable=self.bigtable[list(unionaki)]
        self.bigtable=self.bigtable[[i for i in self.bigtable.columns.unique()]]


        script_directory = os.path.dirname(os.path.realpath(__file__))
    ### Save variables for the channels of the GCN
        self.taxa=self.bigtable.index
        self.num_taxa=self.bigtable.shape[0]
    ### Set the succesion /order of timepoints & sort the metadata table accordingly
        try:
            self.timelist=[int(i) for i in args.Timepoint_series.split(',')]
        except Exception:
            self.timelist=args.Timepoint_series.split(',') if args.Timepoint_series else sorted(list(self.maptable['Timepoint'].unique()))
        self.maptable['Timepoint_listed']= pd.Categorical(self.maptable['Timepoint'], categories=self.timelist, ordered=True)
        self.maptable=self.maptable.sort_values('Timepoint_listed')
        self.bigtable=Node_correspondance(biomeset=self.bigtable,graph_data=self.graph_data,biome=otu_file_location.split('/')[-1].split('_')[0],outdir=self.outdir)

    def Per_Timepoint(self):
        self.timepoint_otus,self.timepoint_data={},{}
        for tt in self.maptable['Timepoint'].unique():
            try:
                tp=tt.item()
            except Exception:
                tp=tt
            self.timepoint_data[tp]=self.maptable[self.maptable['Timepoint']==tp]
            
            self.timepoint_otus[tp]=self.bigtable[self.timepoint_data[tp].index]
            self.timepoint_data[tp].to_csv(f'{self.outdir}/Timepoint_{tp}_metadata.tsv',header=True,index=True,sep='\t')
            self.timepoint_otus[tp].to_csv(f'{self.outdir}/Timepoint_{tp}_OTUs.tsv',header=True,index=True,sep='\t')
        
    def Per_sample(self,):
        self.sample_otus,self.sample_data={},{}
        for sample in self.maptable['Sample'].unique():
            self.sample_data[sample]=self.maptable[self.maptable['Sample']==sample].sort_values('Timepoint_listed')
            self.sample_otus[sample]=self.bigtable[self.maptable[self.maptable['Sample']==sample].sort_values('Timepoint_listed').index]
            #self.sample_data[sample].to_csv(f'{self.outdir}/{sample}_metadata.tsv',header=True,index=True,sep='\t')
            #self.sample_otus[sample].to_csv(f'{self.outdir}/{sample}_OTUs.tsv',header=True,index=True,sep='\t')


    def Train_test_selection(self,):
        self.Per_sample()
        self.training_set=np.random.choice(list(self.sample_otus.keys()),int(len(self.sample_otus)*.8),replace=False)
        self.test_set=[i for i in self.sample_otus.keys() if i not in self.training_set]
        self.timepoint_train={}
        self.timepoint_test={}
        for tp,meta in self.timepoint_data.items():
            
            self.timepoint_train[tp]=self.bigtable[meta[meta['Sample'].isin(self.training_set)].index]
            
            self.timepoint_test[tp]=self.bigtable[meta[~meta['Sample'].isin(self.training_set)].index]


class Timedataset(Dataset):
    def __init__(self, data_dict):
        # Flatten the dictionary into input-target pairs
        self.data=[]
        for key, dataf in data_dict.items():
            for sample in dataf.columns:
                matrix = torch.tensor(dataf[sample].to_numpy(), dtype=torch.float64)
                target = torch.tensor(key, dtype=torch.long)  # First row as target
                self.data.append([matrix, target,sample])

    def __len__(self):
        return len(self.data)  # Number of data entries

    def __getitem__(self, idx):
        return self.data[idx]


class GCNPredictorNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GCNPredictorNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16,16)
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Pooling to get a graph-level embedding
        x = global_mean_pool(x, batch)
        
        # Final classification layer
        x = self.fc(x)
        return x

def GCN_train(outdir):
    if not os.path.isfile(f'{outdir}/First_instance.pkl'):
        Instance_of_Timedataset=Time_mapper(otu_file_location, meta_file_location,outdir,taxfile)
        Instance_of_Timedataset.Per_Timepoint()

        with open(f'{outdir}/First_instance.pkl', 'wb') as file:
            pickle.dump(Instance_of_Timedataset, file)
            
    else:
        with open(f'{outdir}/First_instance.pkl', 'rb') as file:

            Instance_of_Timedataset=pickle.load(file)

    print ("Found the Instance")

    Instance_of_Timedataset.Train_test_selection()


    dataset = Timedataset(Instance_of_Timedataset.timepoint_train)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    edge_data=Instance_of_Timedataset.graph_data.edge_index
    print ("Created Dataloader")

    target_class_transformer={t:i for i,t in enumerate(Instance_of_Timedataset.timelist)}

    gcn_inputs={torch.tensor(Instance_of_Timedataset.timepoint_train[tp][key].values,dtype=torch.float32).view(-1,1):torch.tensor(target_class_transformer[tp],dtype=torch.long) for tp in Instance_of_Timedataset.timepoint_train.keys() for key in Instance_of_Timedataset.timepoint_train[tp].columns}

    data_list=[]
    for i,inputaki in enumerate(gcn_inputs.keys()):
        data_list.append(Data(x=inputaki,edge_index=edge_data,y=gcn_inputs[inputaki]))
    # Create a DataLoader to handle multiple samples

    loader = DataLoader(data_list, batch_size=1, shuffle=True)



    #Timepoint_Predictor = GCNPredictorNet(in_channels=torch.tensor(Instance_of_Timedataset.bigtable[list(targets.keys())[0]].values).shape[0], num_classes=len(Instance_of_Timedataset.timelist))
    Timepoint_Predictor = GCNPredictorNet(in_channels=1, num_classes=len(Instance_of_Timedataset.timelist))
    # Define optimizer and loss function

    optimizer = torch.optim.Adam(Timepoint_Predictor.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    Timepoint_Predictor.train()
    for epoch in range(100):
        for data in loader:  # Iterate over batches
            optimizer.zero_grad()
            out = Timepoint_Predictor(data)  # Forward pass
            loss = criterion(out, data.y)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    torch.save(Timepoint_Predictor,f'{outdir}/Timepoint_Predictor.pth')