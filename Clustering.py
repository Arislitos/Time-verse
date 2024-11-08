from Time_classes import *

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from collections import Counter
import scipy
import scipy.stats as stats

parser = argparse.ArgumentParser()


study='MGYS0000'
study=study+'5207'

parser.add_argument("-d","--dir", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default='/home/arislit/Time-series_data/Time_verse_results/')##home/arislit/Time-series_data/Datasets/MGYS00003476/
parser.add_argument("-t","--table", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default=f'/home/arislit/Time-series_data/Datasets/{study}/{study}.tsv') ## /home/arislit/Time-series_data/Datasets/MGYS00003476/MGYS00003476_reindexed.tsv
parser.add_argument("-m","--mapping", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default=f'/home/arislit/Time-series_data/Datasets/{study}/{study}_mapping.tsv')## /home/arislit/Time-series_data/Datasets/MGYS00003476/MGYS00003476_mapping.tsv
parser.add_argument("-tp","--Timepoint", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default='0')
parser.add_argument("--settype", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default=study)

parser.add_argument("-s","--Timepoint_series", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default='0,3,7')# S00,S02,S04,S06,S08') # ''

parser.add_argument("-e","--num_training_loops", help="Do you have a As to initialise??", nargs='?', type=int, default=1)# S00,S02,S04,S06,S08') # ''


parser.add_argument("--local", help="Do you have a As to initialise??",action='store_true', default=True)

args=parser.parse_args()


adj_matrix=scipy.sparse.load_npz(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Essentials/Adjacency_matrix_backward.tsv.npz')).todense()


def criterion(s1,s2):
    return stats.spearmanr(s1,s2)


class Clustering(Time_mapper):
    def __init__(self,args):
        super().__init__(args)
        self.outdir=self.args.dir[:-1] if self.args.dir[-1]=='/' else self.args.dir
        self.outdir=self.outdir+'/'+self.args.settype
        try:
            os.mkdir(self.outdir)
        except Exception:
            pass

        self.dissimilarity_calculations()
        self.mds_all()



    def dissimilarity_calculations(self,):
        self.disimilarity={}
        self.disimilarity_p={}
        Timepoint_otus=self.Per_Timepoint()
        if not os.path.isfile(f'{self.outdir}/Disimilarity_p-values_All.tsv'):
            for timepoint,table in Timepoint_otus.items():
                self.disimilarity[timepoint]=pd.DataFrame(index=table.columns,columns=table.columns)
                self.disimilarity_p[timepoint]=pd.DataFrame(index=table.columns,columns=table.columns)
                with alive_bar(int((table.shape[1]*(table.shape[1]-1))/2),title='Spearmans correlation') as bar:
                    for indexaki,p0 in enumerate(table.columns):
                        for p1 in table.columns[indexaki+1:]:
                            self.disimilarity[timepoint].loc[p0,p1],self.disimilarity_p[timepoint].loc[p0,p1]=criterion(table[p0],table[p1])
                            self.disimilarity[timepoint].loc[p1,p0]=self.disimilarity[timepoint].loc[p0,p1]
                            self.disimilarity_p[timepoint].loc[p1,p0]=self.disimilarity_p[timepoint].loc[p0,p1]

                            bar()
                np.fill_diagonal(self.disimilarity[timepoint].values,0.0)
                np.fill_diagonal(self.disimilarity_p[timepoint].values,0.0)
                self.disimilarity[timepoint].to_csv(f'{self.outdir}/Disimilarity_{timepoint}.tsv')
                self.disimilarity_p[timepoint].to_csv(f'{self.outdir}/Disimilarity_p-values_{timepoint}.tsv')

        else:
            for timepoint in list(Timepoint_otus.keys())+['All']:
                self.disimilarity[timepoint]=pd.read_csv(f'{self.outdir}/Disimilarity_{timepoint}.tsv',header=0,index_col=0).fillna(0)


        return self.disimilarity
   
   
   
    def mds_all(self,):
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        self.mds_features = mds.fit_transform(self.disimilarity['All'])
    
    
    
    
    def original_plotting(self,):

        self.forscatterplot=pd.DataFrame(index=self.disimilarity['All'].index)
        self.forscatterplot['x']=self.mds_features[:,0]
        self.forscatterplot['y']=self.mds_features[:,1]
        self.forscatterplot['Timepoint']=Xronos2.maptable['Timepoint']
        self.forscatterplot['Sample']=Xronos2.maptable['Sample']

        relplotdata=self.forscatterplot.reset_index()
        #relplotdata=relplotdata.sort_values('Timepoint')
        #print (relplotdata)Bas E. Dutilh
        relplotdata['Timepoint']='All'
        relplotdata=pd.concat([self.forscatterplot.reset_index(),relplotdata])
        custom_palette = sns.color_palette("hsv", len(relplotdata['Sample'].unique()))
        
        relplotdata['Timepoint']=relplotdata['Timepoint'].apply(str)
        relplotdata=relplotdata.sort_values('Timepoint')
        print (sorted(relplotdata['Timepoint'].unique()))
        sns.relplot(data=relplotdata, x="x", y="y",col="Timepoint", hue="Sample",s=100,kind="scatter",palette=custom_palette,col_order=sorted(relplotdata['Timepoint'].unique()))
        plt.savefig(f'{self.outdir}/Timepoint_analysis.png')
       #plt.show()
        plt.close()

    def evaluate_clustering(self,X, labels):
        silhouette_avg = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        return np.array([silhouette_avg, davies_bouldin, calinski_harabasz])
    
    def most_frequent_highest(self,numbers):
        # Count the occurrences of each number
        count = Counter(numbers)

        # Find the maximum frequency
        max_frequency = max(count.values())

        # Get the numbers with the maximum frequency
        most_frequent_numbers = [num for num, freq in count.items() if freq == max_frequency]

        # Return the highest number among those
        return max(most_frequent_numbers)

    def optimal_clusters_calculations(self,):
        

        metrics={'kmeans':[],'gmms':[]}
        for n_clusters in range(2,int(np.sqrt(self.bigtable.shape[1]))):
            
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(self.mds_features)
            metrics['kmeans'].append(self.evaluate_clustering(self.mds_features, kmeans_labels))
            
            # GMM
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm_labels = gmm.fit_predict(self.mds_features)
            metrics['gmms'].append(self.evaluate_clustering(self.mds_features, gmm_labels))

        # Convert metrics lists to np.array

        metrics['kmeans']=np.array(metrics['kmeans'])
        metrics['gmms']=np.array(metrics['gmms'])
    
        return self.most_frequent_highest([np.argmin(i[:,0])+2 for i in metrics.values()])
    
    def plotting_clusters(self,):
        self.original_plotting()
        optimal_clusters=self.optimal_clusters_calculations()

        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(self.mds_features)

        self.forscatterplot['Cluster']=kmeans_labels

        self.forscatterplot['Cluster']=self.forscatterplot['Cluster']+1
        relplotdata=self.forscatterplot.reset_index()
        relplotdata['Timepoint']='All'

        relplotdata=pd.concat([self.forscatterplot.reset_index(),relplotdata])


        sns.relplot(data=relplotdata, x="x", y="y",col="Timepoint", hue="Cluster",style='Sample',s=100,kind="scatter",palette="bright")
        plt.savefig(f'{self.outdir}/Timepoint_analysis_clusters.png')
        plt.show()
        plt.close()


Xronos2=Clustering(args) 
Xronos2.plotting_clusters()

print (Xronos2.forscatterplot)
