from Trend_main import *
from collections import Counter

with open(f'{Xronos.outdir}/Trends.json','r') as f:
    Trends=json.load(f)

trend={0:'lower',1:'equal',2:'higher'}
finder=np.array([0,1,2])

trajectory=pd.DataFrame([],columns=validation_samples)

metrics={}
for microbe in Trends.keys():
    metrics[microbe]={}
    metrics[microbe]['Trend']=Counter([trend[np.dot(i[0],finder).item()] for i in Trends[microbe]['Validation_Targets']]).most_common(1)[0][0]
    metrics[microbe]['Validation_Accuracy']=Trends[microbe]['Validation_Accuracy']
    metrics[microbe]['Whole_Dataset_Accuracy']=Trends[microbe]["Whole_Dataset_Accuracy"]


per_microbe=pd.DataFrame(metrics).T

pm = per_microbe.melt(id_vars='Trend', value_vars=['Validation_Accuracy', 'Whole_Dataset_Accuracy'], 
                    var_name='Dataset', value_name='Accuracy')
sns.violinplot(pm,x='Trend',y='Accuracy',hue='Dataset')
plt.show()
