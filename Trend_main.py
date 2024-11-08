from Trend_classes import *
parser = argparse.ArgumentParser()

study='0368'

#study='5207'

parser.add_argument("-d","--dir", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default=f'/home/arislit/Time-series_data/Time_verse_results/Trends/MGYS0000{study}')##home/arislit/Time-series_data/Datasets/MGYS00003476/
parser.add_argument("-t","--table", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default=f'/home/arislit/Time-series_data/Datasets/MGYS0000{study}/MGYS0000{study}.tsv') ## /home/arislit/Time-series_data/Datasets/MGYS00003476/MGYS00003476_reindexed.tsv
parser.add_argument("-m","--mapping", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default=f'/home/arislit/Time-series_data/Datasets/MGYS0000{study}/MGYS0000{study}_mapping.tsv')## /home/arislit/Time-series_data/Datasets/MGYS00003476/MGYS00003476_mapping.tsv
parser.add_argument("-tp","--Timepoint", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default='0')

parser.add_argument("-s","--Timepoint_series", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default='')# S00,S02,S04,S06,S08') # ''


parser.add_argument("-e","--num_training_loops", help="Do you have a As to initialise??", nargs='?', type=int, default=100)# S00,S02,S04,S06,S08') # ''


parser.add_argument("--first_timepoint_only", help="Should I run only the first timepoint as training??",action='store_true', default=False)# S00,S02,S04,S06,S08') # ''


parser.add_argument("--local", help="Do you have a As to initialise??",action='store_true', default=True)# S00,S02,S04,S06,S08') # ''

args=parser.parse_args()



Xronos=Time_mapper(args)

training_samples=random.choices([i for i in Xronos.paired_samples.keys() if i not in Xronos.later_points],k=int(len(Xronos.directed_otus)*0.8)) if args.first_timepoint_only else random.choices(list(Xronos.paired_samples.keys()),k=int(len(Xronos.paired_samples)*0.8))
training_set={i:Xronos.paired_samples[i] for i in training_samples}
validation_samples=[j for j in Xronos.paired_samples.keys() if j not in training_samples]
validation_set={j:Xronos.paired_samples[j] for j in validation_samples}

if __name__=='__main__':


    Trends={}
    Skipped=[]
    Microbe_accuracy=pd.DataFrame([])
    with alive_bar(len(Xronos.taxalist),title='Microbe NN') as bar:
        for microbe in Xronos.taxalist:
            
            Delta_set=Delta_Trend_Dataset(directed=training_set,microbe=microbe)
            Delta_loader=DataLoader(Delta_set,batch_size=10)
            
            Validation_Delta_set=Delta_Trend_Dataset(directed=validation_set,microbe=microbe)
            Validation_loader=DataLoader(Validation_Delta_set,batch_size=1)


            whole_delta_set=Delta_Trend_Dataset(directed=Xronos.paired_samples,microbe=microbe)
            whose_set_loader=DataLoader(whole_delta_set,batch_size=1)

            model= Microbe_trend_predictor(input_dim=Xronos.input_dimensions,num_layers=3,activation='Sigmoid')
            optimizer=torch.optim.Adam(model.parameters(),lr=.1,weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer,min_lr=0.01,patience=4,cooldown=5, mode='min')

            criterion=nn.CrossEntropyLoss()
            
            if not Delta_set.consider_it:
                Skipped.append(microbe)
                continue
            Microbe_trends,model=Training_models(Delta_loader,optimizer,model,scheduler,criterion,args.num_training_loops)

            Microbe_trends=Validation(model,Microbe_trends,Validation_loader,Validation_Delta_set,'Validation')

            Microbe_accuracy.loc[microbe,'Validation_Mean_Accuracy']=Microbe_trends['Validation_Accuracy']
            Microbe_accuracy.loc[microbe,'Validation_Fold_Change']=np.mean([i for i in Validation_Delta_set.fold_change.values()]).item()
            Microbe_accuracy.loc[microbe,'Mean_Final_value']=np.mean([i['Target'] for i in Microbe_trends['Validation_Actual_Target_Values'].values()]).item()
            
            Microbe_trends=Validation(model,Microbe_trends,whose_set_loader,whole_delta_set,'Whole_Dataset')    

            Trends[microbe]=Microbe_trends
            bar()

    if args.first_timepoint_only:
        Later_tps={i:[] for i in Xronos.later_points}
        for microbe,key in Trends.items():
            for i in Xronos.later_points:
                Later_tps[i].append(key['Accuracy'][i])

        Later_tps={i:np.mean(Later_tps[i]).item() for i in Later_tps.keys()}

        with open(f'{Xronos.outdir}/Later_Timepoints.json','w') as g:
            json.dump(Later_tps,g,indent=4)



    sns.scatterplot(data=Microbe_accuracy,x='Validation_Fold_Change',y='Validation_Mean_Accuracy')
    plt.savefig(f'{Xronos.outdir}/Mean_Acc_FoldChange.png')
    plt.close()
    sns.scatterplot(data=Microbe_accuracy,x='Mean_Final_value',y='Validation_Mean_Accuracy')
    plt.savefig(f'{Xronos.outdir}/Mean_Acc_Mean_TargetValue.png')
    plt.close()

    Microbe_accuracy.to_csv(f'{Xronos.outdir}/Microbe_Accuracy.tsv',sep='\t',header=True,index=True)
    with open(f'{Xronos.outdir}/Trends.json','w') as f:
        json.dump(Trends,f,indent=4)


    print (f'Your files are saved in {Xronos.outdir} \nThank you for choosing us today!')