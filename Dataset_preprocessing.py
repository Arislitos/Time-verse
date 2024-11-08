from Trend_classes import *

parser = argparse.ArgumentParser()

parser.add_argument("-d","--dir", help="Do you have a As to initialise??", nargs='?', type=str, const=1, default=f'/home/arislit/Time-series_data/Datasets/')##home/arislit/Time-series_data/Datasets/MGYS00003476/

args=parser.parse_args()
outdir=args.dir if args.dir[-1]!='/' else args.dir[:-1]

envs={}
study_files={}
for study in os.listdir(outdir):
    if study[-3:]=='.sh':
        continue
    
    mapping_file=pd.read_csv(f'{outdir}/{study}/{study}_mapping.tsv',sep='\t',header=0,index_col=0)
    study_files[study]=mapping_file
    env_column=[col for col in mapping_file.columns if mapping_file[col].astype(str).str.contains('root:', case=True, na=False).any()][0]
    env=mapping_file[env_column].unique()[0]
    if env in envs.keys():
        envs[env].append(study)
    else:
        envs[env]=[study]

for i in envs:
    if len(envs[i])>3:
        for study in envs[i]:
            print (study_files[study]['Sample'].nunique(),study_files[study]['Timepoint'].unique(),study,i)
        