import pandas as pd
import warnings
from sklearn import metrics
warnings.filterwarnings("ignore")


result = pd.DataFrame(index=["split1","split2","split3"], columns=["model"], dtype='float64')

for i in range(1,4):
    first_submission='Eval/mysubmission_'+str(i)+'.txt'
    all=pd.read_csv('data.tsv', sep=' ', quotechar='"', escapechar='\\',index_col=["new_id"]) # read in the entire data
    splits = pd.read_csv("splits.csv",low_memory=True, sep='\t') # read the split data
    review_test= all.loc[splits.iloc[:,i-1], :] # filter test id
    
    first = pd.read_csv(first_submission, header=0,index_col=["new_id"])
    inner1 = pd.merge(review_test, first , how='left', on='new_id')

    fpr, tpr, thresholds = metrics.roc_curve(inner1.sentiment, inner1.prob)

    index="split"+str(i)
    result.loc[index,"model"]= metrics.auc(fpr, tpr)

print(result)