import sys, glob
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# usage: python ensemble.py "./contrastive_run_v2/probs_V2_s*.npy" ./contrastive_run_v2/preds_V2_s2007.npz
pattern, ref = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(pattern))
print('Ensembling', len(files), 'runs:', files)

labels = np.load(ref)['labels']
probs = np.mean([np.load(f) for f in files], axis=0)
preds = probs.argmax(1)

names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
print('Acc: {:.2f}  weighted F1: {:.2f}  macro F1: {:.2f}'.format(
    accuracy_score(labels, preds) * 100,
    f1_score(labels, preds, average='weighted') * 100,
    f1_score(labels, preds, average='macro') * 100))
print(classification_report(labels, preds, target_names=names, digits=4))
cm = confusion_matrix(labels, preds)
print(cm)
print('hap<->exc errors:', cm[0, 4] + cm[4, 0], ' ang<->fru errors:', cm[3, 5] + cm[5, 3])