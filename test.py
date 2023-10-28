import shap
import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

if len(sys.argv) < 6:
    print(
        "Possible usage: python3 getSHAP.py <savedModel> <dataset> <repGCPkl> " +
        "<subscore> <patLabels>"
    )
    sys.exit(1)
else:
    savedModel = Path(sys.argv[1])
    dataset = pd.read_table(Path(sys.argv[2]), sep=' ', index_col="StridePairID")
    repGCPkl = Path(sys.argv[3])
    subscore = sys.argv[4]
    patLabels = pd.read_excel(sys.argv[5], engine="openpyxl", index_col="ID")
with open(savedModel, "rb") as handle:
    rgrModel = pickle.load(handle)

# Extracting the representative samples
with open(repGCPkl, "rb") as handle:
    repGCDict = pickle.load(handle)

repGC_0 = repGCDict[0].iloc[0,:]
repGC_1 = repGCDict[1].iloc[0,:]
repGC_2 = repGCDict[2].iloc[0,:]
repGC_3 = repGCDict[3].iloc[0,:]

repGCDataset = pd.DataFrame(dataset.loc[repGC_0.name, :]).T
repGCDataset = pd.concat([repGCDataset, pd.DataFrame(dataset.loc[repGC_1.name, :]).T])
repGCDataset = pd.concat([repGCDataset, pd.DataFrame(dataset.loc[repGC_2.name, :]).T])
repGCDataset = pd.concat([repGCDataset, pd.DataFrame(dataset.loc[repGC_3.name, :]).T])
repGC_X = repGCDataset.values

nTotalSamples = 90
nSubscore0 = round((repGCDict[0].shape[0]/dataset.shape[0])*nTotalSamples)
nSubscore1 = round((repGCDict[1].shape[0]/dataset.shape[0])*nTotalSamples)
nSubscore2 = round((repGCDict[2].shape[0]/dataset.shape[0])*nTotalSamples)
nSubscore3 = round((repGCDict[3].shape[0]/dataset.shape[0])*nTotalSamples)
nSubscore = np.array((nSubscore0, nSubscore1, nSubscore2, nSubscore3))

if nSubscore.sum() < nTotalSamples:
    nSubscore[np.argmin(nSubscore)] = nSubscore[np.argmin(nSubscore)] + (nTotalSamples - nSubscore.sum())
elif nSubscore.sum() > nTotalSamples:
    nSubscore[np.argmax(nSubscore)] = nSubscore[np.argmax(nSubscore)] - (nSubscore.sum() - nTotalSamples)

# Sampling 90 samples as background dataset for masking (averaged out)
samples0 = repGCDict[0].iloc[0:nSubscore[0]]
samples1 = repGCDict[1].iloc[0:nSubscore[1]]
samples2 = repGCDict[2].iloc[0:nSubscore[2]]
samples3 = repGCDict[3].iloc[0:nSubscore[3]] 
samplesMasking = list(samples0.index) + list(samples1.index) + list(samples2.index) + list(samples3.index)
datasetMask = dataset.loc[samplesMasking,:]

# getPatient = lambda x: x[0:5] if x.startswith("ES") else x[0:8]
# yDatasetMask = pd.DataFrame(
#     data=np.zeros((datasetMask.shape[0], 1)), columns=[subscore], index=datasetMask.index
# )
# for idx in datasetMask.index:
#     yDatasetMask.at[idx, subscore] = patLabels.at[getPatient(idx), subscore]

# print(yDatasetMask.mean())

# 1.333333 (Mean from the masking samples)
# 1.344746 (Base values from SHAP)

# Model 1: Trained model
print("\n\n=== === === ===\nInitializing the explainer")
explainer = shap.Explainer(
    rgrModel.model, masker=datasetMask.values, algorithm="permutation",
    max_evals=1750, feature_names=list(repGCDataset.columns), seed=0
)

print("\n\n=== === === ===\nCalling the explainer")
testExplainer = explainer(
    repGC_X[3,:].reshape(1, repGC_X.shape[1]), max_evals=1750
)

# plt.figure()
# plotTest = shap.plots.waterfall(testExplainer[0], show=False)
# plotTest.set_figheight(7.); plotTest.set_figwidth(12.)
# plotTest.tight_layout()

# plt.show()

sys.exit()

