import shap_
import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

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

# 1.291
# 1.3405128

# Model 1: Trained model
print("\n\n=== === === ===\nInitializing the explainer")
explainer = shap_.Explainer(
    rgrModel.model, masker=dataset.values, algorithm="permutation",
    max_evals=1750, feature_names=list(repGCDataset.columns), seed=0
)

print("\n\n=== === === ===\nCalling the explainer")
testExplainer = explainer(
    repGC_X[3,:].reshape(1, repGC_X.shape[1]), max_evals=1750
)
# print(testExplainer)

# plt.figure()
# plotTest = shap_.plots.waterfall(testExplainer[0], show=False)
# plotTest.set_figheight(7.); plotTest.set_figwidth(12.)
# plotTest.tight_layout()

# plt.show()

sys.exit()

