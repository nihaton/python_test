from sklearn import datasets
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
 
def multi_regression(*feature_index):
    boston = datasets.load_boston()
    df = pd.DataFrame(boston.data)
    df.columns = boston.feature_names
    df['PRICE'] = pd.DataFrame(boston.target)
 
    x = df.loc[:, ['{0}'.format(x) for x in feature_index]]
    y = df['PRICE']
 
    model = smf.OLS(y, x)
    result = model.fit()
    print(result.summary())
 
    # numpy sort
    # https://qiita.com/supersaiakujin/items/c580f2aae90818150b35
    desc_idx = np.argsort(result.tvalues.values)[::-1]
    labels = []
    values = []
    for idx in desc_idx:
        labels.append(feature_index[idx])
        values.append(result.tvalues.values[idx])
 
    plt.bar(labels, values)
    plt.legend(title="r^2(adj) = {0:.6}".format(result.rsquared_adj))
 
    plt.show()
 
 
if __name__ == "__main__":
    multi_regression('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')
    # multi_regression('RM', 'AGE', 'TAX', 'B')