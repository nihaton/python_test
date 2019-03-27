from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate


def main():
    # 乳がんデータセットを読み込む
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    # ロジスティック回帰
    clf = LogisticRegression()
    # Stratified K-Fold CV で性能を評価する
    skf = StratifiedKFold(shuffle=True)
    scoring = {
        'acc': 'accuracy',
        'auc': 'roc_auc',
    }
    scores = cross_validate(clf, X, y, cv=skf, scoring=scoring)

    print('Accuracy (mean):', scores['test_acc'].mean())
    print('AUC (mean):', scores['test_auc'].mean())


if __name__ == '__main__':
    main()
