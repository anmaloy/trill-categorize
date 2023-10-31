import sys
sys.path.append('..')
from src import readSGLX
from src.io import get_ni_analog

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import pandas as pd

from pathlib import Path
from scipy import signal
from sklearn import preprocessing

from matplotlib.colors import ListedColormap
from scipy.signal import find_peaks, peak_prominences, peak_widths
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def prepare_data(data, start=0, stop=0, drop=True):
    label_encoder = preprocessing.LabelEncoder()
    data = data.replace('nan', pd.NA).dropna(axis=0).reset_index(drop=drop)
    try:
        data = data.drop('time(s)', axis=1)
    except KeyError:
        print('KeyError: [time(s)] not found in axis. Already dropped.')
    data['type'] = label_encoder.fit_transform(data['type'])
    X = data.iloc[:, start:stop]
    y = data.iloc[:, stop]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test, label_encoder


def select_features(data):
    X_train, X_test, y_train, y_test, le = prepare_data(data, 0, 8)
    fs = SelectKBest(k='all')
    fs.fit(X_train, y_train)
    for i in range(len(fs.scores_)):
        print(f'Feature {fs.feature_names_in_[i]}: {fs.scores_[i]}')
    plt.bar([fs.feature_names_in_[i] for i in range(len(fs.feature_names_in_))], fs.scores_)
    plt.show()


def nu_net(data):
    featTarg = ['amplitude(v)', 'hwidth(len)', 'hwidth(amp)', 'prominence', 'thresholds', 'type']
    data = data[featTarg]
    X_train, X_test, y_train, y_test, le = prepare_data(data, 0, len(featTarg)-1)
    pred_df = pd.DataFrame(X_test, columns=featTarg)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    pca = PCA(n_components='mle', whiten=True)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    classifier = MLPClassifier(alpha=1, max_iter=1000, random_state=0)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    pred_df['type'] = le.inverse_transform(predictions)

    cm = confusion_matrix(y_test, predictions)
    cor = np.diagonal(cm).sum()
    totalCount = len(X_test)
    pcnt = round((cor / totalCount) * 100, 2)

    print("Type Prediction: ")
    print("  {} / {} Correct:   {}%".format(cor, totalCount, pcnt))
    print("  Confusion Matrix: \n", cm)

    print(classification_report(y_test, predictions, zero_division=0))

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()
    return pred_df


def pca_comparison(data):
    X_train, X_test, y_train, y_test, le = prepare_data(data, 0, 8)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    pca = PCA(n_components=2)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=0),
        SVC(gamma=2, C=1, random_state=0),
        DecisionTreeClassifier(max_depth=5, random_state=0),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=0),
        MLPClassifier(alpha=1, max_iter=1000, random_state=0),
        AdaBoostClassifier(random_state=0),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    i = 1
    for name, clf in zip(names, classifiers):
        print(f'Calculating {name}')
        ax = plt.subplot(1, len(classifiers), i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        y_pred = clf.predict(X_test)

        X_set, y_set = X_test, y_test

        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                       stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1,
                                       stop=X_set[:, 1].max() + 1, step=0.01))

        ax.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
                    cmap=ListedColormap(('yellow', 'salmon', 'aquamarine')))

        ax.set_xlim(X1.min(), X1.max())
        ax.set_ylim(X2.min(), X2.max())

        for a, b in enumerate(np.unique(y_set)):
            ax.scatter(X_set[y_set == b, 0], X_set[y_set == b, 1],
                       c=ListedColormap(('red', 'green', 'blue', 'yellow', 'grey'))(a), label=b)

        # title for scatter plot
        ax.set_xlabel('PC1')  # for Xlabel
        ax.set_ylabel('PC2')  # for Ylabel
        ax.text(1.5, -2, ("%.2f" % score).lstrip("0"), size=15, horizontalalignment="right")
        ax.legend()

        ax.set_title(name)
        i += 1

    # show scatter plot
    plt.show()


def spikes_chart(data, targets, nidaq_time, nidaq_data, cutoff):
    data = data.replace('nan', pd.NA).dropna(axis=0)
    plt.plot(nidaq_time, nidaq_data, linewidth=0.1)
    for index, row in data.iterrows():
        if row['type'] == 'S':
            plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='tomato', alpha=0.75)
        elif row['type'] == 'F':
            plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='forestgreen', alpha=0.75)
        elif row['type'] == 'I':
            plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='gold', alpha=0.75)
        elif row['type'] == 'VC':
            plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='slateblue', alpha=0.75)
        elif row['type'] == 'B':
            plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='silver', alpha=0.75)
        else:
            plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='black', alpha=0.75)
        plt.hlines(data['hwidth(amp)'][index], data['hwidth(start)'][index], data['hwidth(finish)'][index], color="C2")
    plt.ylim(-0.4, 0.9)
    plt.axhline(y=cutoff, color='red', linewidth=0.3)
    legend = [mpatches.Patch(color='yellow', label='Intro Trill', alpha=0.5),
              mpatches.Patch(color='green', label='Fast Trill', alpha=0.5),
              mpatches.Patch(color='red', label='Slow Trill', alpha=0.5),
              mpatches.Patch(color='blue', label='Vocal Comp AP', alpha=0.5),
              mpatches.Patch(color='grey', label='Breathing', alpha=0.5)]
    plt.legend(handles=legend, loc='lower right')
    plt.xlabel('Times (s)')
    plt.ylabel('Nerve (v)')
    plt.title('NIDAQ Nerve Spikes')
    for index, row in targets.iterrows():
        if row[3] == 'S':
            plt.axvspan(row[1], row[2], color='red', alpha=0.25)
        if row[3] == 'F':
            plt.axvspan(row[1], row[2], color='green', alpha=0.25)
        if row[3] == 'I':
            plt.axvspan(row[1], row[2], color='yellow', alpha=0.25)
        if row[3] == 'VC':
            plt.axvspan(row[1], row[2], color='blue', alpha=0.25)
        if row[3] == 'B':
            plt.axvspan(row[1], row[2], color='grey', alpha=0.25)
    plt.show()

    # data.to_csv('data\\data.csv', index=False)
    data.to_csv(f'data\\{fileName}_g{gate}-data.csv', index=False)


def scatter_matrix(data):
    data = data.replace('nan', pd.NA).dropna(axis=0)
    data = data.drop('time(s)', axis=1)
    sns.set_theme(style="ticks")
    sns.pairplot(data, hue="type")
    plt.show()


def get_peaks(nidaq_time, nidaq_data, cutoff=0.04, distance=10, delay=0, other=False):
    peaks, properties = find_peaks(nidaq_data, height=cutoff, threshold=0, distance=distance*sRate/1000)
    width_half = peak_widths(nidaq_data, peaks, rel_height=0.5)
    lst = list(width_half)
    lst[0] = lst[0] / sRate
    lst[2:] = (x / sRate for x in lst[2:])
    width_half = tuple(lst)
    lst[0] = lst[0] / sRate
    lst[2:] = (x / sRate for x in lst[2:])
    peaks_time = nidaq_time[peaks]
    prominence = peak_prominences(nidaq_data, peaks)[0]
    data, targets = populate_data(peaks_time, properties, width_half, prominence, delay)
    if other is True:
        data['type'] = data['type'].replace('nan', 'O')
    return data, targets, cutoff


def populate_data(peaks_time, properties, width_half, prominence, delay):
    targets = pd.read_csv('data\\targets.csv')
    targets.start = (targets.start + delay) / 1000
    targets.end = (targets.end + delay) / 1000
    data = pd.DataFrame()

    data['time(s)'] = peaks_time
    data['amplitude(v)'] = properties['peak_heights']
    data['delay(s)'] = data['time(s)'].diff()
    data['hwidth(len)'] = width_half[0]
    data['hwidth(amp)'] = width_half[1]
    data['hwidth(start)'] = width_half[2]
    data['hwidth(finish)'] = width_half[3]
    data['prominence'] = prominence
    data['thresholds'] = properties['left_thresholds'] + properties['right_thresholds']
    lst = targets.apply(lambda row: (row['start'], row['end']), axis=1)
    conditions = [(data['time(s)'] > x[0]) & (data['time(s)'] < x[1]) for x in lst]
    choices = targets.type.values
    data['type'] = np.select(conditions, choices, default=np.nan)
    return data, targets


channel = 0
fileName = 'NPX-S2-39'
gate = 3
# +~55ms (27_g0), +~120 (27_g7)
t_delay = 0
binPath = Path(f'data\\{fileName}\\catgt_{fileName}_g{gate}\\{fileName}_g{gate}_tcat.nidq.bin')
binMeta = readMeta(binPath)
sRate = SampRate(binMeta)
ni_time, ni_data = get_ni_analog(binPath, channel)

df, targetdf, p_cutoff = get_peaks(ni_time, ni_data, delay=t_delay, other=False)
spikes_chart(df, targetdf, ni_time, ni_data, p_cutoff)
# pdf = nu_net(df)
# plist = pdf.index.tolist()
#
# tdf = df.iloc[plist].copy()
# tdf['otype'] = df.iloc[plist]['type']
# df['type'] = np.nan
# tdf['type'] = pdf.type
# df.update(tdf)
# spikes_chart(df, targetdf, ni_time, ni_data, p_cutoff)

# todo
# pass filepath to get_peaks to dynamically name the targets csv it draws from, put them in individual data/targets
# folders
