import IPython
from IPython.core.display import HTML
import base64
import datetime as datetime
import time
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import cm 
import seaborn as sns
from colorama import Fore, Back, Style
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import time, scipy
from tqdm.notebook import tqdm
from tqdm import tqdm
tqdm.pandas()

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (train_test_split, KFold,
                                     StratifiedKFold, GridSearchCV)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import (LogisticRegression, SGDClassifier,
                                  Perceptron)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,  
                              GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier,
                              BaggingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (make_scorer, recall_score, classification_report,
                             accuracy_score,recall_score, f1_score)

# imblearn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek 
from imblearn.pipeline import Pipeline
import warnings 
warnings.filterwarnings("ignore")

# clustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedians import kmedians
from scipy.spatial.distance import euclidean
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# nlp libraries
import re, nltk, spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud, STOPWORDS

# explainability
import shap
import lime
from lime.lime_text import LimeTextExplainer


class Preprocess(object):
    """Class for word pre-processing"""
    def __init__(self, df):
        self.df = df.copy()
        self.df['Content'] = self.filter_words(self.df)
        self.df['Content'] = (self.df.Content.apply(' '.join))
        self.tfidf = self.vectorize()

    def filter_words(self, df):
        """Remove filler words and stop words"""
        df_hp = df.copy()
        df_hp['Content'] = (df_hp.Content.str.lower()
                       .replace(r'(ooh|yeah|got|know|huh|oh|ayy)', 
                                ' ', regex=True))

        df_hp['Content'] = (df_hp.Content.str.lower()
                           .replace(r'  ', ' ', regex=True))

        # tokenize
        tokenize = df_hp.Content.progress_apply(nltk.word_tokenize)

        # casefold
        lower_case = tokenize.progress_apply(lambda x:
                                             list(map(lambda y: 
                                                      y.casefold(), x)))


        # lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatize = lower_case.progress_apply(lambda x: list(
                                                map(lemmatizer.lemmatize,
                                                 x)))

        # remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_stopwords = lemmatize.progress_apply(lambda x:
                                                      list(filter(lambda y: y
                                                                  not in 
                                                                  stop_words,
                                                                  x)))

        # filter words with less than 3 character length
        filtered_words = filtered_stopwords.progress_apply(lambda x:
                                                           list(
                                                            filter(lambda y:
                                                                   len(y) > 3,
                                                                   x)))

        # filter common song words
        final_stopwords = (stopwords.words('english') 
                           + stopwords.words('french')
                           + stopwords.words('spanish')) 

        furn_words = ['like', 'share', 'subscribe', 'facebook', 'comment',
                     'manila', 'philippines', 'adobo', 'chronicles'] + final_stopwords
        filtered_words = filtered_words.progress_apply(lambda x:
                                                       list(filter(lambda y: y
                                                                   not in
                                                                   furn_words,
                                                                   x)))

        return filtered_words


    def vectorize(self,):
        """Vectorize using TF-IDF"""
        tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b[a-z]+\b', 
                                   ngram_range=(1, 2),
                                   max_df=0.8,
                                   min_df=0.02,
                                          max_features=1000)

        bow_ng = tfidf_vectorizer.fit_transform(self.df.Content)

        tfidf_hp = pd.DataFrame.sparse.from_spmatrix(
                            bow_ng, 
                            columns=tfidf_vectorizer.get_feature_names())
        
        return tfidf_hp
    
def pooled_within_ssd(X, y, centroids, dist):
    """Compute pooled within-cluster sum of squares around the cluster mean
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
        
    Returns
    -------
    float
        Pooled within-cluster sum of squares around the cluster mean
    """
    out = []
    for i in range(len(centroids)):
        out.append(sum(map(lambda x: 
                           dist(x, centroids[i])**2, X[y==i]))
                   /(2*len(X[y==i])))

    return sum(out)

def gap_statistic(X, y, centroids, dist, b, clusterer, random_state=None):
    """Compute the gap statistic
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
    b : int
        Number of realizations for the reference distribution
    clusterer : KMeans
        Clusterer object that will be used for clustering the reference 
        realizations
    random_state : int, default=None
        Determines random number generation for realizations
        
    Returns
    -------
    gs : float
        Gap statistic
    gs_std : float
        Standard deviation of gap statistic
    """
    rng = np.random.default_rng(1337)
    Wk = pooled_within_ssd(X, y, centroids, dist)
    out = []
    for i in range(b):
        sim_x = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)

        sim_y = clusterer.fit_predict(sim_x)
        sim_centroids = clusterer.cluster_centers_

        Wk_sim = pooled_within_ssd(sim_x, sim_y, sim_centroids, dist)
        out.append(np.log(Wk_sim) - np.log(Wk))

    return np.mean(out), np.std(out)


def cluster_range(X, clusterer, k_start, k_stop):
    """ Cluster the input X and compute for its 
        internal and external validation
    
    Parameters
    ----------
    X         :    np.ndarray
                   data containing features and samples
    clusterer :    sklearn.cluster._kmeans.KMeans
                   Kmeans object
    k_start   :    int
                   start number of clusters
    k_stop    :    int 
                   end number of clusters
    
    Returns
    -------
    cluster_range  : dict
                     contains details on clusters formed, and 
                     internal and external validation
    """
    ys = []
    centers = []
    inertias = []
    chs = []
    scs = []
    gss = []
    gssds = []
    ps = []
    amis = []
    ars = []
    for k in tqdm(range(k_start, k_stop+1)):
        clusterer_k = clone(clusterer)
        clusterer_k.set_params(n_clusters=k)
        y = clusterer_k.fit_predict(X)
        
        ys.append(y)
        centers.append(clusterer_k.cluster_centers_)
        inertias.append(clusterer_k.inertia_)
        
        chs.append(calinski_harabasz_score(X, y))
        
        scs.append(silhouette_score(X, y))
        
        gs = gap_statistic(X, y, clusterer_k.cluster_centers_, 
                                 euclidean, 5, 
                                 clone(clusterer).set_params(n_clusters=k), 
                                 random_state=1337)
        
        gss.append(gs[0])
        gssds.append(gs[1])
        
    res = {'ys' : ys, 
           'centers' : centers, 
           'inertias' : inertias, 
           'chs' : chs, 
           'scs' : scs, 
           'gss' : gss, 
           'gssds' : gssds}

    return res


def plot_clusters(X, ys, centers, transformer):
    """Plot clusters given the design matrix and cluster labels"""
    k_max = len(ys) + 1
    k_mid = k_max//2 + 2
    fig, ax = plt.subplots(k_max//2,2, sharex=True, sharey=True, 
                           figsize=(15, 50), 
                           )
    for k,y,cs in zip(range(2,k_max+1), ys, centers):
        centroids_new = transformer.transform(cs)
        if k < k_mid:
            ax[k%k_mid-2][0].scatter(*zip(*X), c=y, cmap=cm.cool,
                                     s=1, alpha=0.8)
            ax[k%k_mid-2][0].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y)) + 1),
                marker='s',
                ec='k',
                lw=1,
                cmap=cm.cool
            );
            ax[k%k_mid-2][0].set_title('$k=%d$'%k)
            ax[k%k_mid-2][0].spines['top'].set_visible(False)
            ax[k%k_mid-2][0].spines['right'].set_visible(False)
        else:
            ax[k%k_mid][1].scatter(*zip(*X), c=y, s=1, alpha=0.8, cmap=cm.cool)
            ax[k%k_mid][1].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y))+1),
                marker='s',
                ec='k',
                lw=1,
                cmap=cm.cool
            );
            ax[k%k_mid][1].set_title('$k=%d$'%k)
            ax[k%k_mid][1].spines['top'].set_visible(False)
            ax[k%k_mid][1].spines['right'].set_visible(False)
        plt.tight_layout()
    return ax

def plot_internal(n_clusters, inertias, chs, scs, gss, k):
    """Plot internal validation scores"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    axes = axes.flatten()
    axes[0].plot(range(2, n_clusters + 2), inertias, c='#E066FF', lw=4)
    axes[0].set_title('Sum of Squared Distance');
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_xlabel('k')
    axes[0].axvline(k, ls='--', color='blue')
        
    
    axes[1].plot(range(2, n_clusters + 2), chs, c='#E066FF', lw=4)
    axes[1].set_title('Calinski-Harabasz');
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_xlabel('k')
    axes[1].axvline(k, ls='--', color='blue')
    
    axes[2].plot(range(2, n_clusters + 2), scs, c='#E066FF', lw=4)
    axes[2].set_title('Silhouette Score');
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].set_xlabel('k')
    axes[2].axvline(k, ls='--', color='blue')
    
    axes[3].plot(range(2, n_clusters + 2), gss, c='#E066FF', lw=4)
    axes[3].set_title('Gap Statistic');
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    axes[3].set_xlabel('k')
    axes[3].axvline(k, ls='--', color='blue')
    
    plt.tight_layout()


def clusters(df,k):
    """Cluster `df` into `k` clusters using K-Means"""
    clusterer = KMeans(n_clusters=k, random_state=1337)
    y = clusterer.fit_predict(df.to_numpy())
    df['cluster'] = y
    return df,y    

def lsa(y,tfidf):
    """Perform LSA on input TF-IDF"""
    lsas = {}
    for i in tqdm(range(len(np.unique(y)))):  
        df_bow = tfidf[tfidf.cluster == i].iloc[:, :-2].copy()
        tsvd = TruncatedSVD(n_components=len(df_bow.columns)-1)
        tsvd.fit(df_bow.to_numpy())

        sv = tsvd.components_.T
        X_bow_new = tsvd.transform(df_bow.to_numpy())
        exp = tsvd.explained_variance_ratio_
        lsas[i] = {
            'sv' : sv, 
            'X_bow_new': X_bow_new, 
            'exp' : exp
        }
    return lsas

def lsa_word_cloud(lsa, df):
    """Generate LSA Word Cloud"""
    for key, val in lsa.items():
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        features = df.columns
        norm = ((val['sv'][:, 0] - val['sv'][:, 1])**2)
        ind = np.argsort(norm, axis=0)[-50:]
        for i in range(10):            
            weights = {i : j for i, j in zip(features[ind], norm[ind])}
            wc = WordCloud(contour_color='white',
                           background_color='white',
                           colormap=cm.Dark2,#cm.Spectral,
                           max_words=100)
            wc.generate_from_frequencies(weights)
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
        plt.suptitle(f'Cluster: {key}')
        plt.tight_layout()
        
## for modelling
def tokenize(corpus):
    """Tokenize the text written in English language."""
    pattern = r'[^\s\,\.\?\!\;\(\)\"\"\#\&\*\/\‘\“\”\:\’\—\-\â\0-9]+'
    tokenizer = RegexpTokenizer(pattern)
    tokenized = [" ".join(tokenizer.tokenize(corpus.values[i].lower()))
                for i in range(len(corpus))]
    return pd.Series(tokenized)

def remove_stopwords(corpus):
    """Remove stop words from the tokenized text."""
    stop_words = stopwords.words('english') + list(en_stop)
    stop_words += ['what\'s', 'there\'s', 'that\'s', 'i\'m', 'pm', 'am', 'nn',
                   'we\'re', 'i\'ve', 'i\'ll', 'i\'d', 'etc','like', 'share',
                   'subscribe', 'facebook', 'comment','manila', 'philippines',
                   'adobo', 'chronicles']
    stopped = [" ".join([text for text in
                        [word for word in corpus.values[i].split()]
                        if text not in stop_words])
               for i in range(len(corpus))]
    return pd.Series(stopped)

def lemmatize(corpus):
    """Lemmatize the texts without stop words."""
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    stemmed = [" ".join([token.lemma_ for token in
                         nlp(corpus.values[i])])
                         for i in range(len(corpus))]
    return pd.Series(stemmed)

def print_shap_contrib(review, shpvs, vocab):

    feature_importance = pd.DataFrame(list(zip(shpvs, vocab)),
                                      columns=['shap_vals', 'word'])
    feature_importance.sort_values(by=['shap_vals'], ascending=False, inplace=True)
    feature_importance = feature_importance.head(30).append(feature_importance.tail(30))
    feature_importance = feature_importance.set_index('word')
    text = ''
    for i in review.split():
        if i in feature_importance.index:
            if feature_importance.loc[i]['shap_vals'] > 0:
                text += (Fore.RED + Style.BRIGHT + i)
            else:
                text += (Fore.BLACK + Style.RESET_ALL + i)
        else:
            text += (Fore.BLACK + Style.RESET_ALL + i)
        text += ' '
    print(text)

