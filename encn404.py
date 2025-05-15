#encn404.py
from ipywidgets import*
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, clear_output, Math

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
try:
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
except AttributeError: # np.VisibleDeprecationWarning is not defined in numpy < 1.20
    pass

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
import itertools, os

def _rolling_window(Tm, Tsd, Ti, ts):
    # read in some data
    #ts=pd.read_csv('traffic_data.csv',parse_dates=[0]).set_index('time')['density']

    # plot the raw data
    f,(ax1,ax2,ax3)=plt.subplots(1,3, figsize=(16,4))
    for ax in [ax1,ax2,ax3]:
        ts.plot(style='k-', lw=0.5, ax=ax, label='raw data')

    # Calculate a rolling mean
    #Tm=150
    ts.rolling(window=Tm).mean().plot(style='b',ax=ax1)

    # Calculate a rolling standard deviation
    #Tsd=30
    ts.rolling(window=Tsd).std().plot(style='b', ax=ax2.twinx())

    # Calculate a rolling X-day harmonic 
    def rolling_fft(x, ti):
        fft = np.fft.fft(x)/len(x)
        psd = np.abs(fft)**2/2
        period_of_interest = ti
        ts=1./(np.fft.fftfreq(len(x)))
        i=np.argmin(abs(ts-ti))
        return psd[i]

    #Ti=30   # harmonic (days)
    ts.rolling(window=240).apply(rolling_fft, args=(Ti,)).plot(style='b', ax=ax3.twinx())

    for ax in [ax1,ax2,ax3]:
        ax.set_ylabel('traffic density')
    ax1.set_title(f'feature 1: {Tm:d}-day average')
    ax2.set_title(f'feature 2: {Tsd:d}-day std. dev.')
    ax3.set_title(f'feature 3: {Ti:d}-day harmonic')
def rolling_window():    
    Tm=IntSlider(value=150, min=20, max=240, step=10, description='$T_m$', continuous_update=False)
    Tsd=IntSlider(value=30, min=20, max=120, step=10, description='$T_{sd}$', continuous_update=False)
    Ti=IntSlider(value=30, min=10, max=50, step=5, description='$T_i$', continuous_update=False)
    ts=pd.read_csv('traffic_data.csv',parse_dates=[0]).set_index('time')['density']
    io=interactive_output(_rolling_window, {'Tm':Tm,'Tsd':Tsd,'Ti':Ti,'ts':fixed(ts)})
    return VBox([HBox([Tm, Tsd, Ti]),io])

def _clustering(step):
    # Data points (3 2D coordinates)
    data_points = np.array([[1., 6.], [3., 4.], [4., 10.], [3., 10.], [2., 8.]])
    num_clusters = 2

    # Initialize centroids (randomly)
    centroids = np.array([[-1., 7.], [6., 7.]])

    # Lists to store cluster assignments and centroids at each iteration
    cluster_assignments_history = []
    centroids_history = [centroids.copy()]

    # Perform K-means for 3 iterations
    for _ in range(4):
        # Step 1: Assign data points to clusters
        distances = np.linalg.norm(data_points[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        cluster_assignments_history.append(cluster_assignments.copy())

        # Step 2: Update centroids
        for i in range(num_clusters):
            centroids[i] = np.mean(data_points[cluster_assignments == i], axis=0)
            distances = np.linalg.norm(data_points[cluster_assignments == i, np.newaxis] - centroids, axis=2)**2
        centroids_history.append(centroids.copy())
    centroids_history.append(centroids.copy())

    # Plot the clusters and membership at each iteration
    f,axs=plt.subplots(1, 4, figsize=(12, 2.5))
    f.set_facecolor((1,1,1,0))
    for ax in axs:
        ax.set_xlim([-2, 7])
        ax.set_ylim([2, 11])
        ax.set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('initialise')
    [ax.set_title(f'iteration {i+1}') for i,ax in enumerate(axs[1:])]
    axs[0].plot(data_points[:, 0], data_points[:, 1], 'bs', ms=10, label='data')
    axs[0].plot(centroids_history[0][:, 0], centroids_history[0][:, 1], 'bo', mfc='w', mew=1.5, ms=10, label='centroid')
    axs[0].legend()
    if step==0: 
        return

    cs=['r','g']
    centroids_history=np.array(centroids_history)
    th=np.linspace(0,2*np.pi,101)
    for i,ax in enumerate(axs[1:]):
        for j,c in enumerate(cs):
            inds=np.where(cluster_assignments_history[i]==j)
            xs,ys=data_points[inds,:].T
            ax.plot(xs, ys,  c+'s', ms=10)
            cx=centroids_history[i,j,0]
            cy=centroids_history[i,j,1]
            ax.plot(cx,cy, c+'o', mfc='w', mew=1.5, ms=10)
            rm=np.max(np.sqrt((xs-cx)**2+(ys-cy)**2))
            # ax.plot(rm*np.sin(th)+cx, rm*np.cos(th)+cy, c+'--', lw=0.5)
            # print(centroids_history[i+1,j,:])
            # print(np.sum((xs-cx)**2+(ys-cy)**2))  
            
        if i*2+1 == step:
            return
        
        for j,c in enumerate(cs):
            ax.plot(centroids_history[i+1,j,0], centroids_history[i+1,j,1], c+'o', ms=10, mfc='w', mew=1.5, zorder=2)
            x0,x1=centroids_history[i:i+2,j,0]
            y0,y1=centroids_history[i:i+2,j,1]
            ax.arrow(x0,y0,(x1-x0)*0.85,(y1-y0)*0.85, color=c, length_includes_head=True, head_width=0.3, head_length=0.3, zorder=5)
        if (i+1)*2 == step:          
            return
def clustering():
    step=IntSlider(value=0, min=0, max=5, step=1, description='step', continuous_update=False)
    io=interactive_output(_clustering, {'step':step})
    return VBox([step,io])

def _kmeans(step):
    # Generate three overlapping clusters of normally distributed data
    n_samples = 300
    n_features = 2
    n_clusters = 3
    random_state = 42

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

    # Create a scatter plot of the generated data
    f,(ax1,ax2)=plt.subplots(1,2,figsize=(12, 5))
    ax2.set_title("Silhouette Score vs. Number of Clusters")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_xlim([1.9, 5.1])
    ax2.set_ylim([0.47, 0.87])
    ax2.set_xticks([2,3,4,5])
    ax2.grid()
    
    ax1.set_title(f"KMeans Clustering (raw data)")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    plt.tight_layout()

    if step == 0:
        ax1.scatter(X[:, 0], X[:, 1], c='gray', marker='o', edgecolor='k', s=50, label='Data points')
        ax1.legend()
        return
    
    # Run KMeans clustering for different number of clusters
    cluster_range = []
    silhouette_scores = []

    for n in [2, 3, 4, 5]:
        kmns = KMeans(n_clusters=n, random_state=random_state)
        kmns.fit(X)
        y_pred = kmns.predict(X)
        cluster_range.append(n)
        
        # Calculate silhouette score
        silhouette_scores.append(silhouette_score(X, y_pred))

        if n-1 == step:
            break

    # Create a scatter plot for each iteration
    ax1.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k', s=50)
    ax1.scatter(kmns.cluster_centers_[:, 0], kmns.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Cluster centers')
    ax1.set_title(f"KMeans Clustering (k={n})")
    ax1.legend()

    # Plot silhouette scores
    ax2.plot(cluster_range, silhouette_scores, marker='o')

def kmeans():
    step=IntSlider(value=0, min=0, max=4, step=1, description='step', continuous_update=False)
    io=interactive_output(_kmeans, {'step':step})
    return VBox([step,io])

from collections import Counter

def assign_running_count(input_sequence):
    counts = Counter(input_sequence)
    unique_values_with_counts = list(counts.items())

    output_sequence = []
    for value in input_sequence:
        count = counts[value]
        output_sequence.append(count)
        counts[value] -= 1

    return output_sequence

def _split_feature(fd1,fd2,fd3,check,sl1,sl2,sl3,df):
    f,(ax,ax1)=plt.subplots(1,2,figsize=(12, 4))
    ax_=ax.twinx()
    df1=df.loc[df[fd1]>sl1,:]
    df0=df.loc[df[fd1]<=sl1,:]
    ms=30
    ax.plot(df0.loc[df0["safe"],fd1], df0.loc[df0["safe"],fd2], 'go', ms=ms)
    ax.plot(df0.loc[~df0["safe"],fd1], df0.loc[~df0["safe"],fd2], 'ro', ms=ms)
    ax_.plot(df1.loc[df1["safe"],fd1], df1.loc[df1["safe"],fd3], 'go', ms=ms)
    ax_.plot(df1.loc[~df1["safe"],fd1], df1.loc[~df1["safe"],fd3], 'ro', ms=ms)
    ax.plot([],[], 'go', ms=10, label='safe')
    ax.plot([],[], 'ro', ms=10, label='unsafe')
    ax.axvline(x=sl1, color="gray", linestyle="--")#, label=f"Split at {fd1}={sl1:.1f}")
    ax.set_xlabel(fd1)
    # ax.set_yticks([])
    ax.set_ylabel(fd2)
    ax_.set_ylabel(fd3)
    if check:
        xlim=ax.get_xlim()
        ax.set_xlim(xlim)
        if fd2=="material_type":
            sl2=0.5
        if fd3=="material_type":
            sl3=0.5
        ax.plot([xlim[0], sl1], [sl2, sl2], '--', color='gray')
        ax_.plot([sl1, xlim[1]], [sl3, sl3], '--', color='gray')
    else:
        ylim0=ax.get_ylim()
        ylim1=ax_.get_ylim()
        ax.set_ylim([np.min([ylim0[0], ylim1[0]]),np.max([ylim0[-1], ylim1[-1]])])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1))

    ax1.set_xlim([-0.3, 2.5])
    ax1.set_ylim([-3.5, 3.5])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    bb=dict(facecolor='white', edgecolor='black')
    ax1.plot([0,1],[0,2], 'k-')
    ax1.plot([0,1],[0,-2], 'k-')
    if check:
        ax1.plot([1,2],[2,3], 'k-')
        ax1.plot([1,2],[2,1], 'k-')
        ax1.plot([1,2],[-2,-3], 'k-')
        ax1.plot([1,2],[-2,-1], 'k-')

    if not check:
        c2=[0.5,0.5,0.5]
        c1='k'
    else:
        c1=[0.5,0.5,0.5]
        c2='k'
    ax1.text(0, 0, f'{fd1}\n>{sl1:.1f}', ha='center', bbox=bb, color=c1)
    if check:
        if fd2 == 'material_type':
            txt=f'{fd2}\n is steel'
        else:
            txt=f'{fd2}\n>{sl2:.1f}'
        ax1.text(1, -2, txt, ha='center', bbox=bb, color=c2)
        if fd3 == 'material_type':
            txt=f'{fd3}\n is steel'
        else:
            txt=f'{fd3}\n>{sl3:.1f}'
        ax1.text(1, 2, txt, ha='center', bbox=bb, color=c2)

    if fd2 == 'material_type':
        df00=df0.loc[df0[fd2]=='Steel',:]
        df01=df0.loc[df0[fd2]=='Concrete',:]
    else:
        df00=df0.loc[df0[fd2]>sl2,:]
        df01=df0.loc[df0[fd2]<=sl2,:]
    if fd3=='material_type':
        df10=df1.loc[df1[fd3]=='Steel',:]
        df11=df1.loc[df1[fd3]!='Steel',:]
    else:
        df10=df1.loc[df1[fd3]>sl3,:]
        df11=df1.loc[df1[fd3]<=sl3,:]

    if check:
        ys=[-1,-3,3,1]
        dfs=[df00,df01,df10,df11]
    else:
        ys=[-2,2]
        dfs=[df0,df1]
    for y,dfi in zip(ys, dfs):
        if dfi.shape[0]==0:
            s=0;us=0
        else:
            s=dfi['safe'].sum()
            us=dfi.shape[0]-s
        if check:
            ax1.text(2, y, f'{s:d} Safe\n{us:d} Unsafe', bbox=bb, ha='center', color=c2)
        else:
            ax1.text(1, y, f'{s:d} Safe\n{us:d} Unsafe', bbox=bb, ha='center', color=c2)
        
    bb=dict(facecolor='white', edgecolor='blue', boxstyle='round')
    ax1.text(0.5, 1, 'True', ha='center', bbox=bb, style='italic', color='b')
    ax1.text(0.5, -1, 'False', ha='center', bbox=bb, style='italic', color='b')
    if check:
        ax1.text(1.5, 2.5, 'True', ha='center', bbox=bb, style='italic', color='b')
        ax1.text(1.5, 1.5, 'False', ha='center', bbox=bb, style='italic', color='b')
        ax1.text(1.5, -1.5, 'True', ha='center', bbox=bb, style='italic', color='b')
        ax1.text(1.5, -2.5, 'False', ha='center', bbox=bb, style='italic', color='b')

    if fd2 == fd3:
        y0=np.min([axi.get_ylim()[0] for axi in [ax,ax_]])
        y1=np.max([axi.get_ylim()[1] for axi in [ax,ax_]])
        [axi.set_ylim([y0,y1]) for axi in [ax,ax_]]

    plt.show()

def decision_tree():
    # Create the dataframe
    data = [
        {"load_capacity": 50, "material_type": "Concrete", "age": 10, "safe": False},
        {"load_capacity": 30, "material_type": "Concrete", "age": 5, "safe": True},
        {"load_capacity": 70, "material_type": "Concrete", "age": 25, "safe": False},
        {"load_capacity": 70, "material_type": "Steel", "age": 35, "safe": False},
        {"load_capacity": 60, "material_type": "Steel", "age": 15, "safe": True},
        {"load_capacity": 50, "material_type": "Steel", "age": 8, "safe": True},
        {"load_capacity": 35, "material_type": "Steel", "age": 3, "safe": True}
    ]
    df = pd.DataFrame(data)

    
    # Create the interactive widgets
    fd1 = widgets.Dropdown(value='load_capacity', options=["load_capacity", "age"], description="Feature:")
    fd2 = widgets.Dropdown(value='age', options=["load_capacity", "age", "material_type"], description="Feature:")
    fd3 = widgets.Dropdown(value='age', options=["load_capacity", "age", "material_type"], description="Feature:")
    check = Checkbox(value=False, description="lock root node")

    sl1 = widgets.FloatSlider(
        value=50,
        min=min(df["load_capacity"].min(), df["age"].min())-1,
        max=max(df["load_capacity"].max(), df["age"].max())+1,
        step=1,
        description="Split value:", continuous_update=False)
    sl2 = widgets.FloatSlider(
        value=10,
        min=min(df["load_capacity"].min(), df["age"].min())-1,
        max=max(df["load_capacity"].max(), df["age"].max())+1,
        step=1,
        description="Split value:", continuous_update=False)
    sl3 = widgets.FloatSlider(
        value=10,
        min=min(df["load_capacity"].min(), df["age"].min())-1,
        max=max(df["load_capacity"].max(), df["age"].max())+1,
        step=1,
        description="Split value:", continuous_update=False)
    # sl1.value=53
    # fd2.value='material_type'
    # check.value=True
    io=interactive_output(_split_feature, {'fd1':fd1,'fd2':fd2,'fd3':fd3,'check':check,'sl1':sl1,'sl2':sl2,'sl3':sl3,'df':fixed(df)})
    return VBox([HBox([fd1, sl1, check]), io, HBox([VBox([fd2,sl2]), VBox([fd3,sl3])])])

def _neural_network(step, show, check, predict, X, y, Xp):
    # this example adapted from 
    # https://iamtrask.github.io/2015/07/12/basic-python-network/
    # "A Neural Network in 11 lines of Python"
    def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    np.random.seed(1)

    # randomly initialize our weights with mean 0
    syn0 = 2*np.random.random((3,4)) - 1
    syn1 = 2*np.random.random((4,1)) - 1

    for j in np.arange(1+int(100*step)):

        # Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))

        # how much did we miss the target value?
        l2_error = y - l2
        
        # if (j% 100) == 0:
        #     print("Error:" + str(np.mean(np.abs(l2_error))))
            
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
        
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)

        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    if predict:        
        l0p = Xp
        l1p = nonlin(np.dot(l0p,syn0))
        yp = nonlin(np.dot(l1p,syn1))[0][0]
        
    f,ax=plt.subplots(1,1, figsize=(8,4))
    ws=[None, syn0, syn1]
    cmap = plt.get_cmap("seismic")
    for i,n in enumerate([3,4,1]):
        ys=np.arange(n)
        ys=ys-np.mean(ys)
        ax.plot(0*ys+i, ys, 'ko', mfc='w', ms=40, zorder=2)

        if i == 0:
            yo=1*ys
            if predict:
                for xi,yi,ti in zip(0*ys+i, ys, Xp[0,:]):
                    ax.text(xi,yi,f'{ti:.1f}',ha='center',va='center', color='r')
            elif show>0:
                for xi,yi,ti in zip(0*ys+i, ys, X[show-1,:]):
                    ax.text(xi,yi,f'{ti:.1f}',ha='center',va='center')
            continue
        w=ws[i]
        for j,y1 in enumerate(yo):
            for k,y2 in enumerate(ys):
                c=w[j,k]/(2*np.max(abs(w)))+0.5
                if abs(c-0.5)<0.05:
                    c='k'
                    ls='--'
                else:
                    c=cmap(c)
                    ls='-'
                ax.plot([i-1,i], [y1,y2], ls, color=c, lw=abs(2*w[j,k])+0.25, zorder=1)
                
        if predict and i == 1:
            for yi,ti in zip(ys, l1p[0,:]):
                ax.text(1,yi,f'{ti:.3f}',ha='center',va='center',color='r')
        elif show>0 and i == 1:
            for yi,ti in zip(ys, l1[show-1,:]):
                ax.text(1,yi,f'{ti:.3f}',ha='center',va='center')
        yo=1*ys
    
    if predict:
        ax.text(2,0.,f'{yp:.2f}',ha='center',va='center', color='r')
    elif show>0:
        ax.text(2,-0.15,f'({y[show-1,0]:.2f})',ha='center',va='center', alpha=0.6)
        ax.text(2,0.15,f'{l2[show-1,0]:.2f}',ha='center',va='center')
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-2.1, 2.6)

    if check:
        ax.text(0, 2.5, 'input\nlayer\n(features)', style='italic', ha='center', va='top')
        ax.text(1, 2.5, 'hidden\nlayer', style='italic', ha='center', va='top')
        ax.text(2, 2.5, 'output\nlayer\n(label)', style='italic', ha='center', va='top')
        
        xi=-0.3
        ax.text(xi, -1, 'current\nrainfall', style='italic', ha='right', va='center')
        ax.text(xi, 0, 'previous\nrainfall', style='italic', ha='right', va='center')
        ax.text(xi, 1, 'previous\nrunoff', style='italic', ha='right', va='center')
        ax.text(2-xi, 0, 'current\nrunoff', style='italic', ha='left', va='center')

    ax.text(2, -1., f'training\nerror\n={np.mean(np.abs(l2_error)):.2f}', ha='center', va='top')

    ax.axis('off')
    plt.show()
    return

def neural_network(X,y,Xp):    
    step = widgets.IntSlider(value=0, min=0, max=10, step=1, description="training steps")
    show = widgets.IntSlider(value=0, min=0, max=4, step=1, description="show datapoint")
    check = Checkbox(value=False, description="labels")
    predict = Checkbox(value=False, description="predict")
    io=interactive_output(_neural_network, {'step':step, 'show':show, 'check':check, 'predict':predict, 'X':fixed(X), 'y':fixed(y), 'Xp':fixed(Xp)})
    return VBox([HBox([VBox([step, show]), VBox([check, predict])]), io])

def _roc(threshold, Ntrees, data):

    X,y=data

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=Ntrees, max_depth=3, max_features=2, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Interactive function
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    i=np.argmin(abs(thresholds-threshold))
    roc_auc = auc(fpr, tpr)
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    if Ntrees !=10:
        model0 = RandomForestClassifier(n_estimators=10, max_depth=3, max_features=2, min_samples_leaf=10, random_state=42)
        model0.fit(X_train, y_train)
        y_probs = model0.predict_proba(X_test)[:, 1]

        # Interactive function
        fpr0, tpr0, thresholds = roc_curve(y_test, y_probs)

    plt.figure(figsize=(10, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    if Ntrees != 10:
        plt.plot(fpr0, tpr0, color='blue', lw=1, alpha=0.5, label=f'ROC curve (10 trees')
    plt.plot(fpr[i], tpr[i], 'ro', label=f'threshold={threshold:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Confusion Matrix
    plt.subplot(1, 2, 2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()

def load_enviro_data():
    fl='enviro_data.csv'
    if not os.path.isfile(fl):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        column_names = [
            'water_age', 'industrial_area', 'pollutant_type', 'flow_rate', 'chemical_oxygen_demand', 
            'agricultural_runoff', 'sensor_faults', 'biological_oxygen_demand', 'nearby_construction', 
            'turbidity', 'treatment_efficiency', 'contaminant_alerts', 'sampling_issues', 'contamination'
        ]
        data = pd.read_csv(url, names=column_names)
        # Preprocess the dataset
        data = data.replace('?', np.nan)
        data = data.dropna()

        # Convert data types
        data = data.astype(float)

        # Convert the contamination to binary (presence or absence of contamination)
        data['contamination'] = data['contamination'].apply(lambda x: 1 if x > 0 else 0)
        data.to_csv(fl, index=False)
    data=pd.read_csv(fl)   

    X = data.drop('contamination', axis=1)
    y = data['contamination']
    return X,y

def roc():
    X,y = load_enviro_data()

    Ntrees = widgets.Dropdown(value=10, options=[10, 20, 30], description="# trees")
    # check = Checkbox(value=False, description="lock root node")

    fs = widgets.FloatSlider(
        value=0.5, min=0.1, max=0.95,
        step=0.05,
        description="threshold:", continuous_update=False)
    data=(X,y)
    
    # _roc(fs.value, Ntrees.value, data)

    io=interactive_output(_roc, {'threshold':fs,'Ntrees':Ntrees,'data':fixed(data)})
    return VBox([HBox([fs, Ntrees]), io])

def _regression_performance(df, **kwargs):
    # Function to train, evaluate, and plot the model
    selected_features = [k for k,v in kwargs.items() if v]
    if not selected_features:
        print("Please select at least one feature.")
        return
    
    X,y=df

    X_selected = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, train_size=0.8, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # print(f'Selected Features: {selected_features}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')

    plt.figure(figsize=(10, 4))

    # Plot actual vs. predicted values
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel('Actual Integrity Score')
    plt.ylabel('Predicted Integrity Score')
    plt.title('Actual vs Predicted')

    # Plot residuals
    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, edgecolors=(0, 0, 0))
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='dashed')
    plt.xlabel('Predicted Integrity Score')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    # Plot residuals
    plt.subplot(1, 3, 3)
    residuals = (y_test - y_pred)/y_test*100
    plt.scatter(y_pred, residuals, edgecolors=(0, 0, 0))
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='dashed')
    plt.xlabel('Predicted Integrity Score')
    plt.ylabel('% error')
    plt.title('relative error')

    plt.tight_layout()
    plt.show()

def load_regression_data():
    # Read data from csv
    df = pd.read_csv('structural_data.csv')

    # Unpack variables
    X = df.drop(columns='integrity_score')
    y = df['integrity_score']

    return X,y

def regression_performance():
    X,y = load_regression_data()

    # Top five features for selection
    top_features = ['floor_area', 'pillar_ratio', 'foundation_type', 'load_bearing_walls', 'concrete_quality', 'building_age']

    checkboxes = [Checkbox(value=True, description=feature, layout=Layout(width='auto')) for feature in top_features]
    ui = HBox([VBox(checkboxes[:3], layout=Layout(padding='0px', width='auto')),
           VBox(checkboxes[3:], layout=Layout(padding='0px', width='auto'))], layout=Layout(padding='0px', width='auto'))

    
    inps=dict(zip(top_features,checkboxes))
    inps.update({'df':fixed((X,y))})

    io=interactive_output(_regression_performance, inps)
    return VBox([ui, io])

def _cross_validation(train_size, max_depth, df):
    
    time=np.linspace(0, 117, df.shape[0])

    # Feature and target extraction
    features=df.columns[:-1]
    X = df[features]  # replace with actual feature columns
    y = df['eyeDetection']  # replace with actual target column

    scaler = StandardScaler()
    X=scaler.fit_transform(X)

    # Convert the scaled features back to a DataFrame for saving
    X = pd.DataFrame(X, columns=features)

    # Calculate split index
    split_idx = int(len(X) * train_size / 100)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    clf = DecisionTreeClassifier(max_depth=max_depth)
    # clf = RandomForestClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    # Predict and calculate errors
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    in_sample_error = 1 - accuracy_score(y_train, y_train_pred)
    out_of_sample_error = 1 - accuracy_score(y_test, y_test_pred)

    # Plot errors
    f,ax=plt.subplots(1,1,figsize=(7, 3.5))
    ax.plot(time, y, 'k-', lw=1.5, label='data')
    ax.plot(time[:split_idx], y_train_pred, 'b-', lw=0.5, alpha=0.5, label=f'training: error - {in_sample_error:.2f}')
    ax.plot(time[split_idx:], y_test_pred, 'r-', lw=0.5, alpha=0.5, label=f'test: error - {out_of_sample_error:.2f}')
#     plt.plot(train_size, out_of_sample_error, label='Out-of-sample Error', marker='o')
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('Label')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['open', 'closed'])
    plt.show()

def cross_validation():

    # Load dataset (assuming a DataFrame `df` with necessary features and target)
    df = pd.read_csv('eye_movement.csv')  # replace with actual path

    # Interactive widgets
    train_size=IntSlider(min=50, max=90, step=10, value=70, description='Training Data Size (%)', continuous_update=False)
    max_depth=IntSlider(min=1, max=20, step=1, value=5, description='Max Depth', continuous_update=False)
    io=interactive_output(_cross_validation, {'train_size':train_size, 'max_depth':max_depth, 'df':fixed(df)})
    return VBox([HBox([train_size, max_depth]), io])

# ----- target functions for each complexity level -----
def target_function(X, n_terms):
    """n_terms = 1, 2, or 3 → progressively richer sine composition."""
    base = np.sin(2 * np.pi * X)                          # always present
    if n_terms == 1:
        return base
    second = 0.4 * np.sin(6 * np.pi * X)                 # + mid-frequency
    if n_terms == 2:
        return base + second
    third = 0.2 * np.sin(12 * np.pi * X)                  # + high-frequency
    return base + second + third

# ----- live LaTeX renderer ----------------------------------------------------
def make_equation_latex(n_terms):
    """Return LaTeX string with inactive terms greyed out."""
    colors = ["black" if i < n_terms else "gray" for i in range(3)]
    terms = [
        rf"\color{{{colors[0]}}}{{\sin(2\pi x)}}",
        rf"\color{{{colors[1]}}}{{+\,0.4\sin(6\pi x)}}",
        rf"\color{{{colors[2]}}}{{+\,0.2\sin(12\pi x)}}",
    ]
    # join without extra '+' in first slot (already included where needed)
    latex = r"".join(terms)
    return r"$$f(x)=" + latex + r"$$"

# ---------- helper to draw a simple network diagram ----------
def draw_network(layer_sizes, ax):
    ax.clear()
    h_spacing = 1.0
    v_spacing = 1.0
    max_neurons = max(layer_sizes)

    for layer_idx, neurons in enumerate(layer_sizes):
        x = layer_idx * h_spacing
        y_start = (max_neurons - neurons) * v_spacing / 2
        for n in range(neurons):
            y = y_start + n * v_spacing
            ax.scatter(
                x,
                y,
                s=300,
                facecolors="white",
                edgecolors="k",
                zorder=3,
            )
            if layer_idx > 0:
                prev_neurons = layer_sizes[layer_idx - 1]
                x_prev = (layer_idx - 1) * h_spacing
                y_prev_start = (max_neurons - prev_neurons) * v_spacing / 2
                for pn in range(prev_neurons):
                    y_prev = y_prev_start + pn * v_spacing
                    ax.plot(
                        [x_prev, x],
                        [y_prev, y],
                        "k-",
                        linewidth=0.4,
                        zorder=1,
                    )

    ax.set_xlim(-0.5, (len(layer_sizes) - 1) * h_spacing + 0.5)
    ax.set_ylim(-0.5, max_neurons * v_spacing - 0.5)
    ax.axis("off")

def function_approximation():
    # ---------- widgets ----------
    layers_slider = IntSlider(
        value=1, min=1, max=8, step=1, description="Hidden layers", continuous_update=False
    )
    width_slider = IntSlider(
        value=4, min=4, max=24, step=5, description="Neurons / layer", continuous_update=False
    )
    complexity_dd = Dropdown(
    options=[("1-term", 1), ("2-term", 2), ("3-term", 3)],
    value=1,
    description="Complexity"
    )
    out = Output()
    
    # ---------- training + plotting routine ----------
    def refresh(n_layers, n_width, n_terms):
        X = np.linspace(-1, 1, 400).reshape(-1, 1)
        y = target_function(X, n_terms)

        model = MLPRegressor(
            hidden_layer_sizes=(n_width,) * n_layers,
            activation="tanh",
            solver="lbfgs",
            max_iter=1500,
            random_state=0,
        )
        model.fit(X, y.ravel())
        y_pred = model.predict(X)
        mse = np.mean((y_pred - y.ravel()) ** 2)

        with out:
            clear_output(wait=True)
            display(Math(make_equation_latex(n_terms)))
            fig, (ax_fit, ax_net) = plt.subplots(1, 2, figsize=(12, 4))

            # --- Fit subplot ---
            ax_fit.plot(X, y, label="Target", lw=2)
            ax_fit.plot(
                X,
                y_pred,
                label=f"NN ({n_layers} × {n_width})\nMSE={mse:0.4f}",
                lw=2,
            )
            ax_fit.set_xlabel("x")
            ax_fit.set_ylabel("y")
            ax_fit.set_title("Function approximation")
            ax_fit.grid(alpha=0.3)
            ax_fit.legend(loc="upper right")

            # --- Network diagram subplot ---
            draw_network([1] + [n_width] * n_layers + [1], ax_net)
            ax_net.set_title("Network architecture")

            plt.tight_layout()
            plt.show()

    # initial render
    refresh(layers_slider.value, width_slider.value, complexity_dd.value)


    # ----- callbacks -----
    def on_widget_change(change):
        if change["name"] == "value":
            refresh(layers_slider.value, width_slider.value, complexity_dd.value)

    for w in (layers_slider, width_slider, complexity_dd):
        w.observe(on_widget_change)

    display(VBox([VBox([layers_slider, width_slider, complexity_dd]), out]))



class GridWorldMDP:
    """Interactive 5 × 5 grid-world with value-iteration and widget controls."""

    # ---------- 1. Static environment definition ----------
    GRID   = (5, 5)            # rows, cols
    START  = (4, 0)            # bottom-left
    GOAL   = (0, 4)            # top-right
    WALLS  = {(1, 1),(1, 2),(2,3)}          # add more walls here

    ACTIONS = {0: ("↑", (-1, 0)),
               1: ("→", ( 0, 1)),
               2: ("↓", ( 1, 0)),
               3: ("←", ( 0,-1))}
    ACTION_IDS = list(ACTIONS.keys())

    STEP_PENALTY = -1.0
    GOAL_REWARD  = +10.0
    WALL_PENALTY = -1.0        # for bumping into a wall / edge

    # ---------- 2. Construction ----------
    def __init__(self):
        # state
        self.print=None
        self.current_state = self.START
        self.total_reward  = 0.0
        self.manual_policy = {}
        self.random_policy = {}    # state  -> action_id

        # figure
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        plt.close(self.fig)                     # keep it hidden until we display()
        self.fig_out = Output()
        with self.fig_out:
            display(self.fig)

        # widgets
        self.policy_dd = Dropdown(options=['Greedy', 'Random', 'Manual'],
                                  description='Policy')
        self.gamma_slider = FloatSlider(value=0.9, min=0.0, max=1.0,
                                        step=0.05, description='γ')
        self.show_policy_chk = Checkbox(value=True, description='Show arrows')

        self.action_radio = RadioButtons(
            options=[('↑', 0), ('→', 1), ('↓', 2), ('←', 3)],
            description='Action'
        )
        self.step_btn  = Button(description='Step')
        self.reset_btn = Button(description='Reset')
        self.state_lab = Label()
        self.reward_lab = Label()

        # value function
        self.V = self.compute_V_optimal(self.gamma_slider.value)

        # wire-up callbacks
        self.step_btn.on_click(self.on_step)
        self.reset_btn.on_click(self.on_reset)
        self.show_policy_chk.observe(self.on_show_arrows)

        self.gamma_slider.observe(self.on_gamma, names='value')
        self.policy_dd.observe(self.on_policy_change, names='value')
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        
        # layout
        controls = VBox([
            self.policy_dd,
            self.gamma_slider,
            self.action_radio,
            self.step_btn,
            self.reset_btn,
            self.show_policy_chk,
            # self.state_lab,
            # self.reward_lab,
        ])
        display(VBox([HBox([self.fig_out, controls])]))

        # first draw
        self.update_display()

    # ---------- 3. Environment helpers ----------
    def in_bounds(self, s):
        return 0 <= s[0] < self.GRID[0] and 0 <= s[1] < self.GRID[1]

    def step(self, state, a_id):
        dr, dc = self.ACTIONS[a_id][1]
        nxt = (state[0] + dr, state[1] + dc)
        if not self.in_bounds(nxt) or nxt in self.WALLS:   # blocked
            nxt, r = state, self.WALL_PENALTY
        elif nxt == self.GOAL:
            r = self.GOAL_REWARD
        else:
            r = self.STEP_PENALTY
        return nxt, r

    # ---------- 4. Planning ----------
    def compute_V_optimal(self, gamma):
        V = np.zeros(self.GRID)
        while True:
            delta = 0.0
            for r in range(self.GRID[0]):
                for c in range(self.GRID[1]):
                    s = (r, c)
                    if s == self.GOAL or s in self.WALLS:
                        continue
                    v_old = V[s]
                    q_vals = []
                    for a in self.ACTION_IDS:
                        s2, rwd = self.step(s, a)
                        q_vals.append(rwd + gamma * V[s2])
                    V[s] = max(q_vals)
                    delta = max(delta, abs(v_old - V[s]))
            if delta < 1e-4:
                break
        return V

    def greedy_action(self, V, s):
        if s == self.GOAL:
            return None
        qs = []
        for a in self.ACTION_IDS:
            s2, r = self.step(s, a)
            qs.append(r + self.gamma_slider.value * V[s2])
        qs = np.asarray(qs, dtype=float)
        # round qs to avoid floating-point errors
        qs = np.round(qs, decimals=3)
        bests = np.flatnonzero(qs == qs.max())
        return int(bests[0])
        # return int(np.random.choice(bests))

    # ---------- 5. Drawing ----------
    def state_to_axcoords(self, s):
        r, c = s
        return c + 0.5, self.GRID[0] - r - 0.5

    def draw_grid(self):
        ax = self.ax
        ax.clear()

        # heat-map
        vmax = abs(self.V).max() or 1
        ax.imshow(self.V, cmap='coolwarm',
                  vmin=-vmax, vmax=vmax,
                  extent=[0, self.GRID[1], 0, self.GRID[0]])
        # ax.invert_yaxis()
        # grid lines
        ax.set_xticks(np.arange(0, self.GRID[1] + 1))
        ax.set_yticks(np.arange(0, self.GRID[0] + 1))
        ax.grid(color='k', lw=1)
        ax.set_xlim(0, self.GRID[1])
        ax.set_ylim(0, self.GRID[0])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # walls
        for w in self.WALLS:
            x, y = self.state_to_axcoords(w)
            ax.add_patch(
                plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black')
            )

        # goal
        gx, gy = self.state_to_axcoords(self.GOAL)
        ax.scatter(gy, gx, marker='*', s=400, c='gold', edgecolors='k')

        # agent
        x, y = self.state_to_axcoords(self.current_state)
        ax.scatter(x, y, marker='o', s=200, c='lime', edgecolors='k')

        # arrows for policy
        if self.show_policy_chk.value:
            for r in range(self.GRID[0]):
                for c in range(self.GRID[1]):
                    s = (r, c)
                    if s == self.GOAL or s in self.WALLS:
                        continue
                    if self.policy_dd.value == 'Greedy':
                        a = self.greedy_action(self.V, s)
                    elif self.policy_dd.value == 'Random':
                        a = self.random_policy.setdefault(s, np.random.choice(self.ACTION_IDS))
                        if s != self.current_state:
                            continue
                    else:      # Manual
                        a = self.manual_policy.get(s, 1)
                    dx, dy = self.ACTIONS[a][1][1], -self.ACTIONS[a][1][0]
                    ax.arrow(c + 0.5, self.GRID[0] - r - 0.5,
                             dx * 0.25, dy * 0.25,
                             head_width=0.15, head_length=0.15,
                             fc='k', ec='k')

        ax.set_aspect('equal')

    # ---------- 6. Display update ----------
    def update_display(self):
        self.draw_grid()
        debug=False
        if debug:
            self.ax.set_title(str(self.print))
        else:
            self.ax.set_title(
                f"State: {self.current_state}    |    "
                f"Cumulative reward: {self.total_reward:.1f}",
                fontsize=12
            )

        self.fig_out.clear_output(wait=True)
        with self.fig_out:
            display(self.fig)

    # ---------- 7. Callbacks ----------
    def on_step(self, _btn):
        if self.current_state == self.GOAL:
            return
        if self.policy_dd.value == 'Greedy':
            a = self.greedy_action(self.V, self.current_state)
            self.print=a
        elif self.policy_dd.value == 'Random':
            # a = np.random.choice(self.ACTION_IDS)
            a = self.random_policy.setdefault(
                    self.current_state, np.random.choice(self.ACTION_IDS))
            self.random_policy.clear()     # new random arrows next redraw
        else:                                   # Manual
            a = self.manual_policy.get(self.current_state,
                                       self.action_radio.value)

        self.current_state, r = self.step(self.current_state, a)
        self.total_reward += r
        self.update_display()

    def on_show_arrows(self, _btn):
        self.update_display()

    def on_reset(self, _btn):
        self.current_state = self.START
        self.total_reward  = 0.0
        self.update_display()

    def on_gamma(self, change):
        self.V = self.compute_V_optimal(change['new'])
        self.update_display()

    def on_policy_change(self, _change):
        if _change["new"] == "Random":
            self.random_policy.clear()     # new random arrows next redraw
        self.update_display()

    def onclick(self, event):
        if self.policy_dd.value != 'Manual' or event.inaxes != self.ax:
            return
        col, row = int(event.xdata), self.GRID[0] - int(event.ydata) - 1
        if (not self.in_bounds((row, col)) or
                (row, col) in self.WALLS or
                (row, col) == self.GOAL):
            return
        current = self.manual_policy.get((row, col), 0)
        self.manual_policy[(row, col)] = (current + 1) % 4
        self.update_display()


class TrafficSignalMDP:
    """Single-intersection toy simulator with user-controlled cycle lengths.

    NEW  (v3 – 2025-05-15)
    ─────────────────────────────────────────────────────────────────────────
    • Vehicles that leave the junction now re-appear on the **opposite side
      but with the same lateral offset** as their entry lane.
    • Each change of phase inserts a **2-second yellow interval** in which
      no traffic is discharged and the previously-green lanes are shown in
      yellow.
    """

    ACTIONS = {0: "NS-green", 1: "EW-green"}         # 0 = NS, 1 = EW

    # ─────────────────────────── construction ──────────────────────────────
    def __init__(self, horizon=300, mu_service=0.9):
        # traffic state ------------------------------------------------------
        self.horizon = horizon
        self.mu = mu_service
        self.queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self.phase = 0              # start with NS green
        self.time_in_phase = 0      # s already spent in current (green) phase
        self.in_yellow = False      # ⇠ NEW
        self.yellow_elapsed = 0     # ⇠ NEW: how many yellow seconds so far
        self.t = 0
        self.total_reward = 0.0
        self.just_served = {k: 0 for k in self.queues}

        # figure -------------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        plt.close(self.fig)
        self.fig_out = Output()
        with self.fig_out:
            display(self.fig)

        # widgets ------------------------------------------------------------
        self.dur_NS = IntSlider(10, 1, 60, 1, description='NS green s')
        self.dur_EW = IntSlider(10, 1, 60, 1, description='EW green s')

        self.lam_NS = FloatSlider(value=0.30, min=0, max=1, step=0.05,
                                  description='λ NS veh/s')
        self.lam_EW = FloatSlider(value=0.30, min=0, max=1, step=0.05,
                                  description='λ EW veh/s')

        self.phase_radio = RadioButtons(
            options=[('NS-green', 0), ('EW-green', 1)],
            description='Current'
        )

        self.step_btn = Button(description='Step (1 s)')
        self.step20_btn = Button(description='Step ×20')
        self.reset_btn = Button(description='Reset')

        # callbacks ----------------------------------------------------------
        self.step_btn.on_click(lambda _: self.simulate(1))
        self.step20_btn.on_click(lambda _: self.simulate(20))
        self.reset_btn.on_click(self.on_reset)

        self.phase_radio.observe(
            lambda ch: setattr(self, "phase", ch["new"]), names="value"
        )

        # layout -------------------------------------------------------------
        controls = VBox([
            self.dur_NS, self.dur_EW,
            self.lam_NS, self.lam_EW,
            self.phase_radio,
            HBox([self.step_btn, self.step20_btn]),
            self.reset_btn,
        ])
        display(VBox([HBox([self.fig_out, controls])]))

        self.update_display()

    # ─────────────────────── core simulation ───────────────────────────────
    def simulate(self, n_seconds):
        """Run *n_seconds* of 1-s ticks, switching phase automatically when
        the programmed green time elapses (with a 2 s yellow clearance)."""
        for _ in range(n_seconds):
            if self.t >= self.horizon:
                break
            reward = self._tick_once()
            self.total_reward += reward
            self.t += 1
        self.update_display()

    def _tick_once(self):
        """One-second environment tick."""
        # reset log of departures
        self.just_served = {k: 0 for k in self.queues}

        # 1) Poisson arrivals – independent λ for NS and EW approaches
        for k in ['N', 'S']:
            self.queues[k] += np.random.poisson(self.lam_NS.value)
        for k in ['E', 'W']:
            self.queues[k] += np.random.poisson(self.lam_EW.value)

        # 2) Departures (only if not in yellow)
        green = ['N', 'S'] if self.phase == 0 else ['E', 'W']
        if not self.in_yellow:
            for k in green:
                served = min(self.queues[k], 2)        # 2 veh/s cap
                self.queues[k] -= served
                self.just_served[k] = served

        # 3) Reward (negative queue; 2 s lost-time penalty when yellow starts)
        reward = -sum(self.queues.values())

        # 4) Timing / phase logic -------------------------------------------
        if self.in_yellow:
            # currently in yellow; count its length then flip phase
            self.yellow_elapsed += 1
            if self.yellow_elapsed >= 2:               # 2 s yellow done
                self.in_yellow = False
                self.phase = 1 - self.phase            # actual switch now
                self.phase_radio.value = self.phase
                self.time_in_phase = 0                 # reset green timer
        else:
            # currently green; advance its timer
            self.time_in_phase += 1
            dur_target = (self.dur_NS.value if self.phase == 0
                          else self.dur_EW.value)
            if self.time_in_phase >= dur_target:
                # begin 2 s yellow clearance
                self.in_yellow = True
                self.yellow_elapsed = 0
                reward -= 2.0                          # lost-time penalty

        return reward

    # ─────────────────────────── drawing ────────────────────────────────────
    def draw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, 5); ax.set_ylim(0, 5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
        ax.set_facecolor('white')
        ax.add_patch(plt.Rectangle((2, 2), 1, 1, fc='lightgrey', ec='k'))

        # queue bars ---------------------------------------------------------
        lane_geom = dict(
            N=((2.1, 3.0),  0,  1),
            S=((2.6, 2.0),  0, -1),
            E=((3.0, 2.1),  1,  0),
            W=((2.0, 2.6), -1,  0)
        )
        max_q = max(1, max(self.queues.values()))
        green_now = ['N', 'S'] if self.phase == 0 else ['E', 'W']
        for k, ((x0, y0), dx, dy) in lane_geom.items():
            # colour logic ---------------------------------------------------
            if self.in_yellow and k in green_now:
                fc = 'gold'               # yellow clearance
            elif (self.phase == 0 and k in ['N', 'S']) or \
                 (self.phase == 1 and k in ['E', 'W']):
                fc = 'tab:green'          # current green phase
            else:
                fc = 'tab:red'            # red

            # draw queue bar --------------------------------------------------
            q = self.queues[k]
            length = 1.5 * (q / max_q)
            ax.add_patch(plt.Rectangle(
                (x0, y0),
                dx * length if dx else 0.3,
                dy * length if dy else 0.3,
                fc=fc, ec='k'))
            text_x = x0 + (dx * length) / 2 + (0.15 if dx == 0 else 0)
            text_y = y0 + (dy * length) / 2 + (0.15 if dy == 0 else 0)
            ax.text(text_x, text_y, str(q),
                    ha='center', va='center', fontsize=10, color='w')

        # departing vehicles -------------------------------------------------
        # (nothing drawn during yellow → just_served all zeros)
        xc, yc = 2.5, 2.5                        # intersection centre
        for k, served in self.just_served.items():
            if served == 0:
                continue
            (x0, y0), dx_q, dy_q = lane_geom[k]
            dx_move, dy_move = -dx_q, -dy_q      # direction of travel

            # start just inside the junction
            start = np.array([xc + 0.15 * dx_move,
                              yc + 0.15 * dy_move])
            # shift laterally so the exit lane has the same offset
            if dx_move == 0:                     # N/S traffic → adjust x
                start[0] += (x0 - xc) + 0.15
            else:                               # E/W traffic → adjust y
                start[1] += (y0 - yc) + 0.15

            # end point a little downstream of the centre
            end = start + np.array([dx_move, dy_move]) * 0.6

            # open green square at end point
            ax.add_patch(plt.Rectangle(
                end - 0.15, 0.3, 0.3,
                fill=False, lw=1.8, ec='green'))

            # arrow from start → end
            ax.annotate("",
                        xy=end,
                        xytext=start,
                        arrowprops=dict(arrowstyle="->",
                                        lw=1.5,
                                        color='green'))

        # headline -----------------------------------------------------------
        phase_name = 'NS' if self.phase == 0 else 'EW'
        bar = 'yellow' if self.in_yellow else 'green'
        ax.set_title(
            f"t = {self.t}s   "#phase = {phase_name} ({bar})   "
            f"time in phase = {self.time_in_phase}s   "
            f"\nΣQ = {sum(self.queues.values())}   "
            f"reward cum = {self.total_reward:.1f}"
        )

    # ───────────────────────── GUI update ───────────────────────────────────
    def update_display(self):
        self.draw()
        self.fig_out.clear_output(wait=True)
        with self.fig_out:
            display(self.fig)

    # ─────────────────────────── reset ──────────────────────────────────────
    def on_reset(self, _btn):
        self.queues = {k: 0 for k in self.queues}
        self.just_served = {k: 0 for k in self.just_served}
        self.phase = 0
        self.time_in_phase = 0
        self.in_yellow = False
        self.yellow_elapsed = 0
        self.t = 0
        self.total_reward = 0.0
        self.update_display()

if __name__=="__main__":
    _cross_validation(50,1,pd.read_csv('eye_movement.csv'))
    # regression_performance()
    # roc()