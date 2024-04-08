#encn404.py
from ipywidgets import*
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
