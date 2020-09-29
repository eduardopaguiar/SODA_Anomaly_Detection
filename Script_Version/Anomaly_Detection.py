import numpy as np
import pandas as pd
import threading
import time
import pickle
import math
from psutil import cpu_percent, swap_memory
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist, squareform
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import os
import Functions as fu
import seaborn as sn
from scipy.stats import moment
from datetime import datetime

# Adding paths for the code to the model's folders
        
add_path1 = "/PCA_Analyses/"
add_path2 = "/Input/"
add_path3 = "/.Kernel/"
add_path4 = "/PCA_Analyses/Figures/"
add_path5 = "/Recovery/"
add_path6 = "/SODA/"
base_path = os.getcwd()
PCA_Analyses_path = base_path + add_path1
Input_path = base_path + add_path2
Kernel_path = base_path + add_path3
PCA_Figures_path = base_path + add_path4
Recovery_path = base_path + add_path5
SODA_path = base_path + add_path6

####### Variables set by user #######

# Number of Data-set divisions
windows = 100

# Number of events in offline and online phase
offline_samples = 5000
online_samples = 1000

# PCA number of components

N_PCs = 8

# Range of SODA granularities

min_granularity = 1
max_granularity = 30

# The cell below declares variables for perfomance analysis:

class performance(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.control = True
    
    def run(self):
        cpu_p = []
        ram_p = []
        ram_u = []
        while self.control:
            cpu_p.append(cpu_percent(interval=1, percpu=True))
            ram_p.append(swap_memory().percent)
            ram_u.append(swap_memory().used/(1024**3))
        self.mean_cpu_p = np.mean(cpu_p)
        self.mean_ram_p = np.mean(ram_p)
        self.mean_ram_u = np.mean(ram_u)
        self.max_cpu_p = np.max(np.mean(cpu_p, axis=1))
        self.max_ram_p = np.max(ram_p)
        self.max_ram_u = np.max(ram_u)
    
    def stop(self):
        self.control = False
    
    def join(self):
        threading.Thread.join(self)
        out = {'mean_cpu_p': self.mean_cpu_p,
               'mean_ram_p': self.mean_ram_p,
               'mean_ram_u': self.mean_ram_u,
               'max_cpu_p': self.max_cpu_p,
               'max_ram_p': self.max_ram_p,
               'max_ram_u': self.max_ram_u}
        return out

# Firstly the model loads the background and signal data, then it removes the 
# attributes first string line, in order to avoid NaN values in the array.

# Changing to the Input folder

os.chdir( Input_path )

# Loading data into the code
  
### Background    

b_name='Input_Background_1.csv'

background = np.genfromtxt(b_name, delimiter=',')
background = background[1:,:]

### Signal

s_name='Input_Signal_1.csv'

signal = np.genfromtxt(s_name, delimiter=',')
signal = signal[1:,:]

# Changing to base folder:

os.chdir( base_path )

# Devide data-set into training and testing sub-sets

background_train, background_test = train_test_split(background, test_size=0.80, random_state=42)

# Defining number of events from Backgorund and Singal into the online sub-set
background_samples = int(online_samples * 0.99)
signal_samples = int(online_samples - background_samples)

# DIVIDE BACKGROUND
reduced_background, background_sample_id = fu.divide(background, windows,offline_samples)

# Devide online Backgound 
reduced_background_on, background_on_sample_id = fu.divide (background_test, windows, background_samples)

# Devide online signal
reduced_signal, signal_sample_id = fu.divide(signal, windows, signal_samples)

# Concatanating IDs and creating labels
sample_id = np.concatenate((background_on_sample_id, signal_sample_id), axis=0)

# Nextly, the reduced data is saved in the Recovery directory.

# Changing to Recovery folder

os.chdir( Recovery_path )

np.savetxt('Reduced_' + b_name,reduced_background,delimiter=',')
np.savetxt('Reduced_ID_' + b_name,background_sample_id,delimiter=',')


np.savetxt('Reduced_online_' + b_name,reduced_background_on,delimiter=',')
np.savetxt('Reduced_ID_online' + b_name,background_on_sample_id,delimiter=',')


np.savetxt('Reduced_' + s_name,reduced_signal,delimiter=',')
np.savetxt('Reduced_ID_' + s_name,signal_sample_id,delimiter=',')

# Changing to base folder

os.chdir( base_path )

# Aiming to improve data patterns differentiation, in the following cell, we calculate the
# [moments](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html) 
# of each event regarding its 21 attributes. The moments used were of 2nd ([variance]
# (https://en.wikipedia.org/wiki/Variance)), 
# 3rd ([skewness](https://en.wikipedia.org/wiki/Skewness)) and 
# 4th ([kurtosis](https://en.wikipedia.org/wiki/Kurtosis)) order. 
# Check the links for further information.

moment2 = moment(reduced_background.transpose(), moment=2).reshape(-1,1)
moment3 = moment(reduced_background.transpose(), moment=3).reshape(-1,1)
moment4 = moment(reduced_background.transpose(), moment=4).reshape(-1,1)

reduced_background = np.concatenate((reduced_background,moment2,moment3,moment4), axis=1)
       
moment2 = moment(reduced_background_on.transpose(), moment=2).reshape(-1,1)
moment3 = moment(reduced_background_on.transpose(), moment=3).reshape(-1,1)
moment4 = moment(reduced_background_on.transpose(), moment=4).reshape(-1,1)

reduced_background_on = np.concatenate((reduced_background_on,moment2,moment3,moment4), axis=1)

moment2 = moment(reduced_signal.transpose(), moment=2).reshape(-1,1)
moment3 = moment(reduced_signal.transpose(), moment=3).reshape(-1,1)
moment4 = moment(reduced_signal.transpose(), moment=4).reshape(-1,1)

reduced_signal = np.concatenate((reduced_signal,moment2,moment3,moment4), axis=1)

### PCA:
# The principal component analysis method is calculated only with the background data, aiming to maximize anomalies differentiation.

# Changing to Kernel folder

os.chdir( PCA_Analyses_path )

background_scaler = StandardScaler().fit(reduced_background)
standard_data = background_scaler.transform(reduced_background)

pca= PCA(n_components = N_PCs)
pca.fit(standard_data)
        
# save the model to disk

pickle.dump(pca, open('pca.sav', 'wb'))
        
variacao_percentual_pca = np.round(pca.explained_variance_ratio_ * 100, decimals = 2)
        
# Now change to PCA Figures directory

os.chdir( PCA_Figures_path )
        
fig = plt.figure(figsize=[16,8])
ax = fig.subplots(1,1)
ax.bar(x=['PC' + str(x) for x in range(1,(N_PCs+1))],height=variacao_percentual_pca[0:N_PCs])

ax.set_ylabel('Percentage of Variance Held',fontsize=20)
ax.set_xlabel('Principal Components',fontsize=20)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=18)
ax.grid()
#plt.show()
fig.savefig('Percentage_of_Variance_Held_1.png', bbox_inches='tight')

print('Variation maintained: %.2f' % variacao_percentual_pca.sum())

# Now change to base directory

os.chdir( base_path )

# Then a PCA variation and relevance of attributes analysis is performed.

# Now change to PCA Figures directory

os.chdir( PCA_Figures_path )

### Attributes analyses ###

background_df = pd.DataFrame(reduced_background, columns=["px1","py1","pz1","E1","eta1","phi1","pt1","px2","py2",\
                                                  "pz2","E2","eta2","phi2","pt2","Delta_R","M12","MET","S",\
                                                  "C","HT","A","M2","M3","M4"])

eigen_matrix = np.array(pca.components_)
eigen_matrix = pow((pow(eigen_matrix,2)),0.5) #invertendo valores negativos

for i in range (eigen_matrix.shape[0]):

    LineSum = sum(eigen_matrix[i,:])
    for j in range (eigen_matrix.shape[1]):
        eigen_matrix[i,j] = ((eigen_matrix[i,j]*100)/LineSum)

weighted_contribution = np.zeros((2,eigen_matrix.shape[1]))

for i in range (eigen_matrix.shape[1]):
    NumeratorSum = 0
    for j in range (N_PCs):
        NumeratorSum += eigen_matrix[j,i] * variacao_percentual_pca[j]

    weighted_contribution[0,i] = NumeratorSum / sum(variacao_percentual_pca)

                    
sensors_contribution = pd.DataFrame (weighted_contribution, columns = background_df.columns)
                    
sensors_contribution = sensors_contribution.sort_values(by=0, axis=1,ascending=False)
                    
sorted_sensors_contribution = sensors_contribution.values[0,:] 
                    
background_df = background_df [sensors_contribution.columns]
                    
                    
#Ploting Cntribution Sensors Results
                    
fig = plt.figure(figsize=[20,8])

fig.suptitle('Attributes Weighted Contribution Percentage', fontsize=16)

ax = fig.subplots(1,1)

s = sorted_sensors_contribution[:]

ax.bar(x=sensors_contribution.columns,height=s)
plt.ylabel('Relevance Percentage',fontsize = 20)
plt.xlabel('Attributes',fontsize = 20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=18)
ax.grid()

#plt.show()
fig.savefig('Attributes_Weighted_Contribution_Percentage_1.png', bbox_inches='tight')



# Now change to base directory

os.chdir( base_path )

# Finaly, the online and offiline data are normalized and projected in the calculated PCs.

# Projecting Background
projected_background = pca.transform(standard_data)

# Normalizing online Background
scaler = StandardScaler().fit(reduced_background_on)
standard_background_on= scaler.transform(reduced_background_on)

# Projecting online Background
projected_background_on = pca.transform(standard_background_on)

# Normalizing online Signal
scaler = StandardScaler().fit(reduced_signal)
standard_signal = scaler.transform(reduced_signal)

# Projecting online Signal
projected_signal = pca.transform(standard_signal)

###### Formmating Original Data

# Formatting Background
Data1 = projected_background
Data1 = np.matrix(Data1)
L1, W = Data1.shape
        
# Formatting Online Background
Data2= projected_background_on
Data2 = np.matrix(Data2)
L2, W2 = Data2.shape

# Formatting Online Signal
Data3= projected_signal
Data3 = np.matrix(Data3)
L3, _ = Data3.shape

# Concatenating Online Background and Signal
streaming_data = np.concatenate((Data2,Data3), axis=0)
data = np.concatenate((Data1, streaming_data), axis=0)

delta = max_granularity - min_granularity + 1

# we create data frames to save each iteration result.

detection_info = pd.DataFrame(np.zeros((delta, 6)), columns=['Granularity','True_Positive', 'True_Negative','False_Positive','False_Negative', 'N_Groups'])

performance_info = pd.DataFrame(np.zeros((delta, 8)), columns=['Granularity', 'Time_Elapsed',
                                                            'Mean CPU_Percentage', 'Max CPU_Percentage',
                                                            'Mean RAM_Percentage', 'Max RAM_Percentage',
                                                            'Mean RAM_Usage_GB', 'Max RAM_Usage_GB'])

#### Method explanation:

# The SODA is a self-organized algorithm which, partitions a data-set into non-parametric 
# data-clouds. In the offline mode, we deliver to SODA a data-set compound only by normal 
# events (background). Afterward, in its online mode, SODA re-organizes those data-clouds 
# to follow the streaming data patterns. Thus, by analyzing the difference between data 
# clouds before and after a streaming data arrival, one can identify anomaly data 
# patterns (from the signal). This analysis calculates how much of the offline data is 
# inside each data-cloud after the streaming data arrival. Consequently, data-cloud with 
# more offline data is more similar to normal events (background). Those with no offline 
# data are regarded as anomaly data-clouds since they don't follow the offline data 
# patterns.



# Furthermore, considering that our testing data-set should respect the proportion of 99% 
# background and 1% signal, we label these events to ensure that they were correctly 
# classified. However, these labels do not take part in the model's decision.


#### Applying SODA to the original Data

os.chdir( SODA_path )

for gra in range (min_granularity, max_granularity+1):
    begin = datetime.now()

    i = gra - min_granularity

    performance_thread = performance()
    performance_thread.start()
    
    detection_info.loc[i,'Granularity'] = gra
    performance_info.loc[i,'Granularity'] = gra
    
    print('### Granularity ' + str(gra) + ' ###')

    Input = {'GridSize':gra, 'StaticData':Data1, 'DistanceType': 'euclidean'}

    out = fu.SelfOrganisedDirectionAwareDataPartitioning(Input,'Offline')

    # Concatanating IDs and creating labels
        
    new_label = np.zeros((int(online_samples)))
    new_label[background_samples:] = 1
    
    new_decision = np.zeros((int(online_samples)))
    
    Input['StreamingData'] = streaming_data
    Input['SystemParams'] = out['SystemParams']
    Input['AllData'] = data
    
    online_out = fu.SelfOrganisedDirectionAwareDataPartitioning(Input,'Evolving')

    
    signal_centers = online_out['C']
    soda_labels = online_out['IDX']
    online_soda_labels = soda_labels[(L1):]
    
    cloud_info = pd.DataFrame(np.zeros((len(signal_centers),4)),columns=['Total_Samples','Old_Samples','Percentage_Old_Samples', 'Percentage_of_Samples'])
    
    for j in range (len(soda_labels)):
        if j < L1:
            cloud_info.loc[int(soda_labels[j]),'Old_Samples'] += 1

        cloud_info.loc[int(soda_labels[j]),'Total_Samples'] += 1

    cloud_info.loc[:,'Percentage_Old_Samples'] = cloud_info.loc[:,'Old_Samples'] * 100 / cloud_info.loc[:,'Total_Samples']
    cloud_info.loc[:,'Percentage_of_Samples'] = cloud_info.loc[:,'Total_Samples'] * 100/ cloud_info.loc[:,'Total_Samples'].sum()

    
    anomaly_clouds=[]
    n_anomalies = 0

    for j in range(len(signal_centers)):
        if cloud_info.loc[j,'Percentage_Old_Samples'] == 0 :
            n_anomalies += cloud_info.loc[j,'Total_Samples']
            anomaly_clouds.append(j)
    
    if n_anomalies != 0:
        
        anomalies_df = pd.DataFrame(np.zeros((int(n_anomalies),2)), columns=['Sample_Id','Cloud_ID'])
        
        k = 0
        
        for j in range(len(online_soda_labels)): 
            if online_soda_labels[j] in anomaly_clouds:
                anomalies_df.loc[k,'Sample_Id'] = sample_id [j]
                anomalies_df.loc[k,'Cloud_ID'] = soda_labels[j]
                new_decision[j] = 1
                k +=1

        # Save Results 
        anomalies_df.to_csv('anomalies_Iterantion_' + str(gra) + '.csv')
        
    for j in range(len(new_label)):
        if new_label[j] == 1:
            if new_decision[j] == new_label[j]:
                detection_info.loc[i,'True_Positive'] += 1
            
            else:
                detection_info.loc[i,'False_Negative'] += 1
                
        else:
            if new_decision[j] == new_label[j]:
                detection_info.loc[i,'True_Negative'] += 1
            
            else:
                detection_info.loc[i,'False_Positive'] += 1


    
    detection_info.loc[i, 'N_Groups'] = max(soda_labels)+1

    performance_thread.stop()
    performance_out = performance_thread.join()
    final = datetime.now()
    performance_info.loc[i,'Time_Elapsed'] = (final - begin)
    performance_info.loc[i,'Mean CPU_Percentage'] = performance_out['mean_cpu_p']
    performance_info.loc[i,'Max CPU_Percentage'] = performance_out['max_cpu_p']
    performance_info.loc[i,'Mean RAM_Percentage'] = performance_out['mean_ram_p']
    performance_info.loc[i,'Max RAM_Percentage'] = performance_out['max_ram_p']
    performance_info.loc[i,'Mean RAM_Usage_GB'] = performance_out['mean_ram_u']
    performance_info.loc[i,'Max RAM_Usage_GB'] = performance_out['max_ram_u']

    print(50*'_')

    print(detection_info.loc[i,:])
    
    print(50*'_')
    
    print(performance_info.loc[i])

    print(50*'_')
    
    detection_info.to_csv('detection_info.csv', index=False)
    performance_info.to_csv('performance_info.csv', index=False)

print(detection_info)

print(50*'_')

print(performance_info)

print(50*'_')

os.chdir( Input_path )

# Load and formats the Original data analyse

accuracy_2 = np.genfromtxt('detection_Original_info_2.csv',delimiter=',')
accuracy_2[:,0] += 1
accuracy_df = pd.DataFrame(accuracy_2[1:],columns = ['Granularity','True_Positive','True_Negative','False_Positive','False_Negative'])

# Calculates the True Positive, and the False Positive rates

accuracy = []
threat_score = []
tp_rate = []
fp_rate = []
for index, row in accuracy_df.iterrows():
    accuracy_dict = row.to_dict()
    #confusionmatrix(accuracy_dict, gra[index])
    
    tp = accuracy_dict['True_Positive']
    tn = accuracy_dict['True_Negative']
    fp = accuracy_dict['False_Positive']
    fn = accuracy_dict['False_Negative']
    
    acc = ((tp + tn) / (tp + tn + fp + fn))*100

    ts = (tp / (tp + fp + fn))*100

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    
    #print('----------------- {} -------------------'.format(gra[index-1]))
    #print("True Positive Rate = {:.2f}".format(tpr*100))
    #print("False Positive Rate = {:.2f}".format(fpr*100))
    #print("Accuracy = {:.2f}".format(acc))
    #print("Threat Score = {:.2f}".format(ts))
    accuracy.append(acc)
    threat_score.append(ts)
    tp_rate.append(tpr)
    fp_rate.append(fpr)

accuracy2 = []
threat_score2 = []
tp_rate2 = []
fp_rate2 = []
for index, row in detection_info.iterrows():

    accuracy_dict = row.to_dict()
    #confusionmatrix(accuracy_dict, gra[index])
    
    tp = accuracy_dict['True_Positive']
    tn = accuracy_dict['True_Negative']
    fp = accuracy_dict['False_Positive']
    fn = accuracy_dict['False_Negative']
    
    acc = ((tp + tn) / (tp + tn + fp + fn))*100

    ts = (tp / (tp + fp + fn))*100

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    
    #print('----------------- {} -------------------'.format(gra[index-1]))
    #print("True Positive Rate = {:.2f}".format(tpr*100))
    #print("False Positive Rate = {:.2f}".format(fpr*100))
    #print("Accuracy = {:.2f}".format(acc))
    #print("Threat Score = {:.2f}".format(ts))
    accuracy2.append(acc)
    threat_score2.append(ts)
    tp_rate2.append(tpr)
    fp_rate2.append(fpr)

os.chdir( SODA_path )

# Plot Without Features extraction

fig = plt.figure(figsize=[20,8])

fig.suptitle('Original Attributes', fontsize=20)

ax = fig.subplots(1,1)

s = np.array(tp_rate) *100

ax.bar(x=['G' + str(x) for x in range(1,(len(s)+1))],height=s)
plt.ylabel('Detection Percentage (%)',fontsize = 20)
plt.xlabel('Granularities',fontsize = 20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=18)
ax.grid()

plt.show()
fig.savefig('Without_Feature_Extraction_tp.png', bbox_inches='tight')

fig = plt.figure(figsize=[20,8])

fig.suptitle('Original Attributes', fontsize=20)

ax = fig.subplots(1,1)

s = np.array(fp_rate) *100

barlist = ax.bar(x=['G' + str(x) for x in range(1,(len(s)+1))],height=s)

plt.ylabel('False Detection Percentage (%)',fontsize = 20)
plt.xlabel('Granularities',fontsize = 20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=18)
ax.grid()


plt.show()
fig.savefig('Without_Feature_Extraction_fp.png', bbox_inches='tight')

# Plot With Features extraction

fig = plt.figure(figsize=[20,8])

fig.suptitle('Original Attributes + Moments', fontsize=20)

ax = fig.subplots(1,1)

s = np.array(tp_rate2) *100

ax.bar(x=['G' + str(x) for x in range(1,(len(s)+1))],height=s)
plt.ylabel('Detection Percentage (%)',fontsize = 20)
plt.xlabel('Granularities',fontsize = 20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=18)
ax.grid()

plt.show()
fig.savefig('With_Feature_Extraction_tp.png', bbox_inches='tight')

fig = plt.figure(figsize=[20,8])

fig.suptitle('Original Attributes + Moments', fontsize=20)

ax = fig.subplots(1,1)

s = np.array(fp_rate2) *100

ax.bar(x=['G' + str(x) for x in range(1,(len(s)+1))],height=s)
plt.ylabel('False Detection Percentage (%)',fontsize = 20)
plt.xlabel('Granularities',fontsize = 20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=18)
ax.grid()


plt.show()
fig.savefig('With_Feature_Extraction_fp.png', bbox_inches='tight')


#Ploting Comparrisons
                    
fig = plt.figure(figsize=[20,8])

fig.suptitle('Percentage of Detection Difference', fontsize=20)

ax = fig.subplots(1,1)

s = (np.array((tp_rate2)) - np.array((tp_rate))) * 100

barlist = ax.bar(x=['G' + str(x) for x in range(1,len(s)+1)],height=s)

negative = []

for i in range(len(s)):
    if float(s[i]) < 0:
        negative.append(i)

for i in negative:
    barlist[i].set_color('r')
        
plt.ylabel('Difference (%)',fontsize = 20)
plt.xlabel('Granularities',fontsize = 20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=18)
ax.grid()

plt.show()
fig.savefig('True_Positive_Comparrison.png', bbox_inches='tight')

fig = plt.figure(figsize=[20,8])

fig.suptitle('Percentage of False Detection Difference', fontsize=20)

ax = fig.subplots(1,1)

s = (np.array((fp_rate2)) - np.array((fp_rate))) * 100

barlist = ax.bar(x=['G' + str(x) for x in range(1,(len(s)+1))],height=s)

negative = []

for i in range(len(s)):
    if float(s[i]) < 0:
        negative.append(i)
        
for i in negative:
    barlist[i].set_color('r')

plt.ylabel('Difference (%)',fontsize = 20)
plt.xlabel('Granularities',fontsize = 20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=18)
ax.grid()

plt.show()
fig.savefig('False_Positive_Comparrison.png', bbox_inches='tight')

os.chdir( base_path )