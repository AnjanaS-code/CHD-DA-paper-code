#!/usr/bin/env python
# coding: utf-8

# # Flowcal - python package for FACS analysis
# 
# #https://flowcal.readthedocs.io/en/latest/fundamentals/density_gate.html

# In[27]:


### importing packages for flowCal
import glob
import os
import FlowCal
import matplotlib.pyplot as plt


# # Experimental design - Cardiomyocytes - flow analysis
# # H9 ES cell derived Day 8 cardiomyocytes
# #  Samples - WT, TBX5 KO, Enh3 KO and Enh5 KO
# # Prime flow probes - FN1, RPL13A
# 

# In[28]:


## directory
os. getcwd()
os.chdir("/Users/anjana_home/Desktop/Gary_Hon/Data/FACS/Cardiomyocytes")


# # files

# In[29]:


## load fcs data
wt_probe= FlowCal.io.FCSData('wt probe.fcs')
wt_noprobe= FlowCal.io.FCSData('wt-no probe.fcs')
TBX5_probe = FlowCal.io.FCSData('tbx5 ko probe.fcs')
TBX5_noprobe = FlowCal.io.FCSData('tbx5_ko_no_probe.fcs')
enh3_probe = FlowCal.io.FCSData('enh3 ko probe.fcs')
enh3_noprobe = FlowCal.io.FCSData('enh3-ko no probe.fcs')
enh5_probe = FlowCal.io.FCSData('enh5 ko probe.fcs')
enh5_noprobe = FlowCal.io.FCSData('enh5 ko no probe.fcs')


# In[ ]:





# # GATING

# # Gating the cells on FSC-H and SS-H plot based on density gating

# In[30]:


##Function for gating

def flowcal_gating(sample, fraction):
    "flowcal_gating takes in raw fcs file and the fraction of cells to retain as input and returns the gataed cells"
    gated_cells =  FlowCal.gate.density2d(sample,
                             channels=['FSC-H', 'SSC-H'],
                              gate_fraction= fraction)
    
    return(gated_cells)
    

def gate_plot(gatedcells):
    "this function will plot the gated cells in a FSC-H and SSC-H  2D plot "
    plot =  FlowCal.plot.density2d(gatedcells,
                       channels=['FSC-H', 'SSC-H'],
                       mode='scatter')
    return(plot)
    


# In[31]:


# Function to plot FN1 and RPL13A histograms

def fluroscence_hist(ungated_sample,gated_sample,channel):
    hist_plot = FlowCal.plot.density_and_hist(ungated_sample,
                               gated_data= gated_sample,
                              #gate_contour= s_g2,
                               density_channels=['FSC-H', 'SSC-H'],
                               density_params={'mode':'scatter'},
                           hist_channels=[channel])
    plt.tight_layout()
    
    return(hist_plot)


# In[ ]:





# # wt probe sample

# In[32]:


s_g2 = flowcal_gating(wt_probe,0.45)
gate_plot(s_g2)


# # FN1

# In[33]:


fluroscence_hist(wt_probe,s_g2,"FL3-H")
plt.savefig('QC_WT_probe_FN1.pdf') 


# # RPL13A

# In[154]:


fluroscence_hist(wt_probe,s_g2,"FL1-H")
#plt.show()
plt.savefig('QC_WT_probe_RPL13A.pdf') 


# In[ ]:





# # wt no probe sample

# In[156]:


#FSC, SSC gating

s_g3 = flowcal_gating(wt_noprobe,0.45)
gate_plot(s_g3)


# # FN1

# In[157]:


fluroscence_hist(wt_noprobe,s_g3,"FL3-H")
plt.savefig('QC_WT_noprobe_FN1.pdf') 


# # RPL13A

# In[158]:


fluroscence_hist(wt_noprobe,s_g3,"FL1-H")
#plt.show()
plt.savefig('QC_WT_noprobe_RPL13A.pdf') 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # TBX5 KO probe

# In[159]:



s_g4 = flowcal_gating(TBX5_probe,0.45)
gate_plot(s_g4)
plt.show()


# # FN1

# In[39]:


fluroscence_hist(TBX5_probe,s_g4,"FL3-H")

#plt.show()
plt.savefig('QC_TBX5KO_probe_FN1.pdf') 


# In[ ]:





# # RPL13A

# In[40]:


fluroscence_hist(TBX5_probe,s_g4,"FL1-H")
#plt.show()
plt.savefig('QC_TBX5KO_probe_RPL13A.pdf') 


# In[ ]:





# In[ ]:





# #  TBX5 KO no probe

# In[160]:



s_g5 = flowcal_gating(TBX5_noprobe,0.45)
gate_plot(s_g5)
plt.show()

#plt.show()


# # FN1

# In[161]:


fluroscence_hist(TBX5_noprobe,s_g5,"FL3-H")
#plt.show()
plt.savefig('QC_TBX5KO_noprobe_FN1.pdf') 


# # RPL13A

# In[162]:


fluroscence_hist(TBX5_noprobe,s_g5,"FL1-H")
#plt.show()
#plt.savefig('QC_TBX5KO_noprobe_RPL13A.pdf') 


# In[ ]:





# In[ ]:





# # Enh3 KO probe

# In[163]:



s_g6 = flowcal_gating(enh3_probe,0.45)
gate_plot(s_g6)


# # FN1

# In[45]:


fluroscence_hist(enh3_probe,s_g6,"FL3-H")
#plt.show()
plt.savefig('QC_enh3KO_probe_FN1.pdf') 


# # RPL13A

# In[46]:


fluroscence_hist(enh3_probe,s_g6,"FL1-H")
#plt.show()
plt.savefig('QC_enh3KO_probe_RPL13A.pdf') 


# # Enh3 KO no probe

# In[164]:



s_g7 = flowcal_gating(enh3_noprobe,0.45)
gate_plot(s_g7)

#plt.show()


# # FN1

# In[48]:


fluroscence_hist(enh3_noprobe,s_g7,"FL3-H")

#plt.show()
plt.savefig('QC_enh3KO_noprobe_FN1.pdf') 


# # RPL13A

# In[49]:


fluroscence_hist(enh3_noprobe,s_g7,"FL1-H")

#plt.show()
plt.savefig('QC_enh3KO_noprobe_RPL13A.pdf') 


# # Enh5 KO probe

# In[165]:


s_g8 = flowcal_gating(enh5_probe,0.45)
gate_plot(s_g8)


# # FN1

# In[51]:


fluroscence_hist(enh5_probe,s_g8,"FL3-H")
#plt.show()
plt.savefig('QC_enh5KO_probe_FN1.pdf') 


# # RPL13A

# In[166]:


fluroscence_hist(enh5_probe,s_g8,"FL1-H")
#plt.show()
#plt.savefig('QC_enh5KO_probe_RPL13A.pdf') 


# In[ ]:





# # Enh5 KO no probe

# In[167]:


s_g9 = flowcal_gating(enh5_noprobe,0.45)
gate_plot(s_g9)


# # FN1

# In[54]:


fluroscence_hist(enh5_noprobe,s_g9,"FL3-H")
#plt.show()
plt.savefig('QC_enh5KO_noprobe_FN1.pdf') 


# # RPL13A

# In[55]:


fluroscence_hist(enh5_noprobe,s_g9,"FL1-H")
#plt.show()
plt.savefig('QC_enh5KO_noprobe_RPL13A.pdf') 


# # subsetting the fluroscence channels after gating

# In[61]:


##  probe samples
##wt probe
wt_probe_RPL13A = s_g2[:, ['FL1-H']] ##RPL13A##
wt_probe_FN1 = s_g2[:, ['FL3-H']] ###FN1##

## TBX5 probe #
TBX5_probe_RPL13A = s_g4[:, ['FL1-H']] ##RPL13A##
TBX5_probe_FN1 =   s_g4[:, ['FL3-H']] ###FN1##


### enhancer 3 probe###
enh3_probe_RPL13A = s_g6[:, ['FL1-H']] ##RPL13A##
enh3_probe_FN1 = s_g6[:, ['FL3-H']] ###FN1##

### enhancer 5 probe###
enh5_probe_RPL13A = s_g8[:, ['FL1-H']] ##RPL13A##
enh5_probe_FN1 = s_g8[:, ['FL3-H']] ###FN1##


# In[62]:


## No probe samples
##wt no probe
wt_noprobe_RPL13A = s_g3[:, ['FL1-H']] ##RPL13A##
wt_noprobe_FN1 = s_g3[:, ['FL3-H']] ###FN1##

## TBX5 no probe #
TBX5_noprobe_RPL13A = s_g5[:, ['FL1-H']] ##RPL13A##
TBX5_noprobe_FN1 =   s_g5[:, ['FL3-H']] ###FN1##


### enhancer 3 no probe###
enh3_noprobe_RPL13A = s_g7[:, ['FL1-H']] ##RPL13A##
enh3_noprobe_FN1 = s_g7[:, ['FL3-H']] ###FN1##

### enhancer 5 no probe###
enh5_noprobe_RPL13A = s_g9[:, ['FL1-H']] ##RPL13A##
enh5_noprobe_FN1 = s_g9[:, ['FL3-H']] ###FN1##


# # Violin plots of the fluroscence intensity

# # FN1 intensity

# In[69]:


x_pos = [0,0.5,1,1.5]
x_labels = ['Wildtype','TBX5 KO','Enh3 KO','Enh5 KO']
plt.violinplot(wt_probe_FN1, positions=[0], showextrema=False)
plt.boxplot(wt_probe_FN1, positions= [0], vert=True,notch = True,showfliers=False)

plt.violinplot(TBX5_probe_FN1, positions=[0.5], showextrema=False)
plt.boxplot(TBX5_probe_FN1, positions= [0.5], vert=True,notch = True,showfliers= False)

plt.violinplot(enh3_probe_FN1, positions=[1], showextrema=False)
plt.boxplot(enh3_probe_FN1, positions= [1], vert=True, notch = True,showfliers= False)

plt.violinplot(enh5_probe_FN1, positions=[1.5], showextrema=False)
plt.boxplot(enh5_probe_FN1, positions= [1.5], vert=True, notch = True,showfliers= False)
plt.xticks(x_pos,x_labels)
plt.ylabel("FN1 intensity", fontsize=16)
plt.savefig('FN1_intensity.pdf') 


# # RPL13A intensity

# In[70]:


x_pos = [0,0.5,1,1.5]
x_labels = ['Wildtype','TBX5 KO','Enh3 KO','Enh5 KO']
plt.violinplot(wt_probe_RPL13A, positions=[0], showextrema=False)
plt.boxplot(wt_probe_RPL13A, positions= [0], vert=True,notch = True,showfliers=False)

plt.violinplot(TBX5_probe_RPL13A, positions=[0.5], showextrema=False)
plt.boxplot(TBX5_probe_RPL13A, positions= [0.5], vert=True,notch = True,showfliers= False)

plt.violinplot(enh3_probe_RPL13A, positions=[1], showextrema=False)
plt.boxplot(enh3_probe_RPL13A, positions= [1], vert=True, notch = True,showfliers= False)

plt.violinplot(enh5_probe_RPL13A, positions=[1.5], showextrema=False)
plt.boxplot(enh5_probe_RPL13A, positions= [1.5], vert=True, notch = True,showfliers= False)
plt.xticks(x_pos,x_labels)
plt.ylabel("RPL13A intensity", fontsize=16)
plt.savefig('RPL13A_intensity.pdf') 


# # Normalizing the target mRNA intensity with control (RPL13A) at a single cell level

# In[71]:


## ratio of the CM markers/RPL13A
### wt probe###
wt_probe_FN1_ratio = wt_probe_FN1/wt_probe_RPL13A
##TBX5 probe##
TBX5_probe_FN1_ratio = TBX5_probe_FN1/TBX5_probe_RPL13A
# enh 3 probe ##
enh3_probe_FN1_ratio = enh3_probe_FN1/enh3_probe_RPL13A
# enh 5 probe ##
enh5_probe_FN1_ratio = enh5_probe_FN1/enh5_probe_RPL13A


# # Violin plots of the normalized FN1 intensity

# In[124]:


x_pos = [0,0.5,1,1.5]
x_labels = ['Wildtype','TBX5 KO','Enh3 KO','Enh5 KO']
plt.violinplot(wt_probe_FN1_ratio, positions=[0], showextrema=False)
plt.boxplot(wt_probe_FN1_ratio, positions= [0], vert=True,notch = True,showfliers=False)

plt.violinplot(TBX5_probe_FN1_ratio, positions=[0.5], showextrema=False)
plt.boxplot(TBX5_probe_FN1_ratio, positions= [0.5], vert=True,notch = True,showfliers= False)

plt.violinplot(enh3_probe_FN1_ratio, positions=[1], showextrema=False)
plt.boxplot(enh3_probe_FN1_ratio, positions= [1], vert=True, notch = True,showfliers= False)

plt.violinplot(enh5_probe_FN1_ratio, positions=[1.5], showextrema=False)
plt.boxplot(enh5_probe_FN1_ratio, positions= [1.5], vert=True, notch = True,showfliers= False)
plt.xticks(x_pos,x_labels)
plt.ylabel("FN1/RPL13A ratio", fontsize=16)



plt.ylim(0,200)
plt.savefig('FN1_RPL13A_ratio.pdf') 


# # statistical tests 

# In[173]:


# statistical tests###

import matplotlib.pyplot as plt
import seaborn as sns
#from statannot import add_stat_annotation
from statannotations.Annotator import Annotator

import math
import numpy as np
from scipy.stats import lognorm
import statsmodels.api as sm


# A few helper functions:
#from utils import *

# To illustrate examples
import numpy as np
from scipy.stats import mannwhitneyu, normaltest


# # normality test

# In[174]:


#log normal test

#create Q-Q plot with 45-degree line added to plot
fig = sm.qqplot(wt_probe_FN1_ratio, line='45')

plt.show()


# In[176]:


sm.qqplot(TBX5_probe_FN1_ratio, line='45')

plt.show()


# In[178]:


sm.qqplot(enh3_probe_FN1_ratio, line='45')

plt.show()


# In[179]:


sm.qqplot(enh5_probe_FN1_ratio, line='45')

plt.show()


# In[188]:


#KS test

from scipy.stats import kstest
kstest_results = [(kstest(wt_probe_FN1_ratio, 'norm')),(kstest(TBX5_probe_FN1_ratio, 'norm')),
                (kstest(enh3_probe_FN1_ratio, 'norm')),(kstest(enh5_probe_FN1_ratio, 'norm'))]
                                                    
print("wt_FN1_ratio: ", kstest_results[0])
print("TBX5_FN1_ratio: ", kstest_results[1])
print("Enh3_FN1_ratio: ", kstest_results[2])
print("Enh5_FN1_ratio: ", kstest_results[3])


# # using non parametric mannwhitneyu test

# In[193]:


stat_results = [mannwhitneyu(TBX5_probe_FN1_ratio,wt_probe_FN1_ratio),
                mannwhitneyu(enh3_probe_FN1_ratio,wt_probe_FN1_ratio ),
                mannwhitneyu(enh5_probe_FN1_ratio,wt_probe_FN1_ratio )]

print("TBX5 KO vs WT: ", stat_results[0])
print("Enh3 KO vs WT: ", stat_results[1])
print("Enh5 KO vs WT: ", stat_results[2])

pvalues = [result.pvalue for result in stat_results]


# # EXTRA QC CODE!

# In[135]:


plt.bar("WT",wt_probe_FN1_ratio.mean())
plt.bar("TBX5 KO",TBX5_probe_FN1_ratio.mean())
plt.bar("Enh3 KO",enh3_probe_FN1_ratio.mean())
plt.bar("Enh5 KO",enh5_probe_FN1_ratio.mean())

plt.rcParams["figure.figsize"] = (15,8)
plt.title("Mean FN1/RPL13A ratio")
plt.show()


# # violin plots for no probes

# # RPL13A

# In[250]:


## No probe violin plots for all samples 

x_pos = [0,0.5,1,1.5]
x_labels = ['Wildtype','TBX5 KO','Enh3 KO','Enh5 KO']
plt.violinplot(wt_noprobe_RPL13A, positions=[0], showextrema=False)
plt.boxplot(wt_noprobe_RPL13A, positions= [0], vert=True,notch = True,showfliers=False)

plt.violinplot(TBX5_noprobe_RPL13A, positions=[0.5], showextrema=False)
plt.boxplot(TBX5_noprobe_RPL13A, positions= [0.5], vert=True,notch = True,showfliers= False)

plt.violinplot(enh3_noprobe_RPL13A, positions=[1], showextrema=False)
plt.boxplot(enh3_noprobe_RPL13A, positions= [1], vert=True, notch = True,showfliers= False)

plt.violinplot(enh5_noprobe_RPL13A, positions=[1.5], showextrema=False)
plt.boxplot(enh5_noprobe_RPL13A, positions= [1.5], vert=True, notch = True,showfliers= False)
plt.xticks(x_pos,x_labels)
plt.ylabel("WT No probe RPL13A intensity", fontsize=16)
#plt.savefig('RPL13A_intensity.pdf') 


# # FN1

# In[251]:


x_pos = [0,0.5,1,1.5]
x_labels = ['Wildtype','TBX5 KO','Enh3 KO','Enh5 KO']
plt.violinplot(wt_noprobe_FN1, positions=[0], showextrema=False)
plt.boxplot(wt_noprobe_FN1, positions= [0], vert=True,notch = True,showfliers=False)

plt.violinplot(TBX5_noprobe_FN1, positions=[0.5], showextrema=False)
plt.boxplot(TBX5_noprobe_FN1, positions= [0.5], vert=True,notch = True,showfliers= False)

plt.violinplot(enh3_noprobe_FN1, positions=[1], showextrema=False)
plt.boxplot(enh3_noprobe_FN1, positions= [1], vert=True, notch = True,showfliers= False)

plt.violinplot(enh5_noprobe_FN1, positions=[1.5], showextrema=False)
plt.boxplot(enh5_noprobe_FN1, positions= [1.5], vert=True, notch = True,showfliers= False)
plt.xticks(x_pos,x_labels)
plt.ylabel("WT No probe FN1 intensity", fontsize=16)
#plt.savefig('RPL13A_intensity.pdf') 

