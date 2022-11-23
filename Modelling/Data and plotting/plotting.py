from pickle import FALSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from itertools import count
from itertools import permutations
import csv 

# Figure 1a
with open('figure_1a_data.csv') as f:
    reader = csv.reader(f)
    figure_1a_data = [[float(x) for x in el] for el in list(reader)]

figure_1a_xdata = figure_1a_data[0]
figure_1a_65 = figure_1a_data[1]
figure_1a_75 = figure_1a_data[2]
figure_1a_85 = figure_1a_data[3]
figure_1a_95 = figure_1a_data[4]

plt.figure("1a")
plt.plot(figure_1a_xdata, figure_1a_65,figure_1a_xdata, figure_1a_75,figure_1a_xdata, figure_1a_85,figure_1a_xdata, figure_1a_95)
plt.xlabel('$R_{eff}$',fontsize = 14)
plt.ylabel('Probability of detection within 7 days', fontsize= 14)
plt.legend(['Test sensitivity = 0.65', 'Test sensitivity = 0.75', 'Test sensitivity = 0.85', 'Test sensitivity = 0.95'],fontsize = 12)
plt.ylim(0, 1.05)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('./figure_1a.eps') 
plt.show()


# Figure 1b
with open('figure_1b_data.csv') as f:
    reader = csv.reader(f)
    figure_1b_data = [[float(x) for x in el] for el in list(reader)]

figure_1b_xdata = figure_1b_data[0]
figure_1b_1 = figure_1b_data[1]
figure_1b_3 = figure_1b_data[2]
figure_1b_7 = figure_1b_data[3]

# Figure 1b data 
xdata_temp = list(range(40,101,1))
figure_1b_xdata = [x/100 for x in xdata_temp]
figure_1b_data = []
testing_days= [1, 3, 7]

plt.figure("1b")
plt.plot(figure_1b_xdata, figure_1b_1,figure_1b_xdata, figure_1b_3,figure_1b_xdata, figure_1b_7)
plt.xlabel('Test sensitivity', fontsize = 14)
plt.ylabel('Probability of detection within 7 days', fontsize = 14)
plt.legend(['Weekly testing', '3/week testing', 'Daily testing'],fontsize =12)
plt.ylim(0,1.05)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('./figure_1b.eps') 
plt.show()


# Figure 1b
with open('supp_figure.csv') as f:
    reader = csv.reader(f)
    supp_figure_data = [[float(x) for x in el] for el in list(reader)]

supp_figure_xdata = supp_figure_data[0]
average_probs = [x[0] for x in supp_figure_data[1:]]
min_probs = [x[1] for x in supp_figure_data[1:]]
max_probs = [x[2] for x in supp_figure_data[1:]]

plt.figure("supp_figure")
plt.plot(supp_figure_xdata, average_probs)
plt.fill_between(supp_figure_xdata, min_probs, max_probs,alpha = .25, color = 'b')
plt.xlabel('Number of times testing occurs per week', fontsize = 14)
plt.ylabel('Probability of detection', fontsize = 14)
plt.savefig('./supp_figure.eps')
plt.show()


