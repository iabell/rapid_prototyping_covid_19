from pickle import FALSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from itertools import count
from itertools import permutations
import csv 
from pathlib import Path

def load_data():
    data = {}
    # Figure 1a
    with open(Path(__file__).parent/'figure_1a_data.csv') as f:
        reader = csv.reader(f)
        data.update({'figure_1a_data': [[float(x) for x in el] for el in list(reader)]}) 
    
    # Figure 1b
    with open(Path(__file__).parent/'figure_1b_data.csv') as f:
        reader = csv.reader(f)
        data.update({'figure_1b_data': [[float(x) for x in el] for el in list(reader)]})

    # Supplementary figure 
    with open(Path(__file__).parent/'supp_figure_data.csv') as f:
        reader = csv.reader(f)
        data.update({'supp_figure_data': [[float(x) for x in el] for el in list(reader)]})

    # Figure 2a
    with open(Path(__file__).parent/'figure_2a_data.csv') as f:
        reader = csv.reader(f)
        data.update({'figure_2a_data': [[float(x) for x in el] for el in list(reader)]})

    # Figure 2b
    with open(Path(__file__).parent/'figure_2b_data.csv') as f:
        reader = csv.reader(f)
        data.update({'figure_2b_data': [[float(x) for x in el] for el in list(reader)]})

    # Figure 3a
    with open(Path(__file__).parent/'figure_3a_data.csv') as f:
        reader = csv.reader(f)
        data.update({'figure_3a_data': [[float(x) for x in el] for el in list(reader)]})

    # Figure 3b
    with open(Path(__file__).parent/'figure_3b_data.csv') as f:
        reader = csv.reader(f)
        data.update({'figure_3b_data': [[float(x) for x in el] for el in list(reader)]})
    

    return data

def exponential_model_plotting():
    data = load_data()

    # Figure 1a
    figure_1a_data = data['figure_1a_data']
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
    plt.savefig(Path(__file__).parent.parent.parent/'Figures'/'figure_1a.eps') 
    plt.show()


    # Figure 1b
    figure_1b_data = data['figure_1b_data']
    figure_1b_xdata = figure_1b_data[0]
    figure_1b_1 = figure_1b_data[1]
    figure_1b_3 = figure_1b_data[2]
    figure_1b_7 = figure_1b_data[3]

    plt.figure("1b")
    plt.plot(figure_1b_xdata, figure_1b_1,figure_1b_xdata, figure_1b_3,figure_1b_xdata, figure_1b_7)
    plt.xlabel('Test sensitivity', fontsize = 14)
    plt.ylabel('Probability of detection within 7 days', fontsize = 14)
    plt.legend(['Weekly testing', '3/week testing', 'Daily testing'],fontsize =12)
    plt.ylim(0,1.05)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig(Path(__file__).parent.parent.parent/'Figures'/'figure_1b.eps') 
    plt.show()


    # Supp figure
    supp_figure_data = data['supp_figure_data']
    supp_figure_xdata = supp_figure_data[0]
    average_probs = [x[0] for x in supp_figure_data[1:]]
    min_probs = [x[1] for x in supp_figure_data[1:]]
    max_probs = [x[2] for x in supp_figure_data[1:]]

    plt.figure("supp_figure")
    plt.plot(supp_figure_xdata, average_probs)
    plt.fill_between(supp_figure_xdata, min_probs, max_probs,alpha = .25, color = 'b')
    plt.xlabel('Number of times testing occurs per week', fontsize = 14)
    plt.ylabel('Probability of detection', fontsize = 14)
    plt.savefig(Path(__file__).parent.parent.parent/'Figures'/'supp_figure.eps')
    plt.show()

def abm_plotting():
    data = load_data()

    # Figure 2a
    # ABM data
    figure_2a_data = data['figure_2a_data']
    figure_2a_xdata = figure_2a_data[0]
    figure_2a_1 = figure_2a_data[1]
    figure_2a_3 = figure_2a_data[2]
    figure_2a_7 = figure_2a_data[3]

    # Exponential model data
    figure_1b_data = data['figure_1b_data']
    figure_1b_xdata = figure_1b_data[0]
    figure_1b_1 = figure_1b_data[1]
    figure_1b_3 = figure_1b_data[2]
    figure_1b_7 = figure_1b_data[3]

    plt.scatter([],[],c = 'k', label = 'ABM')
    plt.plot([],[],'k--', label = 'Exponential model')
    plt.scatter(figure_2a_xdata,figure_2a_1)
    plt.plot(figure_2a_xdata ,figure_2a_1, label = 'Tested once per week')
    plt.plot(figure_1b_xdata , figure_1b_1, 'C0--')
    plt.scatter(figure_2a_xdata , figure_2a_3)
    plt.plot(figure_2a_xdata , figure_2a_3, label = 'Tested three times per week')
    plt.plot(figure_1b_xdata , figure_1b_3, 'C1--')
    plt.scatter(figure_2a_xdata , figure_2a_7)
    plt.plot(figure_2a_xdata , figure_2a_7, label = 'Tested daily')
    plt.plot(figure_1b_xdata , figure_1b_7, 'C2--')
    plt.xlabel('Test sensitivity')
    plt.ylabel('Probability of detection within 7 days')
    plt.legend()
    plt.ylim(0,1.05)
    plt.savefig(Path(__file__).parent.parent.parent/'Figures'/'figure_2a.eps')
    plt.show()


    # Figure 2b
    figure_2b_data = data['figure_2b_data']
    figure_2b_xdata = figure_2b_data[0]
    figure_2b_1 = figure_2b_data[1]
    figure_2b_3 = figure_2b_data[2]
    figure_2b_7 = figure_2b_data[3]

    plt.scatter(figure_2b_xdata,figure_2b_1, label = 'Tested once per week')
    plt.plot(figure_2b_xdata ,figure_2b_1)
    plt.scatter(figure_2b_xdata , figure_2b_3, label = 'Tested three times per week')
    plt.plot(figure_2b_xdata , figure_2b_3)
    plt.scatter(figure_2b_xdata , figure_2b_7, label = 'Tested daily')
    plt.plot(figure_2b_xdata , figure_2b_7)
    plt.xlabel('Test sensitivity')
    plt.ylabel('Probability of detection within 7 days')
    plt.legend()
    plt.ylim(0,1.05)
    plt.savefig(Path(__file__).parent.parent.parent/'Figures'/'figure_2b.eps')
    plt.show()


    # Figure 3a
    figure_3a_data = data['figure_3a_data']
    figure_3a_xdata = figure_3a_data[0]
    figure_3a_1 = figure_3a_data[1]
    figure_3a_2 = figure_3a_data[2]
    figure_3a_3 = figure_3a_data[3]
    figure_3a_4 = figure_3a_data[4]

    plt.scatter(figure_3a_xdata,figure_3a_1)
    plt.plot(figure_3a_xdata,figure_3a_1, label = 'Schedule 1')
    plt.scatter(figure_3a_xdata, figure_3a_2)
    plt.plot(figure_3a_xdata, figure_3a_2, label = 'Schedule 2')
    plt.scatter(figure_3a_xdata, figure_3a_3)
    plt.plot(figure_3a_xdata, figure_3a_3, label = 'Schedule 3')
    plt.scatter(figure_3a_xdata, figure_3a_4)
    plt.plot(figure_3a_xdata, figure_3a_4, label = 'Schedule 4')
    plt.legend()
    plt.xlabel('Test sensitivity')
    plt.ylabel('Probability of detection within 7 days')
    plt.title('Testing 3 times per week')
    plt.ylim(0.5,1.05)
    plt.savefig(Path(__file__).parent.parent.parent/'Figures'/'figure_3a.eps')
    plt.show()


    # Figure 3b
    figure_3b_data = data['figure_3b_data']
    figure_3b_xdata = figure_3b_data[0]
    figure_3b_1 = figure_3b_data[1]
    figure_3b_2 = figure_3b_data[2]
    figure_3b_3 = figure_3b_data[3]
    figure_3b_4 = figure_3b_data[4]

    plt.scatter(figure_3b_xdata,figure_3b_1)
    plt.plot(figure_3b_xdata,figure_3b_1, label = 'Schedule 1')
    plt.scatter(figure_3b_xdata, figure_3b_2)
    plt.plot(figure_3b_xdata, figure_3b_2, label = 'Schedule 2')
    plt.scatter(figure_3b_xdata, figure_3b_3)
    plt.plot(figure_3b_xdata, figure_3b_3, label = 'Schedule 3')
    plt.scatter(figure_3b_xdata, figure_3b_4)
    plt.plot(figure_3b_xdata, figure_3b_4, label = 'Schedule 4')
    plt.legend()
    plt.xlabel('Test sensitivity')
    plt.ylabel('Probability of detection within 7 days')
    plt.title('Testing daily')
    plt.ylim(0.5,1.05)
    plt.savefig(Path(__file__).parent.parent.parent/'Figures'/'figure_3b.eps')
    plt.show()
    

exponential_model_plotting()
abm_plotting()