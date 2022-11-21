from pickle import FALSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from itertools import count
from itertools import permutations
import csv 

# model definition
def simple_exponential_growth(initial_population=1, r_eff=1.5, generation_interval=4.7):
    daily_multiplier = r_eff ** (1 / generation_interval)
    prev_list = [initial_population * (daily_multiplier ** day) for day in range(7)]
    return prev_list

# detection probability definition
def detection_probability(test_schedule, prevalence, sensitivity):
    probability_of_detection = 1 - np.prod([(1 - sensitivity)**(prevalence[i]) if test_schedule[i]!=0 else 1 for i in range(len(prevalence))])
    return probability_of_detection

# creating weekly schedule permutations
def weekly_schedule_permutations(testing_days_per_week):
    if testing_days_per_week == 7:
        return [[1,1,1,1,1,1,1]]

    schedule = set()
    for x in permutations([1]*testing_days_per_week + [0]*(7-testing_days_per_week), 7):
        if x not in schedule:
            schedule.add(x)
    
    per_week_schedule = list(list(x) for x in schedule)

    return per_week_schedule


# data for Figure 1a
def varying_growth_rate(endpoints, step, I0, generation_interval, testing_days_per_week, workplace_size, prop_tested_per_week, sensitivity):
    growth_rate_options = [i/100 for i in list(range(int(endpoints[0]*100), int(endpoints[1]*100), int(step*100)))]
    output = []

    # defining schedules and prevalence
    testing_schedules = weekly_schedule_permutations(testing_days_per_week)
    
    for i in growth_rate_options:
        individual_output = []
        prevalence_list = simple_exponential_growth(I0, i, generation_interval)

        # calculate detection probability for each testing schedule permutation
        for permutation in testing_schedules:
            individual_output.append(detection_probability(permutation, prevalence_list, sensitivity))
        
        # take the average of detection probabilities
        output.append(sum(individual_output)/len(individual_output))
    return output

# data for Figure 1b
def varying_sensitivity(endpoints, step, Reff, I0, generation_interval, testing_days_per_week):
    sensitivity_options = [i/100 for i in list(range(int(endpoints[0]*100), int(endpoints[1]*100), int(step*100)))]
    output = []

    # generating schedules and prevalence
    testing_schedules = weekly_schedule_permutations(testing_days_per_week)
    prevalence_list = simple_exponential_growth(I0, Reff, generation_interval)

    for sen in sensitivity_options:
        individual_output = []

        # calculate detection probability for each testing schedule permutation 
        for permutation in testing_schedules:
            individual_output.append(detection_probability(permutation, prevalence_list, sen))
        output.append(sum(individual_output)/len(individual_output))
    return output

# data for supp figure impact of testing day permutations
def varying_testing_frequency(num_weeks,I0,Reff,generation_interval,testing_days_per_week,sensitivity,workplace_size):
    output = [] #average, min, max
    infected_list = simple_exponential_growth(I0, Reff, generation_interval)
    prevalence_list = [j/workplace_size for j in infected_list]
    for days in testing_days_per_week:
        individual_output = []
        testing_schedules = weekly_schedule_permutations(days)
        #high coverage means if testing occurs, entire workplace is tested
        # testing_schedules = [num_weeks*sched for sched in schedules]
        for permutation in testing_schedules:
            individual_output.append(detection_probability(permutation, prevalence_list, sensitivity,workplace_size))
        # output average, min and max
        average_prob = sum(individual_output)/len(individual_output)
        min_prob = min(individual_output)
        max_prob = max(individual_output)
        output.append([average_prob, min_prob, max_prob])
    return output

# main 
# Figure 1a data




growth_rate_sensitivity_plot = True
test_days_per_week_sensitivity = False
# reff_sensitivity = True
testing_frequency_variance_plot = False

# default parameter values
#workplace size doesn't matter we'll just choose 50 as everyone is tested 
workplace_size = 50
Reff = 1.1
test_sensitivity = 0.85
testing_days_per_week = 1
generation_interval = 4.7
I0 = 1
prop_tested_per_week = 1

# Figure 1a data 
xdata_temp = list(range(100,240,1))
figure_1a_xdata = [x/100 for x in xdata_temp]
figure_1a_data = []
sensitivity_options = [0.65, 0.75, 0.85, 0.95]

for option in sensitivity_options:
    figure_1a_data.append(varying_growth_rate([1.0, 2.4], 0.01, I0, generation_interval, testing_days_per_week, workplace_size, prop_tested_per_week, option))

with open("figure_1a_data.csv", 'w') as f:
    write = csv.writer(f)
    write.writerow(figure_1a_xdata)
    write.writerows(figure_1a_data)

# checking plots are fine 
plt.figure("2")
plt.plot(figure_1a_xdata, figure_1a_data[0],figure_1a_xdata, figure_1a_data[1],figure_1a_xdata, figure_1a_data[2],figure_1a_xdata, figure_1a_data[3])
plt.xlabel('$R_{eff}$',fontsize = 14)
plt.ylabel('Probability of detection within 7 days', fontsize= 14)
plt.title('Exponential Model', fontsize = 14)
plt.legend(['Test sensitivity = 0.65', 'Test sensitivity = 0.75', 'Test sensitivity = 0.85', 'Test sensitivity = 0.95'],fontsize = 12)
plt.ylim(0, 1.05)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('reff_sens_high_7') 
plt.show()


# Figure 1b data 
xdata_temp = list(range(40,101,1))
figure_1b_xdata = [x/100 for x in xdata_temp]
figure_1b_data = []
testing_days= [1, 3, 7]

for days in testing_days:
    figure_1b_data.append(varying_sensitivity([0.4, 1.01], 0.01, Reff, I0, generation_interval, days))

with open("figure_1b_data.csv", 'w') as f:
    write = csv.writer(f)
    write.writerow(figure_1b_xdata)
    write.writerows(figure_1b_data)

plt.figure("1")
plt.plot(figure_1b_xdata, figure_1b_data[0],figure_1b_xdata, figure_1b_data[1],figure_1b_xdata, figure_1b_data[2])
plt.xlabel('Test sensitivity', fontsize = 14)
plt.ylabel('Probability of detection within 7 days', fontsize = 14)
plt.title('Exponential Model',fontsize = 14)
plt.legend(['Weekly testing', '3/week testing', 'Daily testing'],fontsize =12)
plt.ylim(0,1.05)
plt.savefig('test_sens_high_7') 
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

# Supplementary figure data





if testing_frequency_variance_plot:
    num_weeks = 1
    testing_days = range(1,8)
    data = varying_testing_frequency(num_weeks,I0,Reff,generation_interval,testing_days,test_sensitivity,workplace_size)
    average_probs = [x[0] for x in data]
    min_probs = [x[1] for x in data]
    max_probs = [x[2] for x in data]

    plt.plot(testing_days, average_probs)
    plt.fill_between(testing_days, min_probs, max_probs,alpha = .25, color = 'b')
    plt.xlabel('Number of times testing occurs per week')
    plt.ylabel('Expected? probability of detection')
    plt.title('Probability of detection, varying test frequency')
    plt.savefig('med_varying_frequency')
    plt.show()