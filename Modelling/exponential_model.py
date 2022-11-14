from multiprocessing.dummy.connection import families
from pickle import FALSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from itertools import count
from itertools import permutations
import csv 

# model definition
def simple_exponential_growth(initial_population=1, r_eff=1.5, num_weeks=4, generation_interval=4.7):
    daily_multiplier = r_eff ** (1 / generation_interval)
    prev_list = [initial_population * (daily_multiplier ** day) for day in range(7*num_weeks)]
    return prev_list

# detection probability definition
def detection_probability(test_schedule, prevalence, sensitivity, workplace_size):
    probability_of_detection = 1 - np.prod([(1 - sensitivity)**(prevalence[i]*workplace_size) if test_schedule[i]!=0 else 1 for i in range(len(prevalence))])
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

# creating testing schedule
def define_testing_schedule(per_week_schedule, prop_tested_per_week, workplace_size):
    testing_schedule = []

    total_tests_per_week = np.round(workplace_size * prop_tested_per_week)
    average_tests_per_testing_day = int(np.ceil(total_tests_per_week/sum(per_week_schedule[0])))
    for i in range(len(per_week_schedule)):
        testing_schedule.append([average_tests_per_testing_day * day for day in per_week_schedule[i]])

    if sum(testing_schedule[0]) > total_tests_per_week:
        for i in range(len(testing_schedule)):
            remainder = sum(testing_schedule[i]) - total_tests_per_week
            for j in range(7):
                if testing_schedule[i][j] != 0:
                    testing_schedule[i][j] -= 1
                    remainder -= 1
                if remainder == 0:
                    break

    total_testing_schedule = [schedule for schedule in testing_schedule]

    return total_testing_schedule

def generate_permutations(test_days):
    test_schedule = []
    for i in range(7):
        test_schedule.append([test_days[(j + i)% 7] for j in range(7)])
    return test_schedule


def varying_growth_rate_high(endpoints, step, num_weeks, I0, generation_interval, testing_days_per_week, workplace_size, prop_tested_per_week, sensitivity):
    growth_rate_options = [i/100 for i in list(range(int(endpoints[0]*100), int(endpoints[1]*100), int(step*100)))]
    output = []
    schedules = weekly_schedule_permutations(testing_days_per_week)
    testing_schedules = define_testing_schedule(schedules, prop_tested_per_week, workplace_size, num_weeks)
    # testing_schedules = phase_three_a_scheduling(workplace_size, testing_days_per_week, prop_tested_per_week, num_weeks)
    for i in growth_rate_options:
        individual_output = []
        infected_list = simple_exponential_growth(I0, i, num_weeks, generation_interval)
        prevalence_list = [j/workplace_size for j in infected_list]
        for permutation in testing_schedules:
            individual_output.append(detection_probability(permutation, prevalence_list, sensitivity, workplace_size))
        output.append(sum(individual_output)/len(individual_output))
    return output

def varying_sensitivity(endpoints, step, num_weeks, Reff, I0, generation_interval, testing_days_per_week, workplace_size, prop_tested_per_week):
    sensitivity_options = [i/100 for i in list(range(int(endpoints[0]*100), int(endpoints[1]*100), int(step*100)))]
    output = []
    schedules = weekly_schedule_permutations(testing_days_per_week)
    #high coverage means if testing occurs, entire workplace is tested
    testing_schedules = [num_weeks*sched for sched in schedules]
    infected_list = simple_exponential_growth(I0, Reff, num_weeks, generation_interval)
    prevalence_list = [j/workplace_size for j in infected_list]
    for sen in sensitivity_options:
        individual_output = []
        for permutation in testing_schedules:
            individual_output.append(detection_probability(permutation, prevalence_list, sen, workplace_size))
        output.append(sum(individual_output)/len(individual_output))
    return output

def varying_testing_frequency_high_coverage(num_weeks,I0,Reff,generation_interval,testing_days_per_week,sensitivity,workplace_size):
    output = [] #average, min, max
    infected_list = simple_exponential_growth(I0, Reff, num_weeks, generation_interval)
    prevalence_list = [j/workplace_size for j in infected_list]
    for days in testing_days_per_week:
        individual_output = []
        schedules = weekly_schedule_permutations(days)
        #high coverage means if testing occurs, entire workplace is tested
        testing_schedules = [num_weeks*sched for sched in schedules]
        for permutation in testing_schedules:
            individual_output.append(detection_probability(permutation, prevalence_list, sensitivity,workplace_size))
        # output average, min and max
        average_prob = sum(individual_output)/len(individual_output)
        min_prob = min(individual_output)
        max_prob = max(individual_output)
        output.append([average_prob, min_prob, max_prob])
    return output

high_coverage = True
varying_workplace_size_plot = False
prop_tested_sensitivity_plot = False
growth_rate_sensitivity_plot = False
test_days_per_week_sensitivity = True
# reff_sensitivity = True
testing_frequency_variance_plot = False


if high_coverage:
    prop_tested_per_week = 1
    #workplace size doesn't matter we'll just choose 50
    workplace_size = 50
    Reff = 1.1
    test_sensitivity = 0.85
    testing_days_per_week = 1
    generation_interval = 4.7
    I0 = 1

    if test_days_per_week_sensitivity:
        #7 and 14 days
        num_weeks = [1,2]
        xdata = list(range(40,101,1))
        xdata = [x/100 for x in xdata]
        timeframe_data = []
        testing_days= [1, 3, 7]
        for weeks in num_weeks:
            data = []
            for days in testing_days:
                data.append(varying_sensitivity([0.4, 1.01], 0.01, weeks, Reff, I0, generation_interval, days, workplace_size, prop_tested_per_week))
            timeframe_data.append(data)

        with open("data.csv", 'w') as f:
            write = csv.writer(f)
            write.writerows(timeframe_data[0])
        
        plt.figure("1")
        plt.plot(xdata, timeframe_data[0][0],xdata, timeframe_data[0][1],xdata, timeframe_data[0][2])
        plt.xlabel('Test sensitivity', fontsize = 14)
        plt.ylabel('Probability of detection within 7 days', fontsize = 14)
        plt.title('Exponential Model',fontsize = 14)
        plt.legend(['Weekly testing', '3/week testing', 'Daily testing'],fontsize =12)
        plt.ylim(0,1.05)
        plt.savefig('test_sens_high_7') 
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.show()

        # plt.figure("14 days")
        # plt.plot(xdata, timeframe_data[1][0],xdata, timeframe_data[1][1],xdata, timeframe_data[1][2])
        # plt.xlabel('Test sensitivity')
        # plt.ylabel('Probability a positive case is detected in 14 days')
        # plt.title('High coverage')
        # plt.legend(['Weekly testing', '3/week testing', 'Daily testing'])  
        # plt.ylim(0,1.05)
        # plt.savefig('test_sens_high_14')     
        # plt.show()     

    if growth_rate_sensitivity_plot:
        #7 and 14 days
        num_weeks = [1,2]
        xdata = list(range(100,240,1))
        xdata = [x/100 for x in xdata]
        timeframe_data = []
        Reff_options = [1.1, 1.5, 2, 2.5]
        sensitivity_options = [0.65, 0.75, 0.85, 0.95]

        for weeks in num_weeks:
            data = []
            for option in sensitivity_options:
                data.append(varying_growth_rate_high([1.0, 2.4], 0.01, weeks, I0, generation_interval, testing_days_per_week, workplace_size, prop_tested_per_week, option))
            timeframe_data.append(data)
        
        plt.figure("2")
        plt.plot(xdata, timeframe_data[0][0],xdata, timeframe_data[0][1],xdata, timeframe_data[0][2],xdata, timeframe_data[0][3])
        plt.xlabel('$R_{eff}$',fontsize = 14)
        plt.ylabel('Probability of detection within 7 days', fontsize= 14)
        plt.title('Exponential Model', fontsize = 14)
        plt.legend(['Test sensitivity = 0.65', 'Test sensitivity = 0.75', 'Test sensitivity = 0.85', 'Test sensitivity = 0.95'],fontsize = 12)
        plt.ylim(0, 1.05)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.savefig('reff_sens_high_7') 
        plt.show()

        # plt.figure("14 days")
        # plt.plot(xdata, timeframe_data[1][0],xdata, timeframe_data[1][1],xdata, timeframe_data[1][2],xdata, timeframe_data[1][3])
        # plt.xlabel('$R_{eff}$')
        # plt.ylabel('Probability a positive case is detected in 14 days')
        # plt.title('High Coverage')
        # plt.legend(['Test sensitivity = 0.65', 'Test sensitivity = 0.75', 'Test sensitivity = 0.85', 'Test sensitivity = 0.95'])
        # plt.ylim(0,1.05)
        # plt.savefig('reff_sens_high_14')     
        # plt.show()  
    
    if testing_frequency_variance_plot:
        num_weeks = 1
        testing_days = range(1,8)
        data = varying_testing_frequency_high_coverage(num_weeks,I0,Reff,generation_interval,testing_days,test_sensitivity,workplace_size)
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