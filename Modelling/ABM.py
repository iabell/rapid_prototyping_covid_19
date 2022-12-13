from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math
from scipy.stats import binom 
from itertools import count 
import csv
from tqdm import tqdm
from pathlib import Path
from scipy import stats 

# Agent definition 
class Agent:
    def __init__(self, params):
        self.status = "susceptible"
        self.infectiousness = "non-infectious"
        self.infection_clock = 0
        self.present = []
        self.incubation_period = np.random.lognormal(params['inc_mu'], params['inc_sig'])
        # how long someone spends either symptomatic or asymptomatic 
        self.recovery_period = params['symp_min'] + np.random.uniform()*(params['symp_max'] - params['symp_min'])
        self.end_of_incubation = params['latent_period'] + self.incubation_period
        self.end_of_infection = self.end_of_incubation + self.recovery_period

        # random probabilities 
        self.symptomatic_prob = rand.random()


    def update_clock(self, total_daily_prevalence, dt):
        if self.status == 'latent' or self.infectiousness == 'infectious':
            self.infection_clock += dt
            total_daily_prevalence += 1
        
        return total_daily_prevalence

    def update_clock_beta_calibration(self, dt):
        if self.status == 'latent' or self.infectiousness == 'infectious':
            self.infection_clock += dt



    def check_transition(self, params):
        if (self.status == 'symptomatic' or self.status == 'asymptomatic') and self.infection_clock >= self.end_of_infection:
            self.status = 'recovered'
            self.infectiousness = 'non-infectious'
        elif self.status == 'incubating' and (self.infection_clock >= self.end_of_incubation and self.infection_clock < self.end_of_infection):
            if self.symptomatic_prob <= params['asymp_fraction']:
                self.status = 'asymptomatic'
            else:
                self.status = 'symptomatic'
        elif params['latent_period'] != 0 and self.status == 'latent' and (self.infection_clock >= params['latent_period'] and self.infection_clock < self.end_of_incubation):
            self.status = 'incubating'
            self.infectiousness = 'infectious'


    def infection_event(self, params, FOI):
        if self.status == "susceptible":
            infection_prop = np.random.rand()
            infection_prob = 1 - np.exp(-FOI)
            if infection_prop < infection_prob:
                if params['latent_period'] != 0:
                    self.status = 'latent'
                else: 
                    self.status = 'incubating'
                    self.infectiousness = 'infectious'

    def testing(self, params, test_sensitivity, detection_testing, detection_symptoms, people_tested, n_test_t, t):
        # detection via testing 
        if n_test_t - people_tested > 0 and detection_testing == 0 and n_test_t > 0:
            people_tested += 1
            if self.infectiousness == 'infectious':
                test_rand = np.random.rand()
                if test_rand < test_sensitivity:
                    #positive case indentified 
                    detection_testing = t + params['test_report_delay']
        
        if self.status == 'symptomatic':
            detection_symptoms = t + params['symptom_presentation_delay']

        return detection_testing, detection_symptoms, people_tested

def create_agents(params):
    d ={} #dictionary of agents 
    for x in range(params['N']):
        d["agent_{0}".format(x)] = Agent(params)
    return d

# Inner simulation functions 
def define_work_schedule(d, params, roster):
    available_indices = list(range(params['N']))
    temp_indices = []
    work_schedule = [[],[],[],[],[],[],[]]
    for el in range(0,len(roster)):
        people_working = rand.sample(available_indices,roster[el])
        for i in people_working:
            temp_indices.append(i)
            agent = "agent_{0}".format(i)
            d[agent].present = rand.sample(range(0,7),2*el + 1)
            for day in d[agent].present:
                work_schedule[day].append(i)
        available_indices = list(set(available_indices) - set(temp_indices))

    return work_schedule

def seed_infection(d, params, work_schedule):
    #infect a random agent
    #select someone working on first day randomly - maybe no one is working on Monday so need to find first day people are working 
    for day in range(0,7):
        if work_schedule[day] != []:
            seed_index = rand.sample(work_schedule[day],1)[0]
            start = day
            break

    # infection_status[index_case] = 1
    seed_agent = "agent_{0}".format(seed_index)
    d[seed_agent].status = 'latent'
    if params['latent_period'] == 0:
        d[seed_agent].status = 'incubating'
        d[seed_agent].infectiousness = 'infectious'
    
    return start

def status_transitions(d, params, work_schedule, day_of_week, dt):
    infectious_at_work_count= 0

    for n in range(params['N']):
        agent_n = "agent_{0}".format(n)

        # check for transitions between compartments 
        d[agent_n].check_transition(params)

        if d[agent_n].infectiousness == 'infectious' and (day_of_week in d[agent_n].present):
            infectious_at_work_count += 1

    people_at_work = len(work_schedule[day_of_week])

    # Force of infection - frequency dependent transmission 
    FOI = (params['beta']/(people_at_work - 1))*infectious_at_work_count*dt

    return FOI

def infection_and_testing(d, params, FOI, work_schedule, day_of_week, test_sensitivity, n_test_t, t):
        # keeping track of testing
        detection_testing = 0
        detection_symptoms = 0
        people_tested = 0

        #iterate infection clocks
        for person in work_schedule[day_of_week]:
            agent_n = "agent_{0}".format(person)

            detection_testing, detection_symptoms, people_tested = d[agent_n].testing(params, test_sensitivity, detection_testing, detection_symptoms, people_tested, n_test_t, t)

            # check for infection events 
            d[agent_n].infection_event(params, FOI)

            if detection_testing > 0 or detection_symptoms > 0:
                if detection_testing > 0 and detection_symptoms > 0:
                    return min(detection_symptoms, detection_testing)
                else:
                    return detection_testing if detection_testing > 0 else detection_symptoms
        
        # outbreak not detected
        return -1

def update_clock_calculate_prevalence(d, params, dt):
    total_daily_prevalence = 0
    for n in range(params['N']):
        agent_n = "agent_{0}".format(n)
        # update clocks
        total_daily_prevalence = d[agent_n].update_clock(total_daily_prevalence, dt)
    return total_daily_prevalence

#single simulation 
def ABM_model_def(R0, roster,test_sensitivity,test_schedule, asymp_fraction):

    params = dict([
        ('symp_min', 5),
        ('symp_max', 10),
        ('inc_mu', 1.62),
        ('inc_sig', 0.418),
        ('asymp_fraction', asymp_fraction),
        ('test_sensitivity', test_sensitivity),
        ('latent_period', 1),
        ('test_report_delay', 0),
        ('N', int(sum(roster))),
        # beta calibrated for workplaces size 120 using beta calibration
        # R0 = 14.32*beta + 0.004
        ('beta', R0/14.32),
        ('symptom_presentation_delay', 0) 
    ])
    
    # if we're assuming exponential model assumptions, latent_period = 0
    if params['asymp_fraction'] == 1:
        params['latent_period'] = 0
        params['test_report_delay'] = 0

    # timing for simulation
    max_time = 200 #days?

    dt = 0.25

    #maximum number of tests per day (test schedule is % of workforce tested that day) 
    tests_per_day = [np.floor(tests*params['N']) for tests in test_schedule]
    
    d = create_agents(params)

    work_schedule = define_work_schedule(d, params, roster)

    start = seed_infection(d, params, work_schedule)

    for t in np.arange(start, start + max_time, dt):
        day_of_week = int(np.floor(t%7))

        # number of tests for this day 
        # only testing once per day 
        if t - np.floor(t) == 0:
            n_test_t = min(tests_per_day[day_of_week],sum(work_schedule[day_of_week]))
        else:
            n_test_t = 0

        FOI = status_transitions(d, params, work_schedule, day_of_week, dt)

        time_to_detection = infection_and_testing(d, params, FOI, work_schedule, day_of_week, test_sensitivity, n_test_t, t)

        if time_to_detection > 0:
            return time_to_detection

        total_daily_prevalence = update_clock_calculate_prevalence(d, params, dt)
            
        #if we run out of infections before detection (t>10 to ensure we don't break in first step as first latent infection won't register as infectious)
        if total_daily_prevalence == 0 and t> 10:
            return -1

# full simulation 
def ABM_simulation(R0, roster, test_sensitivity, test_schedule,simulations,asymp_fraction):
    results = []
    for i in tqdm(range(simulations)):
        schedule_sim = np.random.permutation(test_schedule)
        results.append(ABM_model_def(R0, roster,test_sensitivity,schedule_sim,asymp_fraction))

    #disregard results if outbreak dies out 
    # results = [i for i in results if i >= 0]
    # time_to_detection = np.average(results)
    time_to_detection = 0
    # time_to_detection = [np.average(results), np.percentile(results,95)]
    prob_of_detection_7_days = sum([1 for x in results if (x <= 7 and x > 0)])/len(results)
    return time_to_detection, prob_of_detection_7_days

def calculate_beta(R0, N):
    beta_range = np.linspace(0.05, 0.2, 100)
    beta_vals = []
    R0_vals = []
    simulations = 500

    dt = 0.25

    for beta in tqdm(beta_range):
        total_simulation_infections = 0
        for sim in range(simulations):
            params = dict([
            ('symp_min', 5),
            ('symp_max', 10),
            ('inc_mu', 1.62),
            ('inc_sig', 0.418),
            ('latent_period', 1),
            ('asymp_fraction', 0), # not important for beta calibration 
            ('test_report_delay', 0),
            ('N', N),
            ('beta', beta),
        ])
            
            # timing for simulation
            max_time = 50 #days?
            
            d = create_agents(params)

            start = 1

            # seed infection
            seed_index = rand.randint(0, N-1)
            seed_agent = "agent_{0}".format(seed_index)
            d[seed_agent].status = 'incubating'
            d[seed_agent].infectiousness = 'infectious'
        
            # force of infection for 1 infected person
            FOI = (beta/(N - 1))*1*dt

            secondary_infection_count = 0

            for t in range(start,max_time+start, dt):

                d[seed_agent].check_transition(params) #checking to see if seed is non-infectious
                if d[seed_agent].infectiousness != 'infectious':
                    break 


                #seeing if someone gets infected
                for person in range(N):
                    agent_n = "agent_{0}".format(person)

                    # check for infection events 
                    d[agent_n].infection_event(params, FOI)
                    if d[agent_n].status != 'susceptible' and d[agent_n].status != 'removed' and person != seed_index:
                        secondary_infection_count += 1
                        d[agent_n].status = 'removed'

                d[seed_agent].update_clock_beta_calibration(dt)

            total_simulation_infections = total_simulation_infections + secondary_infection_count
        
        R0_sim = total_simulation_infections/simulations
        R0_vals.append(R0_sim)

    X = beta_range
    Y = R0_vals

    results = stats.linregress(X, Y)
    slope = results.slope
    intercept = results.intercept
    rval = results.rvalue

    plt.figure('beta_calibration')
    plt.plot(beta_range, R0_vals, label = 'Simulated')
    plt.plot(beta_range, intercept + slope*np.array(beta_range), '--', label='Fitted line')
    plt.xlabel('beta')
    plt.ylabel('$R_0$')
    plt.legend()
    plt.title('slope = ' + "{:.2f}".format(slope) + ', intercept = ' + "{:.2f}".format(intercept) + ', r = ' + "{:.2f}".format(rval))
    
    plt.savefig(Path(__file__).parent/'Data_and_plotting'/'beta_calibration.eps')
    plt.show()

    
    return slope, intercept


def compare_exp_ABM(R0, roster,test_sensitivity_varying,test_schedule,simulations,asymp_fraction, file_name):
    pr_list= [[],[],[]]


    thing = range(len(test_schedule))

    for i in range(len(test_schedule)):
        for sensitivity in test_sensitivity_varying:
            results = ABM_simulation(R0, roster,sensitivity,test_schedule[i],simulations,asymp_fraction)
            pr_list[i].append(results[1])


    with open(file_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(test_sensitivity_varying)
        write.writerows(pr_list)


    # exponential results on same plot 
    xdata = list(range(40,101,1))
    xdata = [x/100 for x in xdata]
    data_exp = []
    with open(Path(__file__).parent/"Data_and_Plotting"/"figure_1b_data.csv", newline = '') as f:
        data= csv.reader(f)
        for row in data:
            data_exp.append([float(x) for x in row])

    # plt.scatter([-1],[0],c = 'k', label = 'ABM')
    # plt.plot([],[],'k--', label = 'Exponential model')
    plt.scatter(test_sensitivity_varying,pr_list[0])
    plt.plot(test_sensitivity_varying,pr_list[0], label = 'Tested once per week')
    # plt.plot(xdata, data_exp[1], 'C0--')
    plt.scatter(test_sensitivity_varying, pr_list[1])
    plt.plot(test_sensitivity_varying, pr_list[1], label = 'Tested three times per week')
    # plt.plot(xdata, data_exp[2], 'C1--')
    plt.scatter(test_sensitivity_varying, pr_list[2])
    plt.plot(test_sensitivity_varying, pr_list[2], label = 'Tested daily')
    # plt.plot(xdata, data_exp[3], 'C2--')
    plt.xlabel('Test sensitivity')
    plt.ylabel('Probability of detection within 7 days')
    plt.legend()
    plt.ylim(0,1.05)
    plt.xlim(0.38,1.02)
    plt.savefig('figure_2b.eps')
    plt.show()

def intermittent_schedule(R0, rosters,test_sensitivity_varying,test_schedule,simulations,file_name):
    pr_list = [[],[],[],[]]
    asymp_fraction = 1/3

    for i in range(len(rosters)):
        for sensitivity in test_sensitivity_varying:
            results = ABM_simulation(R0, rosters[i],sensitivity,test_schedule[0],simulations,asymp_fraction)
            pr_list[i].append(results[1])

    with open(file_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(test_sensitivity_varying)
        write.writerows(pr_list)


def main():
    once_per_week = [1, 0, 0, 0, 0, 0, 0]
    three_per_week = [1, 0, 1, 0, 1, 0, 0 ]
    daily_testing = [1, 1, 1, 1, 1, 1, 1]

    workplace_size = 120
    no_intermittency = [0,0,0,workplace_size]
    R_eff = 1.1
    simulations = 5000
    sensitivity_options = np.linspace(0.4,1,12)

    # calibrating beta 
    # slope, intercept = calculate_beta(R_eff, workplace_size)
    
    # comparing exponential model and abm (Figure 2)
    # exponential assumptions
    compare_exp_ABM(R_eff, no_intermittency, sensitivity_options,[once_per_week,three_per_week,daily_testing], simulations,1, 'figure_2a.csv')
    # ABM assumptions
    compare_exp_ABM(R_eff, no_intermittency, sensitivity_options,[once_per_week,three_per_week,daily_testing], simulations,1/3, 'figure_2b.csv')


    # intermittent testing scheduling 
    roster_1 = [int(x*workplace_size) for x in [0,0,0,1]]
    roster_2 = [int(x*workplace_size) for x in[0,0,1,0]]
    roster_3 = [int(x*workplace_size) for x in[0,0.4, 0.6, 0]]
    roster_4 = [int(x*workplace_size) for x in [0.1,0.3,0.6,0]]
    Reff = 1.1

    # testing 3 times/week
    file_name = 'figure_3a.csv'
    intermittent_schedule(Reff, [roster_1,roster_2,roster_3,roster_4],sensitivity_options,[three_per_week],simulations,file_name)

    # testing daily 
    file_name = 'figure_3b.csv'
    intermittent_schedule(Reff, [roster_1,roster_2,roster_3,roster_4],sensitivity_options,[daily_testing],simulations,file_name)

main()