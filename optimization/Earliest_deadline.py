import numpy as np
import csv
import cvxpy as cp  # convex optimization
import matplotlib.pyplot as plt
import argparse
import time
import data_handler_yearly
from carbon_forecasts import carbon_intensity_forecast

time_step = 5  # min
num_steps = int(24 * (60 / time_step) - 1)  # steps inn 24 hours
total_vehicles = 10
battery_capacity = 50  # kwh
power_capacity = 120.0 / 8  # kw, max power delivery every time step ~24 cars at same time
max_power_u = 5.0 / 8  # max power intake for cars every time step, ~6hours full charge
balancing_fac = 1
total_num_of_data_entries = 6742

arrival_time = np.zeros(total_num_of_data_entries)
departure_time = np.zeros(total_num_of_data_entries)
required_energy = np.zeros(total_num_of_data_entries)
initial_state = np.random.uniform(2.0, 20.0, size=(total_num_of_data_entries,))  # 4% - 40% SoC initial
final_energy = np.zeros(total_num_of_data_entries)
carbon_intensity = np.zeros(num_steps)

# Get Berkley Data
arrival_time, departure_time, required_energy, date = data_handler_yearly.getCleanedBerkleyData(arrival_time,
                                                                                                departure_time,
                                                                                                required_energy)
final_energy = np.minimum(np.array(initial_state + required_energy),
                          battery_capacity * np.ones(total_num_of_data_entries, ))
#print('final',final_energy, 'initial', initial_state, 'required', required_energy)
# print("final_shape", np.shape(final_energy))
# print('maxfinal',max(final_energy))
# print('date',date)
# arrival_time = [int(i*12) for i in arrival_time]
# departure_time = [int(i*12) for i in departure_time]
# print(np.shape(arrival_time))
# print(departure_time)

# Get Carbon Intensity Data
carbon_intensity = np.array(data_handler_yearly.getCarbonIntensityData1year(), dtype=float)
carbon_intensity = carbon_intensity[:, 1:]
print("carbon intensity data shape", np.shape(carbon_intensity))


def Earliest_deadline(num_of_vehicles, timesteps,
                     initial_states, max_power, terminal_states,
                     arrival_time, dept_time, power_capacity,
                     B, factor, day):

    u_mat = np.zeros((num_of_vehicles, num_steps), dtype=float)
    initial_state_EDF = np.copy(initial_states)
    # -5 to avoid computation infeasibility at this time
    for t in range(np.int(arrival_time[0]) + 1, num_steps - 5):
        power_budget = power_capacity  # Change this for variable case
        #Firstly get the states
        #print("current number of arrived cars", (arrival_time < t).sum())
        vehicle_ending_index = (arrival_time < t).sum()
        step_initial_SOC = np.copy(initial_state_EDF[:vehicle_ending_index])
        depart_schedule=np.copy(dept_time[:vehicle_ending_index])
        u_val=np.zeros_like(step_initial_SOC)
        index=np.argsort(depart_schedule) #sort the departure time
        charging_sessions=0

        while power_budget>=0:
            if depart_schedule[index[charging_sessions]] >= t:
                available_charging=np.minimum(max_power, power_budget)
                u_val[index[charging_sessions]] = np.maximum(np.minimum(available_charging, final_energy[index[charging_sessions]]-step_initial_SOC[index[charging_sessions]]),0)

            power_budget-=u_val[index[charging_sessions]]
            charging_sessions+=1

            if charging_sessions>=vehicle_ending_index:
                break

        initial_state_EDF[:vehicle_ending_index] += u_val
        u_mat[:vehicle_ending_index, t]=u_val
        #print(initial_state_EDF)

    return initial_state_EDF, u_mat



starting_index = 0
end_index = 0

feature_mat, carbon_model = carbon_intensity_forecast()

for current_date in range(1, 100):

    starting_index = np.copy(end_index)
    num_of_vehicles = 0

    while date[end_index] == date[starting_index]:
        num_of_vehicles += 1
        end_index += 1
    ini_state = initial_state[starting_index:end_index]
    final_state = final_energy[starting_index:end_index]
    arr_time = arrival_time[starting_index:end_index]
    dept_time = departure_time[starting_index:end_index]

    print("DAY ", current_date)
    print("number of cars", num_of_vehicles)

    #First implement offline algorithm
    x, u = Earliest_deadline(num_of_vehicles, num_steps,
                            ini_state, max_power_u, final_state,
                            arr_time, dept_time, power_capacity,
                            battery_capacity, factor=balancing_fac, day=current_date)
    carbon_emission = np.sum(np.array([u[:, t] * carbon_intensity[current_date, t] for t in range(num_steps)]))
    print(f'the energy delivery: {round(np.sum(u), 2)}, '
          f'the required energy: {round(np.sum(required_energy[starting_index:end_index]), 2)}, '
          f'the carbon emission term: {round(carbon_emission, 2)}')


