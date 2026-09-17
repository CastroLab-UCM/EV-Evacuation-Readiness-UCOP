import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # type: ignore
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

Fontsize =15

##### Worst evacuation time comparison ###########################################################################


x_axis_worst = [20,30,40,50,60,70,80]

df_worst = pd.read_excel(DATA_DIR / 'Mariposa_plots.xlsx', sheet_name='Worst_time_v1', header=None) 

rows_worst = df_worst.values.tolist()


y_axis1_worst = rows_worst[0]
y_axis2_worst = rows_worst[1]
y_axis3_worst = rows_worst[2]
y_axis4_worst = rows_worst[3]
y_axis5_worst = rows_worst[4]
y_axis6_worst = rows_worst[5]
y_axis7_worst = rows_worst[6]

mean_values_worst = [(a + b + c + d + e + f + g) / 7 for a, b, c, d, e, f, g in zip(y_axis1_worst, y_axis2_worst, y_axis3_worst, y_axis4_worst, y_axis5_worst, y_axis6_worst, y_axis7_worst)]
upper_bound_worst = [max(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in zip(y_axis1_worst, y_axis2_worst, y_axis3_worst, y_axis4_worst, y_axis5_worst, y_axis6_worst, y_axis7_worst)]
lower_bound_worst = [min(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in zip(y_axis1_worst, y_axis2_worst, y_axis3_worst, y_axis4_worst, y_axis5_worst, y_axis6_worst, y_axis7_worst)]
print(mean_values_worst)
plt.subplots()
plt.scatter(x_axis_worst, y_axis1_worst, color='red', marker='o')
plt.scatter(x_axis_worst, y_axis2_worst, color='red', marker='o')
plt.scatter(x_axis_worst, y_axis3_worst, color='red', marker='o')
plt.scatter(x_axis_worst, y_axis4_worst, color='red', marker='o')
plt.scatter(x_axis_worst, y_axis5_worst, color='red', marker='o')
plt.scatter(x_axis_worst, y_axis6_worst, color='red', marker='o')
plt.scatter(x_axis_worst, y_axis7_worst, color='red', marker='o')

plt.plot(x_axis_worst, mean_values_worst, color='green', linestyle='--',label='Average evacuation time')
plt.plot(x_axis_worst, upper_bound_worst, color='blue', linestyle='--')
plt.plot(x_axis_worst, lower_bound_worst, color='blue', linestyle='--')
plt.fill_between(x_axis_worst,upper_bound_worst,lower_bound_worst,color='blue',alpha=0.2)
# Add labels and title
plt.xlabel('Initial range [km]',fontsize=Fontsize)
plt.ylabel('Evacuation time [hours]',fontsize=Fontsize)
plt.tick_params(axis='both', labelsize=Fontsize)
# plt.title('Minimizing average evacuation time')
plt.legend(fontsize=Fontsize,loc='upper right')

##### Average evacuation time comparison ###########################################################################


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # type: ignore

Fontsize =15
x_axis_avg = [20,30,40,50,60,70,80]

df_avg = pd.read_excel(DATA_DIR / 'Mariposa_plots.xlsx', sheet_name='Avg_time_v1', header=None) 

rows_avg = df_avg.values.tolist()

y_axis1_avg = rows_avg[0]
y_axis2_avg = rows_avg[1]
y_axis3_avg = rows_avg[2]
y_axis4_avg = rows_avg[3]
y_axis5_avg = rows_avg[4]
y_axis6_avg = rows_avg[5]
y_axis7_avg = rows_avg[6]

mean_values_avg = [(a + b + c + d + e + f + g) / 7 for a, b, c, d, e, f, g in zip(y_axis1_avg, y_axis2_avg, y_axis3_avg, y_axis4_avg, y_axis5_avg, y_axis6_avg, y_axis7_avg)]
upper_bound_avg = [max(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in zip(y_axis1_avg, y_axis2_avg, y_axis3_avg, y_axis4_avg, y_axis5_avg, y_axis6_avg, y_axis7_avg)]
lower_bound_avg = [min(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in zip(y_axis1_avg, y_axis2_avg, y_axis3_avg, y_axis4_avg, y_axis5_avg, y_axis6_avg, y_axis7_avg)]
print(mean_values_avg)
plt.subplots()
plt.scatter(x_axis_avg, y_axis1_avg, color='red', marker='o')
plt.scatter(x_axis_avg, y_axis2_avg, color='red', marker='o')
plt.scatter(x_axis_avg, y_axis3_avg, color='red', marker='o')
plt.scatter(x_axis_avg, y_axis4_avg, color='red', marker='o')
plt.scatter(x_axis_avg, y_axis5_avg, color='red', marker='o')
plt.scatter(x_axis_avg, y_axis6_avg, color='red', marker='o')
plt.scatter(x_axis_avg, y_axis7_avg, color='red', marker='o')

plt.plot(x_axis_avg, mean_values_avg, color='green', linestyle='--',label='Average evacuation time')
plt.plot(x_axis_avg, upper_bound_avg, color='blue', linestyle='--')
plt.plot(x_axis_avg, lower_bound_avg, color='blue', linestyle='--')
plt.fill_between(x_axis_avg,upper_bound_avg,lower_bound_avg,color='blue',alpha=0.2)
# Add labels and title
plt.xlabel('Initial range [km]',fontsize=Fontsize)
plt.ylabel('Evacuation time [hours]',fontsize=Fontsize)
plt.tick_params(axis='both', labelsize=Fontsize)
# plt.title('Minimizing average evacuation time')
plt.legend(fontsize=Fontsize,loc='upper right')

##### Standard deviation evacuation time comparison ###########################################################################


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # type: ignore

Fontsize =15
x_axis_avg_sd = [20,30,40,50,60,70,80]

df_avg_sd = pd.read_excel(DATA_DIR / 'Mariposa_plots.xlsx', sheet_name='Avg_sd_v1', header=None) 

rows_avg_sd = df_avg_sd.values.tolist()

y_axis1_avg_sd = rows_avg_sd[0]
y_axis2_avg_sd = rows_avg_sd[1]
y_axis3_avg_sd = rows_avg_sd[2]
y_axis4_avg_sd = rows_avg_sd[3]
y_axis5_avg_sd = rows_avg_sd[4]
y_axis6_avg_sd = rows_avg_sd[5]
y_axis7_avg_sd = rows_avg_sd[6]

mean_values_avg_sd = [(a + b + c + d + e + f + g) / 7 for a, b, c, d, e, f, g in zip(y_axis1_avg_sd, y_axis2_avg_sd, y_axis3_avg_sd, y_axis4_avg_sd, y_axis5_avg_sd, y_axis6_avg_sd, y_axis7_avg_sd)]
upper_bound_avg_sd = [max(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in zip(y_axis1_avg_sd, y_axis2_avg_sd, y_axis3_avg_sd, y_axis4_avg_sd, y_axis5_avg_sd, y_axis6_avg_sd, y_axis7_avg_sd)]
lower_bound_avg_sd = [min(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in zip(y_axis1_avg_sd, y_axis2_avg_sd, y_axis3_avg_sd, y_axis4_avg_sd, y_axis5_avg_sd, y_axis6_avg_sd, y_axis7_avg_sd)]
print(mean_values_avg_sd)
plt.subplots()
plt.scatter(x_axis_avg_sd, y_axis1_avg_sd, color='red', marker='o')
plt.scatter(x_axis_avg_sd, y_axis2_avg_sd, color='red', marker='o')
plt.scatter(x_axis_avg_sd, y_axis3_avg_sd, color='red', marker='o')
plt.scatter(x_axis_avg_sd, y_axis4_avg_sd, color='red', marker='o')
plt.scatter(x_axis_avg_sd, y_axis5_avg_sd, color='red', marker='o')
plt.scatter(x_axis_avg_sd, y_axis6_avg_sd, color='red', marker='o')
plt.scatter(x_axis_avg_sd, y_axis7_avg_sd, color='red', marker='o')

plt.plot(x_axis_avg_sd, mean_values_avg_sd, color='green', linestyle='--',label='Average evacuation time')
plt.plot(x_axis_avg_sd, upper_bound_avg_sd, color='blue', linestyle='--')
plt.plot(x_axis_avg_sd, lower_bound_avg_sd, color='blue', linestyle='--')
plt.fill_between(x_axis_avg_sd,upper_bound_avg_sd,lower_bound_avg_sd,color='blue',alpha=0.2)
# Add labels and title
plt.xlabel('Initial range [km]',fontsize=Fontsize)
plt.ylabel('Evacuation time [hours]',fontsize=Fontsize)
plt.tick_params(axis='both', labelsize=Fontsize)
# plt.title('Minimizing average evacuation time')
plt.legend(fontsize=Fontsize,loc='upper right')

################################## Theta analysis for different cost function #################################################################
x_axis = [0.3309,0.339]
y_axis = [0.8074,0.8058]
x_axis11 = [0.3309,0.339]
y_axis11 = [0.8074,0.8058]

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
font_properties = {"fontname": "Times New Roman", "fontsize": 50, "color": "black"}
font_properties1 = {"fontname": "Times New Roman", "fontsize": 50, "color": "black"}
plt.subplots()
plt.scatter(x_axis11, y_axis11, color='blue', marker='o')
plt.plot(x_axis, y_axis, color='blue', linestyle='-')
plt.text(0.33129, 0.807389, r'$\theta = 0$', fontsize=30)
plt.text(0.337532, 0.805856, r'$\theta = 1$', fontsize=30)
# Add labels and title
plt.xlabel('J$^{\Delta}$ [hours]',fontsize=33)
plt.ylabel('J$^{avg}$ [hours]',fontsize=33)
plt.tick_params(axis='both', labelsize=28)
plt.grid()
#################################################################################################################################################
###########################################  Effect of MCS on the initial flow across od-pairs ##################################################
x_axis = [5,10,15,20]
y_axis = [25,40,50,60]
x_axis11 = [5,10,15,20]
y_axis11 = [25,40,50,60]

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
font_properties = {"fontname": "Times New Roman", "fontsize": 50, "color": "black"}
font_properties1 = {"fontname": "Times New Roman", "fontsize": 50, "color": "black"}
plt.subplots()
plt.scatter(x_axis11, y_axis11, color='blue', marker='o')
plt.plot(x_axis, y_axis, color='blue', linestyle='-')
# plt.text(0.11624, 0.41406, r'$\theta = 0$', fontsize=30)
# plt.text(0.16578, 0.37747, r'$\theta = 1$', fontsize=30)
# Add labels and title
plt.xlabel('# MCS',fontsize=33)
plt.ylabel('Initial flow',fontsize=33)
plt.tick_params(axis='both', labelsize=28)
plt.grid()
#################################################################################################################################################
#################################################  Road capacity analysis wrt initial vehicle flow  ##############################################
x_axis1 = list(range(1,1066))

dataframe1 = pd.read_excel('Mariposa_analysis.xlsx', sheet_name='Normalised_flow')

column_data1 = dataframe1['a']
column_data2 = dataframe1['b']
column_data3 = dataframe1['c']

# width = 2
# plt.spines['top'].set_linewidth(width)  # change width


font_properties2 = {"fontname": "Times New Roman", "fontsize": 50, "color": "black"}
plt.subplots()
plt.plot(x_axis1, column_data1, color='blue', linestyle='-',linewidth=2,label='$f_{in}$: 150 vehicles/hr')
plt.scatter(x_axis1, column_data1, color='blue', marker='o')
plt.plot(x_axis1, column_data2, color='purple', linestyle='-',linewidth=2,label='$f_{in}$: 200 vehicles/hr')
plt.scatter(x_axis1, column_data2, color='purple', marker='o')
plt.plot(x_axis1, column_data3, color='green', linestyle='-',linewidth=2,label='$f_{in}$: 300 vehicles/hr')
plt.scatter(x_axis1, column_data3, color='green', marker='o')
plt.xlabel('Link number',fontsize=40)
plt.ylabel('Normalized flow',fontsize=40)
plt.tick_params(axis='both', labelsize=20)
# plt.title('Minimizing deviation from avg. evacuation time')
plt.legend(fontsize=30,loc='upper right')
# plt.grid()

#################################################################################################################################################
######################################  box plot for the normalised road flow values#############################################################
dataframe1 = pd.read_excel(DATA_DIR / 'Mariposa_plots.xlsx', sheet_name='Normalised_flow')

column_data1 = np.array(dataframe1['a'])
column_data2 = np.array(dataframe1['b'])
column_data3 = np.array(dataframe1['c'])
column_data4 = np.array(dataframe1['d'])
column_data5 = np.array(dataframe1['e'])
column_data6 = np.array(dataframe1['f'])
column_data7 = np.array(dataframe1['g'])
column_data8 = np.array(dataframe1['h'])
column_data9 = np.array(dataframe1['i'])
column_data10 = np.array(dataframe1['j'])

# def filter(data):
#     Q1 = np.percentile(data, 25)
#     Q3 = np.percentile(data, 75)
#     IQR = Q3 - Q1

#     lower_whisker = Q1 - 1.5 * IQR
#     upper_whisker = Q3 + 1.5 * IQR

#     filtered_data = data[(data >= lower_whisker) & (data <= upper_whisker)]
#     return filtered_data

# column_data1 = filter(column_data1)
# column_data2 = filter(column_data2)
# column_data3 = filter(column_data3)
# column_data4 = filter(column_data4)
# column_data5 = filter(column_data5)
# column_data6 = filter(column_data6)
# column_data7 = filter(column_data7)
# column_data8 = filter(column_data8)
# column_data9 = filter(column_data9)
# column_data10 = filter(column_data10)

data = [column_data1,column_data2,column_data3,column_data4,column_data5,column_data6,column_data7,column_data8,column_data9,column_data10]
# print(data)
# labels = ['UEP', 'MILP-FCS']
labels = ['30', '60','90','120','150','180','210','240','270','300']
fig, ax = plt.subplots(figsize=(5, 6))

bplot = ax.boxplot(data, patch_artist=True,tick_labels=labels,showfliers=True)

colors = ['lightblue', 'lightblue','lightblue', 'lightblue','lightblue', 'lightblue','lightblue', 'lightblue','lightblue', 'lightblue']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# ax.set_title('Comparison of Two Data Sets')
# ax.set_xlabel('Algorithms',fontsize=25)
plt.xlabel('Initial vehicle flow across all od-pairs [vehicle/hour]',fontsize=25)
plt.ylabel('Normalized road vehicle flow value',fontsize=25)
# plt.ylim([-0.00000000000000000000002,0.00000000000000000000002])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

column_data1_nonzero_indices = np.nonzero(column_data1)[0]
data1_len = len(column_data1_nonzero_indices)
column_data1_filtered = np.zeros(data1_len)
for num, i in enumerate(column_data1_nonzero_indices):
    column_data1_filtered[num] = column_data1[i]

column_data2_nonzero_indices = np.nonzero(column_data2)[0]
data2_len = len(column_data2_nonzero_indices)
column_data2_filtered = np.zeros(data2_len)
for num, i in enumerate(column_data2_nonzero_indices):
    column_data2_filtered[num] = column_data2[i]

column_data3_nonzero_indices = np.nonzero(column_data3)[0]
data3_len = len(column_data3_nonzero_indices)
column_data3_filtered = np.zeros(data3_len)
for num, i in enumerate(column_data3_nonzero_indices):
    column_data3_filtered[num] = column_data3[i]

column_data4_nonzero_indices = np.nonzero(column_data4)[0]
data4_len = len(column_data4_nonzero_indices)
column_data4_filtered = np.zeros(data4_len)
for num, i in enumerate(column_data4_nonzero_indices):
    column_data4_filtered[num] = column_data4[i]

column_data5_nonzero_indices = np.nonzero(column_data5)[0]
data5_len = len(column_data5_nonzero_indices)
column_data5_filtered = np.zeros(data5_len)
for num, i in enumerate(column_data5_nonzero_indices):
    column_data5_filtered[num] = column_data5[i]

column_data6_nonzero_indices = np.nonzero(column_data6)[0]
data6_len = len(column_data6_nonzero_indices)
column_data6_filtered = np.zeros(data6_len)
for num, i in enumerate(column_data6_nonzero_indices):
    column_data6_filtered[num] = column_data6[i]

column_data7_nonzero_indices = np.nonzero(column_data7)[0]
data7_len = len(column_data7_nonzero_indices)
column_data7_filtered = np.zeros(data7_len)
for num, i in enumerate(column_data7_nonzero_indices):
    column_data7_filtered[num] = column_data7[i]

column_data8_nonzero_indices = np.nonzero(column_data8)[0]
data8_len = len(column_data8_nonzero_indices)
column_data8_filtered = np.zeros(data8_len)
for num, i in enumerate(column_data8_nonzero_indices):
    column_data8_filtered[num] = column_data8[i]

column_data9_nonzero_indices = np.nonzero(column_data9)[0]
data9_len = len(column_data9_nonzero_indices)
column_data9_filtered = np.zeros(data9_len)
for num, i in enumerate(column_data9_nonzero_indices):
    column_data9_filtered[num] = column_data9[i]

column_data10_nonzero_indices = np.nonzero(column_data10)[0]
data10_len = len(column_data10_nonzero_indices)
column_data10_filtered = np.zeros(data10_len)
for num, i in enumerate(column_data10_nonzero_indices):
    column_data10_filtered[num] = column_data10[i]


x_axis = [180,360,540,720,900,1080,1260,1440,1620,1800]
y_axis = [data1_len,data2_len,data3_len,data4_len,data5_len,data6_len,data7_len,data8_len,data9_len,data5_len]

plt.figure(figsize=(8, 5))
plt.bar(x_axis, y_axis, width=20)
plt.xticks(ticks=x_axis, labels=x_axis)
plt.xlabel("Initial vehicle flow across all od-pairs [vehicle/hour]",fontsize=25)
plt.ylabel("Number of edges utilized",fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.title("Bar Plot of Data Lengths vs Time")

plt.grid(axis="y", linestyle="--", alpha=0.6)


data1 = [column_data1_filtered,column_data2_filtered,column_data3_filtered,
         column_data4_filtered,column_data5_filtered,column_data6_filtered,
         column_data7_filtered,column_data8_filtered,column_data9_filtered,column_data10_filtered]

labels = ['180', '360','540','720','900','1080','1260','1440','1620','1800']
fig, ax = plt.subplots(figsize=(5, 6))

bplot = ax.boxplot(data1, patch_artist=True,tick_labels=labels,showfliers=True)

colors = ['lightblue', 'lightblue','lightblue', 'lightblue','lightblue', 'lightblue','lightblue', 'lightblue','lightblue', 'lightblue']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# ax.set_title('Comparison of Two Data Sets')
# ax.set_xlabel('Algorithms',fontsize=25)
plt.xlabel('Initial vehicle flow across all od-pairs [vehicle/hour]',fontsize=25)
plt.ylabel('Normalized road vehicle flow value',fontsize=25)
# plt.ylim([-0.00000000000000000000002,0.00000000000000000000002])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#################################################################################################################################################
#################################################  Comparison of road capacity for ECM and Baseline ##############################################
x_axis1 = list(range(1,1066))

dataframe1 = pd.read_excel('Mariposa_analysis.xlsx', sheet_name='Comparison_plots')

column_data1 = list(dataframe1['a'])
column_data2 = list(dataframe1['b'])
column_data1.sort(reverse=True)
column_data2.sort(reverse=True)

# width = 2
# plt.spines['top'].set_linewidth(width)  # change width


font_properties2 = {"fontname": "Times New Roman", "fontsize": 50, "color": "black"}
plt.subplots()
plt.plot(x_axis1, column_data1, color='blue', linestyle='-',linewidth=2,label='# MCS = 22')
plt.scatter(x_axis1, column_data1, color='blue', marker='o')
plt.plot(x_axis1, column_data2, color='purple', linestyle='-',linewidth=2,label='# MCS = 13')
plt.scatter(x_axis1, column_data2, color='purple', marker='o')
plt.xlabel('Link number',fontsize=40)
plt.ylabel('Normalized flow',fontsize=40)
plt.tick_params(axis='both', labelsize=20)
# plt.title('Minimizing deviation from avg. evacuation time')
plt.legend(fontsize=30,loc='upper right')
# plt.grid()

#################################################################################################################################################
###### Show the plot #################################################################################################
plt.show()