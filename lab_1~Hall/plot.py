#%% Imports
from pandas import read_csv
import matplotlib.pyplot as plot
from matplotlib.pyplot import cm
import numpy as np

def get_ls_line(x, y):
    """This returns the Slope & Y-intercept for the least-squares method of fitting"""
    pts_len = len(x)
    sum_x = sum([xpt for xpt in x])
    sum_y = sum([ypt for ypt in y])
    sum_xy = sum([xpt*ypt for xpt, ypt in zip(x, y)])
    sum_xsq = sum([xpt**2 for xpt in x])
    m = ((pts_len*sum_xy - sum_x*sum_y)/(pts_len*sum_xsq - sum_x**2))
    b = ((sum_y-m*sum_x)/pts_len)
    return m, b

def get_fit_quality_chi_sq(y, fit_y, y_acc):
    """This returns the chi-squared value for the data & fit line"""
    y_diff = [ypt - fitpt for ypt, fitpt in zip(y, fit_y)]
    return sum([(ypt - fitpt)**2/acc**2 for ypt, fitpt, acc in zip(y, fit_y, y_acc)])

#%% Get Data (Cr)
data_1 = read_csv('Cr_src1.txt')
data_2 = read_csv('Cr_src2.txt')
data_3 = read_csv('Cr_src3.txt')
plot_data = [np.array(data_1), np.array(data_2), np.array(data_3)]

trial_field_strengths = [770, 460, 320]

W = 2.2
L = 2.4

#%% Calculate
for i in range(0,3,1):
    plot_data[i][:,0] = plot_data[i][:,0]/W
    plot_data[i][:,2] = plot_data[i][:,2]/(L*W)

#%% Data Fitting for Sets
ls_fit = [[],[],[]]
fits = [[],[],[]]

for i in range(0, 3, 1):
    m, b = get_ls_line(plot_data[i][:,0], plot_data[i][:,2])
    ls_fit[i] = [b + m * x for x in plot_data[i][:,0]]
    fits[i] = [m, b]

#%% Plotting Average

# Build Plot
plot.style.use('ggplot')
plot.title("Hall Effect at Different Magnetic Field Strengths (Cr)", color='k')
plot.xlabel("Electric Field Strength($E_y$) [J]")
plot.ylabel("Charge Density($J_x$) [$A/m^2$]")

# Plot data
for i in range(0,3,1):
    colors = cm.jet((0.85,0.25,0.45))
    plot.errorbar(plot_data[i][:,0], plot_data[i][:,2], yerr=plot_data[i][:,3], 
                        xerr=plot_data[i][:,1], fmt='o', color=colors[i])
    plot.plot(plot_data[i][:,0], ls_fit[i], color=colors[i])

plot.legend(['Field Strength: {}G'.format(x) for x in trial_field_strengths])

# Display
fig = plot.gcf()
plot.figure()

#%% Save Figure
# fig.savefig('sexyplot.png', facecolor='w')

#%% Fit Quality for Average
for i in range(0,3,1):
    q = get_fit_quality_chi_sq(plot_data[i][:,2], ls_fit[i], plot_data[i][:,3])
    N = 2 # Always 2 for linear fit, really DOF
    print("reduced chi-squared: {}; chi-squared: {}; DOF: {};".format(q/N, q, N))
    # Get uncertainty
    pts_len = len(plot_data[i][:,0])
    delta = pts_len*sum([(1/x)**2 for x in plot_data[i][:,0]]) - sum([1/x for x in plot_data[i][:,0]])**2
    s_yxsq = (1/(pts_len - 2))*sum([(ypt - yest)**2 for ypt, yest in zip(plot_data[i][:,2], ls_fit[i])])
    s_m = np.sqrt(pts_len*(s_yxsq/delta))
    s_b = np.sqrt((s_yxsq*sum([(1/x)**2 for x in plot_data[i][:,0]]))/delta)
    print("slope: {}; intercept: {};".format(fits[i][0], fits[i][1]))
    print("slope error: {}; intercept error: {};".format(s_m, s_b))

#%% Plot Residuals
plot.style.use('ggplot')
for i in range(0, len(plot_data), 1):
    # print(len(ls_fit))
    plot.errorbar(1/plot_data[i][:,0], np.array(plot_data[i][:,2]) - np.array(ls_fit[i]), yerr=plot_data[i][:,3], 
                                            xerr=plot_data[i][:,1], fmt='o')
    plot.plot(1/plot_data[i][:,0], [0]*len(plot_data[i][:,0]))

plot.title("Residuals of Charge Density in Electric Fields (Cr)", color='k')
plot.xlabel("Electric Field Strength($E_y$) [J]")
plot.ylabel("Standardized Residuals for Charge Density($J_x$)")
plot.legend(['Field Strength: {}G'.format(x) for x in trial_field_strengths])
fig = plot.gcf()
plot.figure()

#%% Get Data (Ag)
data_1 = read_csv('Ag_src1.txt')
data_2 = read_csv('Ag_src2.txt')
data_3 = read_csv('Ag_src3.txt')
plot_data = [np.array(data_1), np.array(data_2), np.array(data_3)]

trial_field_strengths = [810, 590, 390]

W = 2.3
L = 2.2

#%% Calculate
for i in range(0,3,1):
    plot_data[i][:,0] = plot_data[i][:,0]/W
    plot_data[i][:,2] = plot_data[i][:,2]/(L*W)

#%% Data Fitting for Sets
ls_fit = [[],[],[]]
fits = [[],[],[]]

for i in range(0, 3, 1):
    m, b = get_ls_line(plot_data[i][:,0], plot_data[i][:,2])
    ls_fit[i] = [b + m * x for x in plot_data[i][:,0]]
    fits[i] = [m, b]

#%% Plotting Average

# Build Plot
plot.style.use('ggplot')
plot.title("Hall Effect at Different Magnetic Field Strengths (Ag)", color='k')
plot.xlabel("Electric Field Strength($E_y$) [J]")
plot.ylabel("Charge Density($J_x$) [$A/m^2$]")

# Plot data
for i in range(0,3,1):
    colors = cm.jet((0.85,0.25,0.45))
    plot.errorbar(plot_data[i][:,0], plot_data[i][:,2], yerr=plot_data[i][:,3], 
                        xerr=plot_data[i][:,1], fmt='o', color=colors[i])
    plot.plot(plot_data[i][:,0], ls_fit[i], color=colors[i])

plot.legend(['Field Strength: {}G'.format(x) for x in trial_field_strengths])

# Display
fig = plot.gcf()
plot.figure()

#%% Save Figure
# fig.savefig('sexyplot.png', facecolor='w')

#%% Fit Quality for Average
for i in range(0,3,1):
    q = get_fit_quality_chi_sq(plot_data[i][:,2], ls_fit[i], plot_data[i][:,3])
    N = 2 # Always 2 for linear fit, really DOF
    print("reduced chi-squared: {}; chi-squared: {}; DOF: {};".format(q/N, q, N))
    # Get uncertainty
    pts_len = len(plot_data[i][:,0])
    delta = pts_len*sum([(1/x)**2 for x in plot_data[i][:,0]]) - sum([1/x for x in plot_data[i][:,0]])**2
    s_yxsq = (1/(pts_len - 2))*sum([(ypt - yest)**2 for ypt, yest in zip(plot_data[i][:,2], ls_fit[i])])
    s_m = np.sqrt(pts_len*(s_yxsq/delta))
    s_b = np.sqrt((s_yxsq*sum([(1/x)**2 for x in plot_data[i][:,0]]))/delta)
    print("slope: {}; intercept: {};".format(fits[i][0], fits[i][1]))
    print("slope error: {}; intercept error: {};".format(s_m, s_b))

#%% Plot Residuals
plot.style.use('ggplot')
for i in range(0, len(plot_data), 1):
    # print(len(ls_fit))
    plot.errorbar(1/plot_data[i][:,0], np.array(plot_data[i][:,2]) - np.array(ls_fit[i]), yerr=plot_data[i][:,3], 
                                            xerr=plot_data[i][:,1], fmt='o')
    plot.plot(1/plot_data[i][:,0], [0]*len(plot_data[i][:,0]))

plot.title("Residuals of Charge Density in Electric Fields (Ag)", color='k')
plot.xlabel("Electric Field Strength($E_y$) [J]")
plot.ylabel("Standardized Residuals for Charge Density($J_x$)")
plot.legend(['Field Strength: {}G'.format(x) for x in trial_field_strengths])
fig = plot.gcf()
plot.figure()