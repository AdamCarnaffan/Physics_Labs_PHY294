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
initPower = 76.1
silverPower = 14.7
plotTransmittances = [0.6, 0.9, 1.1, 1.5, 2.8, 3.5, 4.3, 5.2, 7.5]
plotThicknesses = [73, 68, 65.5, 62, 54.5, 52.8, 50, 46.5, 28]
plot_data = np.array([[np.log(x), 0.05, y, 0] for x, y in zip(plotTransmittances, plotThicknesses)])

print(plot_data)

#%% Data Fitting for Sets
m, b = get_ls_line(plot_data[:,0], plot_data[:,2])
ls_fit = [b + m * x for x in plot_data[:,0]]
fits = [m, b]

#%% Plotting Average

# Build Plot
plot.style.use('ggplot')
plot.title("Transmissions of $470\lambda$ at Different Silver Film Thicknesses", color='k')
plot.xlabel("Transmission (ln(T)) [%]")
plot.ylabel("Thickness [$nm$]")

# Plot data
# colors = cm.jet((0.85,0.25,0.45))
plot.errorbar(plot_data[:,0], plot_data[:,2], yerr=plot_data[:,3], 
                    xerr=plot_data[:,1], fmt='o')
plot.plot(plot_data[:,0], ls_fit)

# plot.legend(['Field Strength: {}G'.format(x) for x in trial_field_strengths])

# Display
fig = plot.gcf()
plot.figure()

#%% Fit Quality for Average
    # q = get_fit_quality_chi_sq(plot_data[:,2], ls_fit, plot_data[:,3])
    # N = 2 # Always 2 for linear fit, really DOF
    # print("reduced chi-squared: {}; chi-squared: {}; DOF: {};".format(q/N, q, N))
    # Get uncertainty
    pts_len = len(plot_data[:,0])
    delta = pts_len*sum([(1/x)**2 for x in plot_data[:,0]]) - sum([1/x for x in plot_data[:,0]])**2
    s_yxsq = (1/(pts_len - 2))*sum([(ypt - yest)**2 for ypt, yest in zip(plot_data[:,2], ls_fit)])
    s_m = np.sqrt(pts_len*(s_yxsq/delta))
    s_b = np.sqrt((s_yxsq*sum([(1/x)**2 for x in plot_data[:,0]]))/delta)
    print("slope: {}; intercept: {};".format(fits[0], fits[1]))
    print("slope error: {}; intercept error: {};".format(s_m, s_b))

#%% Plot Residuals
plot.style.use('ggplot')
    # print(len(ls_fit))
plot.errorbar(1/plot_data[:,0], np.array(plot_data[:,2]) - np.array(ls_fit), yerr=plot_data[:,3], 
                                        xerr=plot_data[:,1], fmt='o')
plot.plot(1/plot_data[:,0], [0]*len(plot_data[:,0]))

plot.title("Residuals of Charge Density in Electric Fields (Cr)", color='k')
plot.xlabel("Electric Field Strength($E_y$) [J]")
plot.ylabel("Standardized Residuals for Charge Density($J_x$)")
plot.legend(['Field Strength: {}G'.format(x) for x in trial_field_strengths])
fig = plot.gcf()
plot.figure()