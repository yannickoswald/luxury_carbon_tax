
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
##############################################  Static model  ##############################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################ 

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  SET UP BASICS #######################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

##### 'infrastructure' loading ######
import os
os.getcwd()
os.chdir("your path")


import pandas as pd 
import math 
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
import scipy as scipy
import scipy.special as ssp
import copy
from gini import *
from giniold import *
from gini_dynamic import *
from gini_v2 import *
from gini_array_version import *
from lin_fit import *
from lin_fit_non_log import *
from scipy.stats import gamma
import math as math
from matplotlib import rc
import matplotlib.gridspec as gridspec
#import dash
#import dash_core_components as dcc
#import dash_html_components as html
#from dash.dependencies import Input, Output
from scipy.special import erfinv
#import plotly.express as px
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
from numpy import asarray
from numpy import savetxt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
import matplotlib.image as mpimg


meta_data_countries = np.genfromtxt('country_national_level_meta.csv', dtype = str, delimiter=',');
BIG_exp_2019_pc = np.genfromtxt('df_BIG_total_exp_2019_pc.csv', dtype = float, delimiter=',');
carbon_intensities_2019_estimate = np.genfromtxt('carbon_intensities_2019_estimate.csv', dtype = float, delimiter=',');
income_elasticities = np.expand_dims(np.genfromtxt('income_elasticities.csv', dtype = float, delimiter=','),axis=1); ###e-01 so it is not wrong just notated that way
income_elasticities_SE = np.expand_dims(np.genfromtxt('income_elasticities_SE.csv', dtype = float, delimiter=','),axis=1);
population_WB_2019 = np.genfromtxt('population_WB_2019.csv', dtype = str, delimiter=',');
labels = np.genfromtxt('labels.csv', dtype = str, delimiter=',');
one_column_ginis = np.genfromtxt('one_column_ginis.csv', dtype = float, delimiter=',');

cum_dist_fig2 = np.genfromtxt('cum_dist_fig2.csv', dtype = str, delimiter=',');


labels2 = list()  ##### are the income groups
for i in range(1,101):
       labels2.append(str(i))

income_elasticities_original_save = income_elasticities

############ ROBUSTNESS SIMPLIFICATION, REIGNING IN OUTLIERS WITH ASSUMPTIONS #####################
income_elasticities = np.where(income_elasticities < 0, 0.1, income_elasticities) #### replace negative elasticities with concave engel curve.
np.nan_to_num(income_elasticities_SE, copy = False, nan = 0)
##### we clip standard error of elasticities to avoid hypersensitivity of model to parameters, only an extremely minor share of parameters is affected by this. 
SE_as_fraction_of_elas = income_elasticities_SE/abs(income_elasticities) #### coefficient_of_variation SE/mean https://en.wikipedia.org/wiki/Coefficient_of_variation
SE_as_fraction_of_elas_2 = SE_as_fraction_of_elas.clip(max = 1) ### restrict coefficient of variation to 1 , https://www.readyratios.com/reference/analysis/coefficient_of_variation.html
SE_as_fraction_of_elas_2 = np.where(SE_as_fraction_of_elas_2 == 0, 1, SE_as_fraction_of_elas_2) ### assuming maximum uncertainty where we interpolated data with an elasticity of 1, i.e. set coefficient of variation to 1.
income_elasticities_SE = SE_as_fraction_of_elas_2 * abs(income_elasticities)
carbon_intensities_2019_estimate = carbon_intensities_2019_estimate.clip(max = 13.89)  ###we clip carbon intensities of consumption to avoid hypersensitivity of model to parameters, only one value, heating and electricity in Belarus, is affected by this.



### create population per percentile vector 1232 long
pop_percentile_2019 = np.zeros((1232,1));
for i in range(1,89):
       pop_percentile_2019[14*i-14:14*i] = float(population_WB_2019[i-1][1])/100


df_BIG_exp_2019_pc  = pd.DataFrame(data = BIG_exp_2019_pc, columns = labels2, index = labels)
df_BIG_carbon_2019_pc = df_BIG_exp_2019_pc.multiply(carbon_intensities_2019_estimate, axis = 0)
df_BIG_carbon_2019_total =  df_BIG_carbon_2019_pc.multiply(pop_percentile_2019, axis = 0)
df_BIG_exp_2019_total = df_BIG_exp_2019_pc.multiply(pop_percentile_2019, axis = 0)
check1 = sum(df_BIG_carbon_2019_total.sum())

df_BIG_total_emissions_per_category_in_country = df_BIG_carbon_2019_total.sum(axis = 1)

array_BIG_total_emissions_per_country = np.zeros((88,1));

for i in range(1,89):
   array_BIG_total_emissions_per_country[i-1] = sum(df_BIG_carbon_2019_total.iloc[14*i-14:14*i].sum());



#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  MODEL SET-UP analysis ###############################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

###### !!!!!!!!!!!!!!!!!!!!stand alone plots  !!!!!!!!!!!!!!!!!!!!!!!!!!#######
##########################################################
####### set up #1 PLOT elasticities vs. Gini coefficient #
##########################################################
##########################################################
##https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
### potential colormaps https://matplotlib.org/stable/tutorials/colors/colormaps.html
x = np.squeeze(income_elasticities_original_save, axis =1);
y = one_column_ginis;
#### fit linear model ####
results_lin_fit_1 = lin_fit_non_log(x,y)
x_model = np.linspace(-0.2,3.5,100)
y_model = results_lin_fit_1[0][0]+results_lin_fit_1[0][1]*x_model
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=50, cmap=plt.cm.plasma);### https://matplotlib.org/stable/tutorials/colors/colormaps.html
ax.set_xlabel('income elasticity of demand');
ax.set_ylabel('Gini coefficient');
ax.plot(x_model, y_model, linewidth=4.0, c = 'r', linestyle = '--');
ax.plot([0, 0], [0, 1], linewidth = 2.0, c = 'black', linestyle = '--');
ax.margins(x=0,y=0);
ax.text(2, 0.1, r'$y=0.06+0.24*x$', fontsize=10)
ax.text(2, 0.02, r'$R^2$= 0.55', fontsize=10)
plt.savefig('fig1a.png',bbox_inches = "tight", dpi = 300);
plt.show();

##### histogram/distribution of residuals --> check for normality
plt.hist(results_lin_fit_1[1]);
plt.xlabel('residual value');
plt.ylabel('frequency')
plt.savefig('supp_fig_fig1a_residuals.png',bbox_inches = "tight", dpi = 300);
plt.show();

#######################################################################################################
#######################################################################################################
####### set up #2 PLOT carbon emissions per capita vs cumulative population "parade of dwarfs" ########
#######################################################################################################
#######################################################################################################

df_pc_yearly_emissions_total_2019 = pd.DataFrame(columns = labels2, index = meta_data_countries[:,0])
for i in range(1,89):  
     for j in range(0,100):
            df_pc_yearly_emissions_total_2019.iloc[i-1,j] = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14,j].sum();
            
stacked_pc_emissions_per_country = df_pc_yearly_emissions_total_2019.stack();
stacked_pc_emissions_per_country = stacked_pc_emissions_per_country.to_frame();
stacked_population_percentiles = np.zeros((8800,1));
for i in range(1,89):
    stacked_population_percentiles[i*100-100:i*100]= float(population_WB_2019[i-1][1])/100;
            
stacked_pc_emissions_per_country.insert(1, '1', stacked_population_percentiles) 
stacked_pc_emissions_per_country = stacked_pc_emissions_per_country.sort_values(0)
stacked_pc_emissions_per_country[0] = stacked_pc_emissions_per_country[0]/1000
cumulative_population_dist_graph = stacked_pc_emissions_per_country['1'].cumsum()/stacked_pc_emissions_per_country['1'].sum()

######## empirical data for 2011 CO2emissions ########
y1 = np.expand_dims(cum_dist_fig2[:,3],axis=1).astype(np.float)
x1 = np.expand_dims(cum_dist_fig2[:,2],axis=1).astype(np.float)
ya2 = stacked_pc_emissions_per_country.loc['USA', '100'][0]
xa2 = cumulative_population_dist_graph.loc['USA', '100']
ya3 = stacked_pc_emissions_per_country.loc['USA', '1'][0]
xa3 = cumulative_population_dist_graph.loc['USA', '1']
ya4 = stacked_pc_emissions_per_country.loc['CHN', '100'][0]
xa4 = cumulative_population_dist_graph.loc['CHN', '100']
ya5 = stacked_pc_emissions_per_country.loc['CHN', '1'][0]
xa5 = cumulative_population_dist_graph.loc['CHN', '1']
ya6 = stacked_pc_emissions_per_country.loc['IND', '100'][0]
xa6 = cumulative_population_dist_graph.loc['IND', '100']
ya7 = stacked_pc_emissions_per_country.loc['IND', '1'][0]
xa7 = cumulative_population_dist_graph.loc['IND', '1']

#### linewidth = 5
#### more code for annotations ####
####https://matplotlib.org/stable/gallery/text_labels_and_annotations/annotation_demo.html######
####### graph #######
plt.plot(cumulative_population_dist_graph , stacked_pc_emissions_per_country[0], label = 'model for 2019', linewidth = 4);
plt.plot(x1,y1, linestyle = '--', label = 'empirical for 2011', linewidth = 3);
plt.yscale("log");
plt.xlabel("Cumulative population");
plt.ylabel("CO2e tonnes/capita");
plt.legend(frameon = False);
plt.annotate('USA top 1%',
            xy=(xa2, ya2), xytext =(0.84,0.5), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))
plt.annotate('USA bottom 1%',
            xy=(xa3, ya3), xytext =(0.55,0.2), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))
plt.annotate('CHN top 1%',
            xy=(xa4, ya4), xytext =(0.55,30), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(0.3, 0.6, 0.7), ec="none"))
plt.annotate('CHN bottom 1%',
            xy=(xa5, ya5), xytext =(0.2,0.1), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(0.3, 0.6, 0.7), ec="none"))
plt.annotate('IND top 1%',
            xy=(xa6, ya6), xytext =(0.4,10), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(0.8, 0.6, 0.2), ec="none"))
plt.annotate('IND bottom 1%',
            xy=(xa7, ya7), xytext =(0.1,4), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(0.8, 0.6, 0.2), ec="none"))
plt.savefig('fig1c.png',bbox_inches = "tight", dpi = 300);
plt.show();

###########################################################################################################
###########################################################################################################



###################################################################################
###################################################################################
########set up #3 PLOT example differentiated pricing system; Example = USA #######
###################################################################################
###################################################################################
###### without normalizing
category_names = labels[0:14, 3].tolist()
category_names = [i[:-4] for i in category_names]
category_names = [i[1:] for i in category_names]
category_names[5]= 'Household Appliances'
category_names[8]= 'Vehicle Fuel'
category_names[13]= 'Education and Luxury'
price_uniform = np.repeat(150,14)
income_elasticities_USA  = income_elasticities[1232-14:1232]
price_differentiated = np.multiply(np.expand_dims(price_uniform,axis=1),income_elasticities_USA)*1.15908 ### normalization constant for the USA


fig, ax = plt.subplots()
y_pos = np.arange(len(category_names))
ax.barh(y_pos, price_uniform, align='center', label = 'uniform')
ax.barh( y_pos, np.squeeze(price_differentiated, axis =1 ), height =0.45, align='center', label = 'luxury')
ax.set_yticks(y_pos)
ax.set_yticklabels(category_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Carbon price $/tonne');
ax.legend(frameon = False)
ax.annotate('USA', xy=(200,4),fontsize=20)
plt.savefig('fig1b.png',bbox_inches = "tight", dpi = 300);
plt.show()


##### same plot for China for control and comparison
price_uniform_CHINA = np.repeat(50,14)
income_elasticities_CHINA = income_elasticities[14*12-14:12*14]
price_differentiated_CHINA = np.multiply(np.expand_dims(price_uniform_CHINA,axis=1),income_elasticities_CHINA)*1.02232
 ### normalization constant for China


fig, ax = plt.subplots()
y_pos = np.arange(len(category_names))
ax.barh(y_pos, price_uniform_CHINA, align='center', label = 'uniform')
ax.barh(y_pos, np.squeeze(price_differentiated_CHINA, axis =1 ), height =0.45, align='center', label = 'luxury')
ax.set_yticks(y_pos)
ax.set_yticklabels(category_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Carbon price $/tonne');
ax.legend(frameon = False)
ax.annotate('CHINA', xy=(55,4),fontsize=20)
plt.savefig('fig1b_CHINA.png',bbox_inches = "tight", dpi = 300);
plt.show()



##### same plot for South Africa for control and comparison
price_uniform_SA = np.repeat(50,14)
income_elasticities_SA = income_elasticities[14*48-14:48*14]
price_differentiated_SA = np.multiply(np.expand_dims(price_uniform_SA,axis=1),income_elasticities_SA)*1.05383



fig, ax = plt.subplots()
y_pos = np.arange(len(category_names))
ax.barh(y_pos, price_uniform_SA, align='center', label = 'uniform')
ax.barh(y_pos, np.squeeze(price_differentiated_SA, axis =1 ), height =0.45, align='center', label = 'luxury')
ax.set_yticks(y_pos)
ax.set_yticklabels(category_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Carbon price $/tonne');
ax.legend(frameon = False)
ax.annotate('South Africa', xy=(55,4),fontsize=20)
plt.savefig('fig1b_South_Africa.png',bbox_inches = "tight", dpi = 300);
plt.show()


##### same for cubed elasticity version ############




###################################################################################
###################################################################################
########set up #3 PLOT example differentiated pricing system; Example = USA #######
###################################################################################
###################################################################################
###### without normalizing
category_names = labels[0:14, 3].tolist()
category_names = [i[:-4] for i in category_names]
category_names = [i[1:] for i in category_names]
category_names[5]= 'Household Appliances'
category_names[8]= 'Vehicle Fuel'
category_names[13]= 'Education and Luxury'
price_uniform = np.repeat(150,14)
income_elasticities_USA  = income_elasticities[1232-14:1232]
price_differentiated = np.multiply(np.expand_dims(price_uniform,axis=1),income_elasticities_USA**3)*0.938024 ### normalization constant for the USA


fig, ax = plt.subplots()
y_pos = np.arange(len(category_names))
ax.barh(y_pos, price_uniform, align='center', label = 'uniform')
ax.barh( y_pos, np.squeeze(price_differentiated, axis =1 ), height =0.45, align='center', label = 'luxury')
ax.set_yticks(y_pos)
ax.set_yticklabels(category_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Carbon price $/tonne');
ax.legend(frameon = False)
ax.annotate('USA', xy=(200,4),fontsize=20)
plt.savefig('fig1b.png',bbox_inches = "tight", dpi = 300);
plt.show()


##### same plot for China for control and comparison
price_uniform_CHINA = np.repeat(50,14)
income_elasticities_CHINA = income_elasticities[14*12-14:12*14]
price_differentiated_CHINA = np.multiply(np.expand_dims(price_uniform_CHINA,axis=1),income_elasticities_CHINA**3)*0.973105 ## cubed normalization constant
 ### normalization constant for China


fig, ax = plt.subplots()
y_pos = np.arange(len(category_names))
ax.barh(y_pos, price_uniform_CHINA, align='center', label = 'uniform')
ax.barh(y_pos, np.squeeze(price_differentiated_CHINA, axis =1 ), height =0.45, align='center', label = 'luxury')
ax.set_yticks(y_pos)
ax.set_yticklabels(category_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Carbon price $/tonne');
ax.legend(frameon = False)
ax.annotate('CHINA', xy=(55,4),fontsize=20)
plt.savefig('fig1b_CHINA.png',bbox_inches = "tight", dpi = 300);
plt.show()



##### same plot for South Africa for control and comparison
price_uniform_SA = np.repeat(50,14)
income_elasticities_SA = income_elasticities[14*48-14:48*14]
price_differentiated_SA = np.multiply(np.expand_dims(price_uniform_SA,axis=1),income_elasticities_SA**3)*0.620955




fig, ax = plt.subplots()
y_pos = np.arange(len(category_names))
ax.barh(y_pos, price_uniform_SA, align='center', label = 'uniform')
ax.barh(y_pos, np.squeeze(price_differentiated_SA, axis =1 ), height =0.45, align='center', label = 'luxury')
ax.set_yticks(y_pos)
ax.set_yticklabels(category_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Carbon price $/tonne');
ax.legend(frameon = False)
ax.annotate('South Africa', xy=(55,4),fontsize=20)
plt.savefig('fig1b_South_Africa.png',bbox_inches = "tight", dpi = 300);
plt.show()


######!!!!!!!!!!!!!!!!!!!!!!! JOINT PLOT!!!!!!!!!!!!!!!!!!!! #######


#https://matplotlib.org/stable/tutorials/intermediate/gridspec.html

widths_grid = [2, 1]
heights_grid = [2, 2]
fig = plt.figure(constrained_layout=True, figsize=(6,5))


gs = GridSpec(2, 2, figure=fig ,width_ratios = widths_grid, height_ratios = heights_grid)
ax1 = fig.add_subplot(gs[1, :-1])
ax2 = fig.add_subplot(gs[1, -1])
ax3 = fig.add_subplot(gs[0, :])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))

### potential colormaps https://matplotlib.org/stable/tutorials/colors/colormaps.html
x = np.squeeze(income_elasticities_original_save, axis =1);
y = one_column_ginis;
#### fit linear model ####
results_lin_fit_1 = lin_fit_non_log(x,y)
x_model = np.linspace(-0.2,3.5,100)
y_model = results_lin_fit_1[0][0]+results_lin_fit_1[0][1]*x_model
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
ax1.scatter(x, y, c=z, s=15, cmap=plt.cm.plasma);### https://matplotlib.org/stable/tutorials/colors/colormaps.html
ax1.set_xlabel('income elasticity of demand');
ax1.set_ylabel('Gini coefficient');
ax1.plot(x_model, y_model, linewidth=3.0, c = 'black', linestyle = '--');
ax1.plot([0, 0], [0, 1], linewidth = 2.0, c = 'black', linestyle = '--');
ax1.margins(x=0,y=0);
ax1.text(1.7, 0.12, r'$y=0.06+0.24*x$', fontsize=8.5)
ax1.text(1.7, 0.01, r'$R^2$= 0.55', fontsize=8.5)
ax1.text(0.1, 0.8, r'N = 1232', fontsize=8.5)

ax1.annotate('b', xy=(ax1.get_xlim()[0],1.03),fontsize=12 ,annotation_clip=False)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


###################################################################################
###################################################################################
########set up #3 PLOT example differentiated pricing system; Example = USA #######
###################################################################################
###################################################################################
###### without normalizing
category_names = labels[0:14, 3].tolist()
category_names = [i[:-4] for i in category_names]
category_names = [i[1:] for i in category_names]
category_names[5]= 'Household Appliances'
category_names[8]= 'Vehicle Fuel'
category_names[13]= 'Education and Luxury'
price_uniform = np.repeat(150,14)
income_elasticities_USA  = income_elasticities[1232-14:1232]
price_differentiated = np.multiply(np.expand_dims(price_uniform,axis=1),income_elasticities_USA)*1.15908 ### normalization constant for the USA
y_pos = np.arange(len(category_names))
ax2.barh(y_pos, price_uniform, align='center', label = 'uniform')
ax2.barh(y_pos, np.squeeze(price_differentiated, axis =1 ), height =0.45, align='center', label = 'luxury')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(category_names, fontsize = 6)
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.set_xlabel('carbon price $/tonne');
ax2.annotate('USA', xy=(200,4),fontsize=10)
ax2.legend(frameon = False, fontsize = 5)
ax2.annotate('c', xy=(ax2.get_xlim()[0],-1.5),fontsize=12 ,annotation_clip=False)

#######################################################################################################
#######################################################################################################
####### set up #2 PLOT carbon emissions per capita vs cumulative population "parade of dwarfs" ########
#######################################################################################################
#######################################################################################################
#### linewidth = 5

#### more code for annotations ####
####https://matplotlib.org/stable/gallery/text_labels_and_annotations/annotation_demo.html######
####### graph #######
ax3.plot(cumulative_population_dist_graph , stacked_pc_emissions_per_country[0], label = 'model for 2019', linewidth = 4);
ax3.plot(x1,y1, linestyle = '--', label = 'empirical for 2011', linewidth = 3);
ax3.set_yscale("log");
ax3.set_xlabel("cumulative population");
ax3.set_ylabel("CO2e tonnes/capita");
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax3.set_yticks([0.1,1,10,100]);
#ax3.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax3.legend(frameon = False, fontsize = 8);
ax3.annotate('USA top 1%',
            xy=(xa2, ya2), xytext =(0.84,0.5), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"), fontsize = 8)
ax3.annotate('USA bottom 1%',
            xy=(xa3, ya3), xytext =(0.55,0.2), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"), fontsize = 8)
ax3.annotate('CHN top 1%',
            xy=(xa4, ya4), xytext =(0.6,30), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(0.3, 0.6, 0.7), ec="none"), fontsize = 8)
ax3.annotate('CHN bottom 1%',
            xy=(xa5, ya5), xytext =(0.2,0.1), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(0.3, 0.6, 0.7), ec="none"), fontsize = 8)
ax3.annotate('IND top 1%',
            xy=(xa6, ya6), xytext =(0.4,10), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(0.8, 0.6, 0.2), ec="none"), fontsize = 8)
ax3.annotate('IND bottom 1%',
            xy=(xa7, ya7), xytext =(0.1,3), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0), bbox=dict(boxstyle="round", fc=(0.8, 0.6, 0.2), ec="none"), fontsize = 8)
ax3.annotate('a', xy=(ax3.get_xlim()[0],ax3.get_ylim()[1]+25),fontsize=12 ,annotation_clip=False)

plt.savefig('fig2.png',bbox_inches = "tight", dpi = 300);
plt.show();

###########################################################################################################
###########################################################################################################





#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  Scenario Analysis #1 ################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
###### SET CARBON PRICES $/tonne ####

Low_income_Cprice = 10;
Lower_middle_income_Cprice = 25;
Upper_middle_income_Cprice = 50;
High_income_Cprice = 150;



##############make budget share data frame ##############
df_budget_share_pc =pd.DataFrame(columns = labels2, index = labels)
for i in range(1,1233):
      j = math.ceil(i/14)
      df_budget_share_pc.iloc[i-1,0:100] = df_BIG_exp_2019_pc.iloc[i-1,0:100]/df_BIG_exp_2019_pc.iloc[j*14-14:j*14,0:100].sum()

N = 100


#########################################################################
#####uniform SCENARIO PREPARATION carbon price = 0.05$/kg is 50 $/tonne####
#########################################################################
scen1_national_emissions = np.zeros((88,N+1))
scen1_national_expenditure = np.zeros((88,2))
#scen1_df_embodied_carbon_costs_pc = df_BIG_carbon_2019_pc.multiply(0.05)



#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  CAUTION !!!!! ###############################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


##### make price array with uniform prices ######

scen1_df_embodied_carbon_costs_pc = pd.DataFrame(columns = labels2, index = labels);
for i in range(1,89):
    if df_BIG_carbon_2019_pc.index[14*i-14][0]  == 'Low income': scen1_df_embodied_carbon_costs_pc.iloc[i*14-14:i*14] = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14].multiply(Low_income_Cprice/1000);
    if df_BIG_carbon_2019_pc.index[14*i-14][0]  == 'Lower middle income': scen1_df_embodied_carbon_costs_pc.iloc[i*14-14:i*14] = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14].multiply(Lower_middle_income_Cprice/1000);
    if df_BIG_carbon_2019_pc.index[14*i-14][0]  == 'Upper middle income': scen1_df_embodied_carbon_costs_pc.iloc[i*14-14:i*14] = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14].multiply(Upper_middle_income_Cprice/1000);
    if df_BIG_carbon_2019_pc.index[14*i-14][0]  == 'High income': scen1_df_embodied_carbon_costs_pc.iloc[i*14-14:i*14] = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14].multiply(High_income_Cprice/1000);
   


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  CAUTION !!!!! ###############################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################




   
scen1_df_price_increase_through_tax = scen1_df_embodied_carbon_costs_pc/df_BIG_exp_2019_pc   
    
#########################################################################
#####LUXURY SCENARIO PREPARATION carbon price is differentiated by elasticity and country group
#########################################################################
#### luxury scenario pre calculation
#scen2_df_embodied_carbon_costs_pc_pre_normalization = df_BIG_carbon_2019_pc.multiply(income_elasticities*0.05)

scen2_df_embodied_carbon_costs_pc_pre_normalization = pd.DataFrame(columns = labels2, index = labels);

exponent_parameter = 1
##### costs in $/kg
for i in range(1,89):
    if df_BIG_carbon_2019_pc.index[14*i-14][0]  == 'Low income': scen2_df_embodied_carbon_costs_pc_pre_normalization.iloc[i*14-14:i*14] = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14].multiply(income_elasticities[i*14-14:i*14]**exponent_parameter*Low_income_Cprice/1000);
    if df_BIG_carbon_2019_pc.index[14*i-14][0]  == 'Lower middle income': scen2_df_embodied_carbon_costs_pc_pre_normalization.iloc[i*14-14:i*14] = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14].multiply(income_elasticities[i*14-14:i*14]**exponent_parameter*Lower_middle_income_Cprice/1000);
    if df_BIG_carbon_2019_pc.index[14*i-14][0]  == 'Upper middle income': scen2_df_embodied_carbon_costs_pc_pre_normalization.iloc[i*14-14:i*14] = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14].multiply(income_elasticities[i*14-14:i*14]**exponent_parameter*Upper_middle_income_Cprice/1000);
    if df_BIG_carbon_2019_pc.index[14*i-14][0]  == 'High income': scen2_df_embodied_carbon_costs_pc_pre_normalization.iloc[i*14-14:i*14] = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14].multiply(income_elasticities[i*14-14:i*14]**exponent_parameter*High_income_Cprice/1000);
   




##### calculate NORMALIZATION CONSTANT for each country for scenario #2 so that carbon costs embodied per country can be made equivalent to scenario #1
embodied_costs_per_country_uniform = np.zeros((88,1));
embodied_costs_per_country_luxury_pre = np.zeros((88,1));
for i in range(1,89):
    embodied_costs_per_country_uniform[i-1] = scen1_df_embodied_carbon_costs_pc.multiply(pop_percentile_2019).sum(axis=1).iloc[14*i-14:i*14].sum()
    embodied_costs_per_country_luxury_pre[i-1] = scen2_df_embodied_carbon_costs_pc_pre_normalization.multiply(pop_percentile_2019).sum(axis=1).iloc[14*i-14:i*14].sum()

normalization_constant_scen_luxury_to_uniform = embodied_costs_per_country_uniform/embodied_costs_per_country_luxury_pre

#####Luxury SCENARIO carbon price is differentiated per product with income elasticity of demand
###### once with normalization (i.e. carbon costs in society same as in uniform scenario, once without)

### 2.1 without normalization            
scen2_1_df_embodied_carbon_costs_pc = scen2_df_embodied_carbon_costs_pc_pre_normalization;           
scen2_1_df_price_increase_through_tax = scen2_1_df_embodied_carbon_costs_pc/df_BIG_exp_2019_pc;
ratio_between_carbon_costs_embodied_scen1_and_scen2_1 = scen2_1_df_embodied_carbon_costs_pc.sum(axis = 1)/scen1_df_embodied_carbon_costs_pc.sum(axis = 1)

scen2_1_national_emissions = np.zeros((88,N+1));
scen2_1_national_expenditure = np.zeros((88,2));

### 2.2 with normalization
scen2_2_df_embodied_carbon_costs_pc = pd.DataFrame(columns = labels2, index = labels)
for i in range(1,89):
     scen2_2_df_embodied_carbon_costs_pc.iloc[i*14-14:i*14] = scen2_df_embodied_carbon_costs_pc_pre_normalization.iloc[i*14-14:i*14]*float(normalization_constant_scen_luxury_to_uniform[i-1]);   
scen2_2_df_price_increase_through_tax = scen2_2_df_embodied_carbon_costs_pc/df_BIG_exp_2019_pc;

scen2_2_national_emissions = np.zeros((88,N+1));
scen2_2_national_expenditure = np.zeros((88,2));


################ SET UP LOOP DATA COLLECTIONS FOR VARIOUS VARIABLES WHICH ARE COLLECTED IN BIG SIM LOOP BELOW  ###########

##tax revenue
scen1_national_tax_revenue = np.zeros((88,N));
scen2_1_national_tax_revenue = np.zeros((88,N));
scen2_2_national_tax_revenue = np.zeros((88,N));

##total expenditure
scen1_total_exp = np.zeros((1232, 100));
scen2_2_total_exp = np.zeros((1232, 100));

### total diff for all countries 1st and last percentile
diff_first_and_last_percentile_reduction_emissions_SCEN1_simruns = np.zeros((88,100));
diff_first_and_last_percentile_reduction_emissions_SCEN2_2_simruns = np.zeros((88,100));

##USA

USA_post_tax_carbon_cons_pc_SCEN1_total = np.zeros((100,100));
USA_post_tax_carbon_cons_pc_SCEN2_2_total = np.zeros((100,100)); 
USA_per_cent_reduction_emissions_pc_SCEN2_2 = np.zeros((100,100));
USA_per_cent_reduction_emissions_pc_SCEN1 = np.zeros((100,100));
USA_abated_carbon_per_category_share_SCEN1_simruns = np.zeros((14,100));              
USA_abated_carbon_per_category_share_SCEN2_2_simruns = np.zeros((14,100));


##CHINA
CHINA_post_tax_carbon_cons_pc_SCEN1_total = np.zeros((100,100));
CHINA_post_tax_carbon_cons_pc_SCEN2_2_total = np.zeros((100,100));            
CHINA_per_cent_reduction_emissions_pc_SCEN2_2 = np.zeros((100,100));
CHINA_per_cent_reduction_emissions_pc_SCEN1 = np.zeros((100,100));
CHINA_abated_carbon_per_category_share_SCEN1_simruns = np.zeros((14,100));              
CHINA_abated_carbon_per_category_share_SCEN2_2_simruns = np.zeros((14,100));

##SOUTH AFRICA
SA_post_tax_carbon_cons_pc_SCEN1_total = np.zeros((100,100));
SA_post_tax_carbon_cons_pc_SCEN2_2_total = np.zeros((100,100));            
SA_per_cent_reduction_emissions_pc_SCEN2_2 = np.zeros((100,100));
SA_per_cent_reduction_emissions_pc_SCEN1 = np.zeros((100,100));
SA_abated_carbon_per_category_share_SCEN1_simruns = np.zeros((14,100));              
SA_abated_carbon_per_category_share_SCEN2_2_simruns = np.zeros((14,100));



##### make list for reduced cons. dataframes##

list1 = []
list2 = []
list_cons1 = []
list_cons2 = []
list_tax1 = []
list_tax2 = []

##including cons. and tax fee
list_tax_incidence_luxury = []
list_tax_incidence_uniform = [] 


####
share_of_spends_tax_scen2_2 = np.zeros((88,100))
share_of_spends_tax_scen1 = np.zeros((88,100))

#############################################################
############ MAIN SCENARIO SIMULATION LOOP ##################
print("start tax impact simulation")
#############################################################

for k in range(1,N+1):    
                 
            
            #### make price elasticity dataframe #### based on Sabetelli 2016 mapping model, method section 
            roh = -1.26 + np.random.normal(0,0.05) #### elasticity of the marginal utility of income, #### one std ~0.05 based on layard et al. 95% CI
            #roh_upper_bound = -1.19
            #roh_lower_bound = -1.34 a
            ####according to Laynard et al. 2008 so therefore 0.1~ 2 standard deviations i.e. 95% CI of normal dist.
            
            ########### PRICE ELASTICIT MODEL BASE CASE ###########
            
            df_first_term_0 = (-1/roh)*df_budget_share_pc
            df_first_term = df_first_term_0.multiply((income_elasticities+np.random.normal(0,income_elasticities_SE))**2, axis = 1)
            df_second_term_0 = (1/roh)-df_budget_share_pc
            df_second_term = df_second_term_0.multiply(income_elasticities+np.random.normal(0,income_elasticities_SE), axis = 1)
            df_price_elasticities = df_first_term + df_second_term
            
            
            ######### make new (phyiscal so to speak) consumption volume
            
            
            
            test = (1+(scen1_df_price_increase_through_tax.mul(df_price_elasticities)))
            test2 = (1+(scen2_2_df_price_increase_through_tax.mul(df_price_elasticities)))
            
            ######SCEN #1
            scen1_df_post_tax_consumption_volume_pc = df_BIG_exp_2019_pc.mul((1+(scen1_df_price_increase_through_tax.mul(df_price_elasticities))))  
            scen1_df_post_tax_consumption_volume_pc[scen1_df_post_tax_consumption_volume_pc<0]=0; #### make sure no negative consumption occurs. If price high enough, it becomes prohibitive price. 
            
            scen1_df_post_tax_consumption_volume_total = scen1_df_post_tax_consumption_volume_pc.multiply(pop_percentile_2019, axis = 0)
            ######### make post tax carbon emissions
            scen1_df_post_tax_emissions_pc = scen1_df_post_tax_consumption_volume_pc.multiply(carbon_intensities_2019_estimate, axis = 0)
            scen1_df_post_tax_emissions_total = scen1_df_post_tax_consumption_volume_total.multiply(carbon_intensities_2019_estimate, axis = 0)
            
            scen1_tax_revenue = scen1_df_price_increase_through_tax*scen1_df_post_tax_consumption_volume_total;
    
            
            scen1_reduced_consumption = df_BIG_exp_2019_total - scen1_df_post_tax_consumption_volume_total
            
            list1.append(scen1_reduced_consumption)
            list_cons1.append(scen1_df_post_tax_consumption_volume_total)
            list_tax1.append(scen1_tax_revenue)
    
    
    
            ######SCEN #2.1 without normalization
            scen2_1_df_post_tax_consumption_volume_pc = df_BIG_exp_2019_pc.mul((1+(scen2_1_df_price_increase_through_tax.mul(df_price_elasticities)))) 
            scen2_1_df_post_tax_consumption_volume_pc[scen2_1_df_post_tax_consumption_volume_pc<0]=0;
            
            scen2_1_df_post_tax_consumption_volume_total = scen2_1_df_post_tax_consumption_volume_pc.multiply(pop_percentile_2019, axis = 0)
            scen2_1_df_post_tax_emissions_pc = scen2_1_df_post_tax_consumption_volume_pc.multiply(carbon_intensities_2019_estimate, axis = 0)
            scen2_1_df_post_tax_emissions_total = scen2_1_df_post_tax_consumption_volume_total.multiply(carbon_intensities_2019_estimate, axis = 0)
            
            scen2_1_tax_revenue = scen2_1_df_price_increase_through_tax*scen2_1_df_post_tax_consumption_volume_total;
            
            
            ######SCEN #2.2 with normalization
            scen2_2_df_post_tax_consumption_volume_pc = df_BIG_exp_2019_pc.mul((1+(scen2_2_df_price_increase_through_tax.mul(df_price_elasticities)))) 
            scen2_2_df_post_tax_consumption_volume_pc[scen2_2_df_post_tax_consumption_volume_pc<0]=0;
            
            scen2_2_df_post_tax_consumption_volume_total = scen2_2_df_post_tax_consumption_volume_pc.multiply(pop_percentile_2019, axis = 0)
            scen2_2_df_post_tax_emissions_pc = scen2_2_df_post_tax_consumption_volume_pc.multiply(carbon_intensities_2019_estimate, axis = 0)
            scen2_2_df_post_tax_emissions_total = scen2_2_df_post_tax_consumption_volume_total.multiply(carbon_intensities_2019_estimate, axis = 0)
                        
            scen2_2_tax_revenue = scen2_2_df_price_increase_through_tax*scen2_2_df_post_tax_consumption_volume_total;
            
            
            
            scen2_2_reduced_consumption = df_BIG_exp_2019_total - scen2_2_df_post_tax_consumption_volume_total
            
            list2.append(scen2_2_reduced_consumption)
            list_cons2.append(scen2_2_df_post_tax_consumption_volume_total)
            list_tax2.append(scen2_2_tax_revenue)
            
            ##################### SAVE TOTAL EXPENDITURE FOR ILLUSTRATING IMPACT ON EXPENDITURE VOLUME #########
            scen1_total_exp[:,k-1] = scen1_df_post_tax_consumption_volume_total.sum(axis =1)
            scen2_2_total_exp[:,k-1] = scen2_2_df_post_tax_consumption_volume_total.sum(axis =1)
            
            ##### detailed emission reduction data for USA ####
            
            
            USA_pre_tax_carbon_cons_total = df_BIG_carbon_2019_total.iloc[88*14-14:88*14]
            USA_post_tax_carbon_cons_total_SCEN1 = scen1_df_post_tax_emissions_total.iloc[88*14-14:88*14]
            USA_post_tax_carbon_cons_total_SCEN2_2 = scen2_2_df_post_tax_emissions_total.iloc[88*14-14:88*14]
            
            USA_pre_tax_carbon_cons_pc = df_BIG_carbon_2019_pc.iloc[88*14-14:88*14]
            USA_post_tax_carbon_cons_pc_SCEN1 = scen1_df_post_tax_emissions_pc.iloc[88*14-14:88*14]
            USA_post_tax_carbon_cons_pc_SCEN2_2 = scen2_2_df_post_tax_emissions_pc.iloc[88*14-14:88*14]
            
            USA_pre_tax_carbon_cons_pc_total = USA_pre_tax_carbon_cons_pc.sum()
            USA_post_tax_carbon_cons_pc_SCEN1_total[:,k-1] = USA_post_tax_carbon_cons_pc_SCEN1.sum()
            USA_post_tax_carbon_cons_pc_SCEN2_2_total[:,k-1]= USA_post_tax_carbon_cons_pc_SCEN2_2.sum()
            
            USA_per_cent_reduction_emissions_pc_SCEN2_2[:,k-1] = (1-USA_post_tax_carbon_cons_pc_SCEN2_2_total[:,k-1]/USA_pre_tax_carbon_cons_pc_total)*100
            USA_per_cent_reduction_emissions_pc_SCEN1[:,k-1] = (1-USA_post_tax_carbon_cons_pc_SCEN1_total[:,k-1]/USA_pre_tax_carbon_cons_pc_total)*100
                                  
                            
            USA_total_abated_carbon_SCEN1 = (sum(USA_pre_tax_carbon_cons_total.sum(axis = 1))-sum(USA_post_tax_carbon_cons_total_SCEN1.sum(axis = 1)))/(10**12) #### unit in gigatonne
            USA_total_abated_carbon_SCEN2_2 = (sum(USA_pre_tax_carbon_cons_total.sum(axis = 1))-sum(USA_post_tax_carbon_cons_total_SCEN2_2.sum(axis = 1)))/(10**12) #### unit in gigatonne
            USA_abated_carbon_per_category_SCEN1 = (USA_pre_tax_carbon_cons_total.sum(axis = 1) - USA_post_tax_carbon_cons_total_SCEN1.sum(axis = 1))/(10**12) #### gigatonne unit
            USA_abated_carbon_per_category_share_SCEN1_simruns[:,k-1] = USA_abated_carbon_per_category_SCEN1/USA_total_abated_carbon_SCEN1              
            USA_abated_carbon_per_category_SCEN2_2 = (USA_pre_tax_carbon_cons_total.sum(axis = 1) - USA_post_tax_carbon_cons_total_SCEN2_2.sum(axis = 1))/(10**12) #### gigatonne unit
            USA_abated_carbon_per_category_share_SCEN2_2_simruns[:,k-1] = USA_abated_carbon_per_category_SCEN2_2/USA_total_abated_carbon_SCEN2_2

            
            
            ##### detailed emission reduction data for INDIA ####
            
            
            
            ##### detailed emission reduction data for Brazil ####
            
            
            
             ##### detailed emission reduction data for South Africa ####
            
            
            SA_pre_tax_carbon_cons_total = df_BIG_carbon_2019_total.iloc[48*14-14:48*14] ### 48 index SA +1 
            SA_post_tax_carbon_cons_total_SCEN1 = scen1_df_post_tax_emissions_total.iloc[48*14-14:48*14]
            SA_post_tax_carbon_cons_total_SCEN2_2 = scen2_2_df_post_tax_emissions_total.iloc[48*14-14:48*14]
            
            SA_pre_tax_carbon_cons_pc = df_BIG_carbon_2019_pc.iloc[48*14-14:48*14]
            SA_post_tax_carbon_cons_pc_SCEN1 = scen1_df_post_tax_emissions_pc.iloc[48*14-14:48*14]
            SA_post_tax_carbon_cons_pc_SCEN2_2 = scen2_2_df_post_tax_emissions_pc.iloc[48*14-14:48*14]
            
            SA_pre_tax_carbon_cons_pc_total = SA_pre_tax_carbon_cons_pc.sum()
            SA_post_tax_carbon_cons_pc_SCEN1_total[:,k-1] = SA_post_tax_carbon_cons_pc_SCEN1.sum()
            SA_post_tax_carbon_cons_pc_SCEN2_2_total[:,k-1]= SA_post_tax_carbon_cons_pc_SCEN2_2.sum()
            
            SA_per_cent_reduction_emissions_pc_SCEN2_2[:,k-1] = (1-SA_post_tax_carbon_cons_pc_SCEN2_2_total[:,k-1]/SA_pre_tax_carbon_cons_pc_total)*100
            SA_per_cent_reduction_emissions_pc_SCEN1[:,k-1] = (1-SA_post_tax_carbon_cons_pc_SCEN1_total[:,k-1]/SA_pre_tax_carbon_cons_pc_total)*100
                                  
                            
            SA_total_abated_carbon_SCEN1 = (sum(SA_pre_tax_carbon_cons_total.sum(axis = 1))-sum(SA_post_tax_carbon_cons_total_SCEN1.sum(axis = 1)))/(10**12) #### unit in gigatonne
            SA_total_abated_carbon_SCEN2_2 = (sum(SA_pre_tax_carbon_cons_total.sum(axis = 1))-sum(SA_post_tax_carbon_cons_total_SCEN2_2.sum(axis = 1)))/(10**12) #### unit in gigatonne
            SA_abated_carbon_per_category_SCEN1 = (SA_pre_tax_carbon_cons_total.sum(axis = 1) - SA_post_tax_carbon_cons_total_SCEN1.sum(axis = 1))/(10**12) #### gigatonne unit
            SA_abated_carbon_per_category_share_SCEN1_simruns[:,k-1] = SA_abated_carbon_per_category_SCEN1/SA_total_abated_carbon_SCEN1              
            SA_abated_carbon_per_category_SCEN2_2 = (SA_pre_tax_carbon_cons_total.sum(axis = 1) - SA_post_tax_carbon_cons_total_SCEN2_2.sum(axis = 1))/(10**12) #### gigatonne unit
            SA_abated_carbon_per_category_share_SCEN2_2_simruns[:,k-1] = SA_abated_carbon_per_category_SCEN2_2/SA_total_abated_carbon_SCEN2_2

            
            
            
            ##### detailed emission reduction data for China ####
                       
            CHINA_pre_tax_carbon_cons_total = df_BIG_carbon_2019_total.iloc[12*14-14:12*14]
            CHINA_post_tax_carbon_cons_total_SCEN1 = scen1_df_post_tax_emissions_total.iloc[12*14-14:12*14]
            CHINA_post_tax_carbon_cons_total_SCEN2_2 = scen2_2_df_post_tax_emissions_total.iloc[12*14-14:12*14]
            
            CHINA_pre_tax_carbon_cons_pc = df_BIG_carbon_2019_pc.iloc[12*14-14:12*14]
            CHINA_post_tax_carbon_cons_pc_SCEN1 = scen1_df_post_tax_emissions_pc.iloc[12*14-14:12*14]
            CHINA_post_tax_carbon_cons_pc_SCEN2_2 = scen2_2_df_post_tax_emissions_pc.iloc[12*14-14:12*14]
            
            CHINA_pre_tax_carbon_cons_pc_total = CHINA_pre_tax_carbon_cons_pc.sum()
            CHINA_post_tax_carbon_cons_pc_SCEN1_total[:,k-1] = CHINA_post_tax_carbon_cons_pc_SCEN1.sum()
            CHINA_post_tax_carbon_cons_pc_SCEN2_2_total[:,k-1] = CHINA_post_tax_carbon_cons_pc_SCEN2_2.sum()
            
            CHINA_per_cent_reduction_emissions_pc_SCEN2_2[:,k-1] = (1-CHINA_post_tax_carbon_cons_pc_SCEN2_2_total[:,k-1]/CHINA_pre_tax_carbon_cons_pc_total)*100
            CHINA_per_cent_reduction_emissions_pc_SCEN1[:,k-1] = (1-CHINA_post_tax_carbon_cons_pc_SCEN1_total[:,k-1]/CHINA_pre_tax_carbon_cons_pc_total)*100
                       
            
            
            CHINA_total_abated_carbon_SCEN1 = (sum(CHINA_pre_tax_carbon_cons_total.sum(axis = 1))-sum(CHINA_post_tax_carbon_cons_total_SCEN1.sum(axis = 1)))/(10**12) #### unit in gigatonne
            CHINA_total_abated_carbon_SCEN2_2 = (sum(CHINA_pre_tax_carbon_cons_total.sum(axis = 1))-sum(CHINA_post_tax_carbon_cons_total_SCEN2_2.sum(axis = 1)))/(10**12) #### unit in gigatonne        
            CHINA_abated_carbon_per_category_SCEN1 = (CHINA_pre_tax_carbon_cons_total.sum(axis = 1) - CHINA_post_tax_carbon_cons_total_SCEN1.sum(axis = 1))/(10**12) #### gigatonne unit
            CHINA_abated_carbon_per_category_share_SCEN1_simruns[:,k-1] = CHINA_abated_carbon_per_category_SCEN1/CHINA_total_abated_carbon_SCEN1    
            CHINA_abated_carbon_per_category_SCEN2_2 = (CHINA_pre_tax_carbon_cons_total.sum(axis = 1) - CHINA_post_tax_carbon_cons_total_SCEN2_2.sum(axis = 1))/(10**12) #### gigatonne unit
            CHINA_abated_carbon_per_category_share_SCEN2_2_simruns[:,k-1] = CHINA_abated_carbon_per_category_SCEN2_2/CHINA_total_abated_carbon_SCEN2_2
                        
            
            ####### compare pre tax emissions per country vs post tax   ######
            
            if k == 1:
                for i in range(1,89):
                    scen1_national_emissions[i-1,0]= sum(df_BIG_carbon_2019_total.iloc[i*14-14:i*14,0:100].sum()) ##### 2019 pre tax
                    scen1_national_expenditure[i-1,1] = sum(df_BIG_exp_2019_total.iloc[i*14-14:i*14,0:100].sum())  ##### pre tax
                    scen2_1_national_emissions[i-1,0]= sum(df_BIG_carbon_2019_total.iloc[i*14-14:i*14,0:100].sum()) ##### 2019 pre tax
                    scen2_1_national_expenditure[i-1,1] = sum(df_BIG_exp_2019_total.iloc[i*14-14:i*14,0:100].sum())  ##### pre tax
                    scen2_2_national_emissions[i-1,0]= sum(df_BIG_carbon_2019_total.iloc[i*14-14:i*14,0:100].sum()) ##### 2019 pre tax
                    scen2_2_national_expenditure[i-1,1] = sum(df_BIG_exp_2019_total.iloc[i*14-14:i*14,0:100].sum())  ##### pre tax
            
            for i in range(1,89):  
                scen1_national_emissions[i-1,k]= sum(scen1_df_post_tax_emissions_total.iloc[i*14-14:i*14,0:100].sum()); #### post tax 
                scen2_1_national_emissions[i-1,k] = sum(scen2_1_df_post_tax_emissions_total.iloc[i*14-14:i*14,0:100].sum());
                scen2_2_national_emissions[i-1,k] = sum(scen2_2_df_post_tax_emissions_total.iloc[i*14-14:i*14,0:100].sum());
                
                scen1_national_tax_revenue[i-1,k-1] = sum(scen1_tax_revenue.iloc[i*14-14:i*14,0:100].sum());
                scen2_1_national_tax_revenue[i-1,k-1] = sum(scen2_1_tax_revenue.iloc[i*14-14:i*14,0:100].sum());
                scen2_2_national_tax_revenue[i-1,k-1] = sum(scen2_2_tax_revenue.iloc[i*14-14:i*14,0:100].sum());
                
                ######### compute global emissions distribution like in Fig 1c post tax so that later difference between the two can be checked and who had to reduce most 
                ######## progessivity vs. regressivity of tax 
            
             
                countryX_pre_tax_carbon_cons_total = df_BIG_carbon_2019_total.iloc[i*14-14:i*14]
                countryX_post_tax_carbon_cons_total_SCEN1 = scen1_df_post_tax_emissions_total.iloc[i*14-14:i*14]
                countryX_post_tax_carbon_cons_total_SCEN2_2 = scen2_2_df_post_tax_emissions_total.iloc[i*14-14:i*14]
                    
                countryX_pre_tax_carbon_cons_pc = df_BIG_carbon_2019_pc.iloc[i*14-14:i*14]
                countryX_post_tax_carbon_cons_pc_SCEN1 = scen1_df_post_tax_emissions_pc.iloc[i*14-14:i*14]
                countryX_post_tax_carbon_cons_pc_SCEN2_2 = scen2_2_df_post_tax_emissions_pc.iloc[i*14-14:i*14]
                    
                countryX_pre_tax_carbon_cons_pc_total = countryX_pre_tax_carbon_cons_pc.sum()
                countryX_post_tax_carbon_cons_pc_SCEN1_total = countryX_post_tax_carbon_cons_pc_SCEN1.sum()
                countryX_post_tax_carbon_cons_pc_SCEN2_2_total = countryX_post_tax_carbon_cons_pc_SCEN2_2.sum()
                    
                countryX_per_cent_reduction_emissions_pc_SCEN2_2 = (1-countryX_post_tax_carbon_cons_pc_SCEN2_2_total/countryX_pre_tax_carbon_cons_pc_total)*100
                countryX_per_cent_reduction_emissions_pc_SCEN1 = (1-countryX_post_tax_carbon_cons_pc_SCEN1_total/countryX_pre_tax_carbon_cons_pc_total)*100
                percentiles = np.arange(1,101,1)
                    
                diff_first_and_last_percentile_reduction_emissions_SCEN2_2_simruns[i-1, k-1] = countryX_per_cent_reduction_emissions_pc_SCEN2_2[0]-countryX_per_cent_reduction_emissions_pc_SCEN2_2[99]
                diff_first_and_last_percentile_reduction_emissions_SCEN1_simruns[i-1, k-1] = countryX_per_cent_reduction_emissions_pc_SCEN1[0]-countryX_per_cent_reduction_emissions_pc_SCEN1[99]

            



                share_of_spends_tax_scen2_2[i-1,:] = (scen2_2_df_price_increase_through_tax*scen2_2_df_post_tax_consumption_volume_pc).iloc[14*i-14:14*i,:].sum(axis = 0)/(scen2_2_df_post_tax_consumption_volume_pc+scen2_2_df_price_increase_through_tax*scen2_2_df_post_tax_consumption_volume_pc).iloc[14*i-14:14*i,:].sum(axis = 0)
        
                share_of_spends_tax_scen1[i-1,:] = (scen1_df_price_increase_through_tax*scen1_df_post_tax_consumption_volume_pc).iloc[14*i-14:14*i,:].sum(axis = 0)/(scen1_df_post_tax_consumption_volume_pc+scen1_df_price_increase_through_tax*scen1_df_post_tax_consumption_volume_pc).iloc[14*i-14:14*i,:].sum(axis = 0)
                       
            list_tax_incidence_luxury.append(share_of_spends_tax_scen2_2)
            list_tax_incidence_uniform.append(share_of_spends_tax_scen1)
            
            print('iteration is ' + str(k))
                
                

scen1_average_reduced_emissions = np.mean(scen1_national_emissions[:,1:101], axis = 1);
scen2_1_average_reduced_emissions = np.mean(scen2_1_national_emissions[:,1:101], axis = 1);
scen2_2_average_reduced_emissions = np.mean(scen2_2_national_emissions[:,1:101], axis = 1);


scen1_average_national_tax_revenue = np.mean(scen1_national_tax_revenue[:,0:100], axis = 1);
scen2_1_average_national_tax_revenue = np.mean(scen2_1_national_tax_revenue[:,0:100], axis = 1);
scen2_2_average_national_tax_revenue = np.mean(scen2_2_national_tax_revenue[:,0:100], axis = 1);


summ1 = pd.DataFrame(data = 0, columns = labels2, index = labels)
summ2 = pd.DataFrame(data = 0, columns = labels2, index = labels)
summ3 = pd.DataFrame(data = 0, columns = labels2, index = labels)
summ4 = pd.DataFrame(data = 0, columns = labels2, index = labels) 
summ5 = pd.DataFrame(data = 0, columns = labels2, index = labels)
summ6 = pd.DataFrame(data = 0, columns = labels2, index = labels)
summ7 = np.zeros((88,100))
summ8 = np.zeros((88,100))
for i in range(1,101):
    summ1 = summ1 + list1[i-1]
    summ2 = summ2 + list2[i-1]
    summ3 = summ3 + list_cons1[i-1]
    summ4 = summ4 + list_cons2[i-1]
    summ5 = summ5 + list_tax1[i-1]
    summ6 = summ6 + list_tax2[i-1]
    summ7 = summ7 + list_tax_incidence_luxury[i-1]
    summ8 = summ8 + list_tax_incidence_uniform[i-1]
    
    
scen1_average_reduced_consumption_granular = summ1/100;
scen2_2_average_reduced_consumption_granular = summ2/100;
scen1_average_consumption_granular = summ3/100;
scen2_2_average_consumption_granular = summ4/100;
scen1_average_tax_revenue_granular = summ5/100;
scen2_2_average_tax_revenue_granular = summ6/100

scen2_2_average_tax_incidence_granular = summ7/100;
scen1_average_tax_incidence_granular = summ8/100



######### plots for FIG 1b with flexible normalization constant


###################################################################################
###################################################################################
########set up #3 PLOT example differentiated pricing system; Example = USA #######
###################################################################################
###################################################################################
###### without normalizing
category_names = labels[0:14, 3].tolist()
category_names = [i[:-4] for i in category_names]
category_names = [i[1:] for i in category_names]
category_names[5]= 'Household Appliances'
category_names[8]= 'Vehicle Fuel'
category_names[13]= 'Education and Luxury'
price_uniform = np.repeat(150,14)
income_elasticities_USA  = income_elasticities[1232-14:1232]
price_differentiated = np.multiply(np.expand_dims(price_uniform,axis=1),income_elasticities_USA**exponent_parameter)*normalization_constant_scen_luxury_to_uniform[87] ### normalization constant for the USA


fig, ax = plt.subplots()
y_pos = np.arange(len(category_names))
ax.barh(y_pos, price_uniform, align='center', label = 'uniform')
ax.barh( y_pos, np.squeeze(price_differentiated, axis =1 ), height =0.45, align='center', label = 'luxury')
ax.set_yticks(y_pos)
ax.set_yticklabels(category_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Carbon price $/tonne');
ax.legend(frameon = False)
ax.annotate('USA', xy=(200,4),fontsize=20)
plt.savefig('fig2b.png',bbox_inches = "tight", dpi = 300);
plt.show()


##### same plot for China for control and comparison
price_uniform_CHINA = np.repeat(50,14)
income_elasticities_CHINA = income_elasticities[14*12-14:12*14]
price_differentiated_CHINA = np.multiply(np.expand_dims(price_uniform_CHINA,axis=1),income_elasticities_CHINA**exponent_parameter)*normalization_constant_scen_luxury_to_uniform[11]
 ### normalization constant for China


fig, ax = plt.subplots()
y_pos = np.arange(len(category_names))
ax.barh(y_pos, price_uniform_CHINA, align='center', label = 'uniform')
ax.barh(y_pos, np.squeeze(price_differentiated_CHINA, axis =1 ), height =0.45, align='center', label = 'luxury')
ax.set_yticks(y_pos)
ax.set_yticklabels(category_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Carbon price $/tonne');
ax.legend(frameon = False)
ax.annotate('CHINA', xy=(55,4),fontsize=20)
plt.savefig('fig2b_CHINA.png',bbox_inches = "tight", dpi = 300);
plt.show()





#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  NEW SECTION!!!!! ###############################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


#####################################################################################################################################
#####################################################################################################################################
                              ############## ANALYSIS SCEN and Graphing ##################   
#####################################################################################################################################
#####################################################################################################################################


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  NEW SECTION !!!!! ###############################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################



######################################## GRAPH #2 #######################################################################################################################
###################### revenue as % of GDP vs. emissions reduced through reduced consumption as % of total emissions national level #####################################
#########################################################################################################################################################################


GDP_national_total = meta_data_countries[:,3].astype(float)*meta_data_countries[:,5].astype(float);
scen1_revenue_as_percent_of_GDP = scen1_average_national_tax_revenue/GDP_national_total * 100;
scen2_1_revenue_as_percent_of_GDP = scen2_1_average_national_tax_revenue/GDP_national_total * 100;
scen2_2_revenue_as_percent_of_GDP = scen2_2_average_national_tax_revenue/GDP_national_total * 100;

scen1_average_reduced_emissions_as_percent_total = (1-np.divide(np.squeeze(scen1_average_reduced_emissions),np.squeeze(array_BIG_total_emissions_per_country)))*100
scen2_1_average_reduced_emissions_as_percent_total = (1-np.divide(np.squeeze(scen2_1_average_reduced_emissions),np.squeeze(array_BIG_total_emissions_per_country)))*100
scen2_2_average_reduced_emissions_as_percent_total = (1-np.divide(np.squeeze(scen2_2_average_reduced_emissions),np.squeeze(array_BIG_total_emissions_per_country)))*100




df = pd.DataFrame(dict(x = scen1_revenue_as_percent_of_GDP, y = scen1_average_reduced_emissions_as_percent_total, label=meta_data_countries[:,1]))
df2 = pd.DataFrame(dict(x = scen2_2_revenue_as_percent_of_GDP, y = scen2_2_average_reduced_emissions_as_percent_total, label=meta_data_countries[:,1]))

groups = df.groupby('label')
groups2 = df2.groupby('label')
#colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
#colors1 = plt.cm.get_cmap('Blues') # colors1(1-1/run/3)
#colors2 = plt.cm.get_cmap('Reds')
markers = ['.', '^', "+", "*"]

run = 0 
run2 = 0
fig, ax1 = plt.subplots()
for name, group in groups:
    run = run + 1
    ax1.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=4, label=name, color = 'b');
for name, group in groups2:
    run2 = run2 + 1
    ax1.plot(group.x, group.y, marker=markers[run2-1], linestyle='', ms=4, label=name, color = 'r');
ax1.set_yscale('log' ,basey = 2)
ax1.set_xscale('log', basex = 2)
ax1.set_xlim(1/16,8)
ax1.set_ylim(1/8,32)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.set_xlabel('Tax revenue as % of GDP')
ax1.set_ylabel('National emissions reduction %')
for i in range(0, len(scen1_revenue_as_percent_of_GDP)):
    ax1.plot([scen1_revenue_as_percent_of_GDP[i], scen2_2_revenue_as_percent_of_GDP[i]], [scen1_average_reduced_emissions_as_percent_total[i], scen2_2_average_reduced_emissions_as_percent_total[i]], color = 'black', linestyle = '-', linewidth = 1/2)
#https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
#https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph
#### https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
legend_elements2 = [Line2D([0], [0], marker ='^', color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='Low income, CP = $10'),
                    Line2D([0], [0], marker = '+', color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='Lower middle income, CP = $25'),
                    Line2D([0], [0], marker ='*',  color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='Upper middle income, CP = $50'),
                    Line2D([0], [0], marker ='.',  color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='High income, CP = $150'),
                      ]
ax1.legend(handles=legend_elements2, loc='lower right', title = "Symbol", fontsize = 10, frameon = False)
ax1.plot([0.1, 0.125], [16, 16],color = 'red', linestyle = '-', linewidth = 4);
ax1.plot([0.1, 0.125], [12, 12],color = 'blue', linestyle = '-', linewidth = 4);
ax1.annotate("uniform tax", (0.14, 11), (0.14, 11));
ax1.annotate("Luxury tax", (0.14, 15), (0.14, 15));
ax1.annotate("Color", (0.12, 22), (0.12, 22));
plt.show();



######################################## GRAPH #3 ########################################################################################################
###################### GDP per capita vs. emissions reduced through reduced consumption as % of total emissions national level #####################################




df3 = pd.DataFrame(dict(x = meta_data_countries[:,3].astype(float), y = scen1_average_reduced_emissions_as_percent_total, label=meta_data_countries[:,6]))

groups = df3.groupby('label')
markers = ['*', '^', "+", ".", "o"]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
run = 0;
fig, ax1 = plt.subplots()
for name, group in groups:
    run = run + 1
    ax1.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=4, label=name, color = colors[run-1]);
#plt.scatter(meta_data_countries[:,3].astype(float),abs((1-scen1_national_emissions[:,0]/scen1_national_emissions[:,1])*100)) #### % decrease in emissions vs. national carbon intensity 
#ax1.plot(group.x,group.y, linestyle='',marker=markers[run-1], ms=4, label=name, color = 'b') #### % decrease in emissions vs. national carbon intensity 
ax1.set_yscale('log')
ax1.set_ylim(0.1,100)
ax1.set_xlabel('GDPpc PPP 2019')
ax1.set_ylabel('% reductonne CO2e through luxury tax')
ax1.legend(frameon = False)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.annotate('USA',
            xy=(df3.iloc[87][0], df3.iloc[87][1]), xytext =(80000,50), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0))
ax1.annotate('Germany',
            xy=(df3.iloc[61][0], df3.iloc[61][1]), xytext =(50000,40), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0))
ax1.annotate('China',
            xy=(df3.iloc[11][0], df3.iloc[11][1]), xytext =(20000,30), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0))
ax1.annotate('India',
            xy=(df3.iloc[21][0], df3.iloc[21][1]), xytext =(5500,20), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.1, headwidth = 0))




######################################## GRAPH #4 ########################################################################################################
###################### USA percentile vs. emission reduction across both scenarios  #####################################


USA_per_cent_reduction_emissions_pc_SCEN2_2_mean = np.mean(USA_per_cent_reduction_emissions_pc_SCEN2_2,axis = 1)
USA_per_cent_reduction_emissions_pc_SCEN1_mean = np.mean(USA_per_cent_reduction_emissions_pc_SCEN1, axis = 1)

USA_per_cent_reduction_emissions_pc_SCEN2_2_SE = stats.sem(USA_per_cent_reduction_emissions_pc_SCEN2_2, axis = 1)
USA_per_cent_reduction_emissions_pc_SCEN1_SE = stats.sem(USA_per_cent_reduction_emissions_pc_SCEN1, axis = 1)

USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low = USA_per_cent_reduction_emissions_pc_SCEN2_2_mean - 2.57*USA_per_cent_reduction_emissions_pc_SCEN2_2_SE
USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high = USA_per_cent_reduction_emissions_pc_SCEN2_2_mean + 2.57*USA_per_cent_reduction_emissions_pc_SCEN2_2_SE

USA_per_cent_reduction_emissions_pc_SCEN1_99CI_low = USA_per_cent_reduction_emissions_pc_SCEN1_mean - 2.57*USA_per_cent_reduction_emissions_pc_SCEN1_SE
USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high = USA_per_cent_reduction_emissions_pc_SCEN1_mean + 2.57*USA_per_cent_reduction_emissions_pc_SCEN1_SE


percentiles = np.arange(1,101,1)

fig, ax1 = plt.subplots()
ax1.plot(percentiles, USA_pre_tax_carbon_cons_pc_total/1000, color = "black", label ="emissions per capita")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax1.set_yscale('log', basey = 2)
ax1.set_ylim(4,2**6)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.set_xlabel('percentile')
ax1.set_ylabel('CO2e tonnes/capita')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('% reduction in emissions per capita', color=color)  # we already handled the x-label with ax1
ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = color, label =  "reduction luxury tax")
ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_mean, color = color, label = "reduction uniform tax")
#ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, color = color)
#ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, color = color)

#ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, color = color)
#ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, color = color)
ax2.fill_between(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax2.fill_between(percentiles,USA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "red", alpha = 0.5)

ax2.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='best', bbox_to_anchor=(0.5, 0.15, 0.5, 0.5), frameon = False)
ax2.legend(loc='best', bbox_to_anchor=(0.5, 0, 0.5, 0.5), frameon = False)
ax1.annotate('USA',xy= (45,40), xytext =(45,40), fontsize = 16)



# https://matplotlib.org/2.2.5/gallery/api/two_scales.html










######################################## GRAPH #6 ########################################################################################################
###################### USA tax emission reduction composition per category #####################################

#USA_total_abated_carbon_SCEN1 = (sum(USA_pre_tax_carbon_cons_total.sum(axis = 1))-sum(USA_post_tax_carbon_cons_total_SCEN1.sum(axis = 1)))/(10**12) #### unit in gigatonne
#USA_total_abated_carbon_SCEN2_2 = (sum(USA_pre_tax_carbon_cons_total.sum(axis = 1))-sum(USA_post_tax_carbon_cons_total_SCEN2_2.sum(axis = 1)))/(10**12) #### unit in gigatonne


#USA_abated_carbon_per_category_SCEN1 = (USA_pre_tax_carbon_cons_total.sum(axis = 1) - USA_post_tax_carbon_cons_total_SCEN1.sum(axis = 1))/(10**12) #### gigatonne unit
#USA_abated_carbon_per_category_share_SCEN1 = USA_abated_carbon_per_category_SCEN1/USA_total_abated_carbon_SCEN1

#USA_abated_carbon_per_category_SCEN2_2 = (USA_pre_tax_carbon_cons_total.sum(axis = 1) - USA_post_tax_carbon_cons_total_SCEN2_2.sum(axis = 1))/(10**12) #### gigatonne unit
#USA_abated_carbon_per_category_share_SCEN2_2 = USA_abated_carbon_per_category_SCEN2_2/USA_total_abated_carbon_SCEN2_2




USA_frame1 = pd.DataFrame(np.mean(USA_abated_carbon_per_category_share_SCEN1_simruns,axis = 1)).transpose()
USA_frame2 = pd.DataFrame(np.mean(USA_abated_carbon_per_category_share_SCEN2_2_simruns,axis = 1)).transpose()


LABELS = ['uniform', 'luxury']
df_US_bar = pd.concat([USA_frame1, USA_frame2], ignore_index = True)

df_US_bar = df_US_bar.rename(index={df_US_bar.index[0]: 'uniform', df_US_bar.index[1]: ' luxury'})
df_US_bar = df_US_bar.rename(columns={df_US_bar.columns[0]: 'Food', df_US_bar.columns[1]: 'Alc&Tobac', df_US_bar.columns[2]: 'Wearables', df_US_bar.columns[3]: 'Housing', df_US_bar.columns[4]: 'Heat&Elect.', df_US_bar.columns[5]: 'Items&Services', df_US_bar.columns[6]: 'Health', df_US_bar.columns[7]: 'Vehicle Purchase', df_US_bar.columns[8]: 'Vehicle Fuel', df_US_bar.columns[9]: 'Other transport', df_US_bar.columns[10]: 'ICT', df_US_bar.columns[11]: 'Recreation', df_US_bar.columns[12]: 'Holiday', df_US_bar.columns[13]: 'Other luxury'})

#https://www.python-graph-gallery.com/5-control-width-and-space-in-barplots
ax1 = df_US_bar.plot.bar(stacked = True, legend = None, rot=0, figsize = (2,5), width = 3/4, title = "USA")
ax1.set_title("USA", fontsize= 20) # ti
#ax1.legend(bbox_to_anchor=(1.0, 1), frameon = False, title = "USA", title_fontsize=25)
ax1.set_ylabel("% abated emission composition")
ax1.set_yticklabels([0, 20, 40, 60, 80, 100])
#ax1.annotate('USA',xy= (1.8,1), xytext =(1.8,1), fontsize = 14)





######################################## GRAPH #6 ########################################################################################################
###################### CHINA percentile vs. emission reduction across both scenarios  #####################################




CHINA_per_cent_reduction_emissions_pc_SCEN2_2_mean = np.mean(CHINA_per_cent_reduction_emissions_pc_SCEN2_2,axis = 1)
CHINA_per_cent_reduction_emissions_pc_SCEN1_mean = np.mean(CHINA_per_cent_reduction_emissions_pc_SCEN1, axis = 1)

CHINA_per_cent_reduction_emissions_pc_SCEN2_2_SE = stats.sem(CHINA_per_cent_reduction_emissions_pc_SCEN2_2, axis = 1)
CHINA_per_cent_reduction_emissions_pc_SCEN1_SE = stats.sem(CHINA_per_cent_reduction_emissions_pc_SCEN1, axis = 1)

CHINA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low = CHINA_per_cent_reduction_emissions_pc_SCEN2_2_mean - 2.57*CHINA_per_cent_reduction_emissions_pc_SCEN2_2_SE
CHINA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high = CHINA_per_cent_reduction_emissions_pc_SCEN2_2_mean + 2.57*CHINA_per_cent_reduction_emissions_pc_SCEN2_2_SE

CHINA_per_cent_reduction_emissions_pc_SCEN1_99CI_low = CHINA_per_cent_reduction_emissions_pc_SCEN1_mean - 2.57*CHINA_per_cent_reduction_emissions_pc_SCEN1_SE
CHINA_per_cent_reduction_emissions_pc_SCEN1_99CI_high = CHINA_per_cent_reduction_emissions_pc_SCEN1_mean + 2.57*CHINA_per_cent_reduction_emissions_pc_SCEN1_SE




percentiles = np.arange(1,101,1)

fig, ax1 = plt.subplots()
ax1.plot(percentiles, CHINA_pre_tax_carbon_cons_pc_total/1000, color = "black", label = "emissions per capita")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax1.set_yscale('log', basey = 2)
ax1.set_ylim(2**-2,2**5)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.set_xlabel('percentile')
ax1.set_ylabel('CO2e tonnes/capita',)
ax1.tick_params(axis='y')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('% reduction in emissions per capita', color=color)  # we already handled the x-label with ax1
ax2.plot(percentiles, CHINA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = color, label = "reduction luxury tax")
ax2.plot(percentiles, CHINA_per_cent_reduction_emissions_pc_SCEN1_mean, color = color, label = "reduction uniform tax")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim((4,6))
ax2.set_yticks([4,4.5,5,5.5,6])
ax1.legend(bbox_to_anchor=(0.2, 0.0, 0.5, 0.5), frameon = False)
ax2.legend(bbox_to_anchor=(0.2, 0.05, 0.5, 0.5), frameon = False)
ax2.fill_between(percentiles, CHINA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, CHINA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax2.fill_between(percentiles,CHINA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, CHINA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "red", alpha = 0.5)
ax1.annotate('CHINA',xy= (45,20), xytext =(45,20), fontsize = 16)




######################################## GRAPH #7 ########################################################################################################



CHINA_frame1 = pd.DataFrame(np.mean(CHINA_abated_carbon_per_category_share_SCEN1_simruns,axis = 1)).transpose()
CHINA_frame2 = pd.DataFrame(np.mean(CHINA_abated_carbon_per_category_share_SCEN2_2_simruns,axis = 1)).transpose()

LABELS = ['uniform', 'luxury']
df_CHINA_bar = pd.concat([CHINA_frame1, CHINA_frame2], ignore_index = True)
df_CHINA_bar = df_CHINA_bar.rename(index={df_CHINA_bar.index[0]: 'uniform', df_CHINA_bar.index[1]: ' luxury'})
df_CHINA_bar = df_CHINA_bar.rename(columns={df_CHINA_bar.columns[0]: 'Food', df_CHINA_bar.columns[1]: 'Alc&Tobac', df_CHINA_bar.columns[2]: 'Wearables', df_CHINA_bar.columns[3]: 'Housing', df_CHINA_bar.columns[4]: 'Heat&Elect.', df_CHINA_bar.columns[5]: 'Items&Services', df_CHINA_bar.columns[6]: 'Health', df_CHINA_bar.columns[7]: 'Vehicle Purchase', df_CHINA_bar.columns[8]: 'Vehicle Fuel', df_CHINA_bar.columns[9]: 'Other transport', df_CHINA_bar.columns[10]: 'ICT', df_CHINA_bar.columns[11]: 'Recreation', df_CHINA_bar.columns[12]: 'Holiday', df_CHINA_bar.columns[13]: 'Other luxury'})

#https://www.python-graph-gallery.com/5-control-width-and-space-in-barplots
ax1 = df_CHINA_bar.plot.bar(stacked = True, legend = None, rot=0, figsize = (2,5), width = 3/4, title = "CHINA")
ax1.set_title("CHINA", fontsize= 20) # ti
#ax1.legend(bbox_to_anchor=(1.0, 1), frameon = False, title = "USA", title_fontsize=25)
ax1.set_ylabel("% abated emission composition")
ax1.set_yticklabels([0, 20, 40, 60, 80, 100])



######################################## GRAPH #8 ########################################################################################################
###################### SOUTH AFRICA percentile vs. emission reduction across both scenarios  #####################################




SA_per_cent_reduction_emissions_pc_SCEN2_2_mean = np.mean(SA_per_cent_reduction_emissions_pc_SCEN2_2,axis = 1)
SA_per_cent_reduction_emissions_pc_SCEN1_mean = np.mean(SA_per_cent_reduction_emissions_pc_SCEN1, axis = 1)

SA_per_cent_reduction_emissions_pc_SCEN2_2_SE = stats.sem(SA_per_cent_reduction_emissions_pc_SCEN2_2, axis = 1)
SA_per_cent_reduction_emissions_pc_SCEN1_SE = stats.sem(SA_per_cent_reduction_emissions_pc_SCEN1, axis = 1)

SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low = SA_per_cent_reduction_emissions_pc_SCEN2_2_mean - 2.57*SA_per_cent_reduction_emissions_pc_SCEN2_2_SE
SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high = SA_per_cent_reduction_emissions_pc_SCEN2_2_mean + 2.57*SA_per_cent_reduction_emissions_pc_SCEN2_2_SE

SA_per_cent_reduction_emissions_pc_SCEN1_99CI_low = SA_per_cent_reduction_emissions_pc_SCEN1_mean - 2.57*SA_per_cent_reduction_emissions_pc_SCEN1_SE
SA_per_cent_reduction_emissions_pc_SCEN1_99CI_high = SA_per_cent_reduction_emissions_pc_SCEN1_mean + 2.57*SA_per_cent_reduction_emissions_pc_SCEN1_SE




percentiles = np.arange(1,101,1)

fig, ax1 = plt.subplots()
ax1.plot(percentiles, SA_pre_tax_carbon_cons_pc_total/1000, color = "black", label = "emissions per capita")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax1.set_yscale('log', basey = 2)
ax1.set_ylim(2**-2,2**6)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.set_xlabel('percentile')
ax1.set_ylabel('CO2e tonnes/capita',)
ax1.tick_params(axis='y')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('% reduction in emissions per capita', color=color)  # we already handled the x-label with ax1
ax2.plot(percentiles, SA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = color, label = "reduction luxury tax")
ax2.plot(percentiles, SA_per_cent_reduction_emissions_pc_SCEN1_mean, color = color, label = "reduction uniform tax")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim((3,10))
ax2.set_yticks([4,4.5,5,5.5,6])
ax1.legend(bbox_to_anchor=(0.2, 0.0, 0.5, 0.5), frameon = False)
ax2.legend(bbox_to_anchor=(0.2, 0.05, 0.5, 0.5), frameon = False)
ax2.fill_between(percentiles, SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax2.fill_between(percentiles,SA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, SA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "red", alpha = 0.5)
ax1.annotate('South Africa',xy= (45,20), xytext =(45,20), fontsize = 16)




######################################## GRAPH #9 SA ########################################################################################################


SA_frame1 = pd.DataFrame(np.mean(SA_abated_carbon_per_category_share_SCEN1_simruns,axis = 1)).transpose()
SA_frame2 = pd.DataFrame(np.mean(SA_abated_carbon_per_category_share_SCEN2_2_simruns,axis = 1)).transpose()


LABELS = ['uniform', 'luxury']
df_SA_bar = pd.concat([SA_frame1, SA_frame2], ignore_index = True)

df_SA_bar = df_SA_bar.rename(index={df_SA_bar.index[0]: 'uniform', df_SA_bar.index[1]: ' luxury'})
df_SA_bar = df_SA_bar.rename(columns={df_SA_bar.columns[0]: 'Food', df_SA_bar.columns[1]: 'Alc&Tobac', df_SA_bar.columns[2]: 'Wearables', df_SA_bar.columns[3]: 'Housing', df_SA_bar.columns[4]: 'Heat&Elect.', df_SA_bar.columns[5]: 'Items&Services', df_SA_bar.columns[6]: 'Health', df_SA_bar.columns[7]: 'Vehicle Purchase', df_SA_bar.columns[8]: 'Vehicle Fuel', df_SA_bar.columns[9]: 'Other transport', df_SA_bar.columns[10]: 'ICT', df_SA_bar.columns[11]: 'Recreation', df_SA_bar.columns[12]: 'Holiday', df_SA_bar.columns[13]: 'Other luxury'})

#https://www.python-graph-gallery.com/5-control-width-and-space-in-barplots
ax1 = df_SA_bar.plot.bar(stacked = True, legend = None, rot=0, figsize = (2,5), width = 3/4, title = "SA")
ax1.set_title("SA", fontsize= 20) # ti
#ax1.legend(bbox_to_anchor=(1.0, 1), frameon = False, title = "USA", title_fontsize=25)
ax1.set_ylabel("% abated emission composition")
ax1.set_yticklabels([0, 20, 40, 60, 80, 100])











####### GRAPH # 8 #####################################################
##### GLOBAL changes in expenditure per category 


sum1 = np.zeros((1,14));
sum2 = np.zeros((1,14));
sum3 = np.zeros((1,14));

for i in range(0,14):
            sum1[:,i] = sum(np.asarray(df_BIG_exp_2019_total.sum(axis = 1))[i::14])
            sum2[:,i] = sum(np.mean(scen1_total_exp, axis = 1)[i::14])
            sum3[:,i] = sum(np.mean(scen2_2_total_exp, axis = 1)[i::14])


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  NEW SECTION!!!!! ###############################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


#####################################################################################################################################
#####################################################################################################################################
                              ############## PLOT PROGRESSIVIY FOR ALL COUNTRIES ##################   
#####################################################################################################################################
#####################################################################################################################################


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  NEW SECTION !!!!! ###############################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################





fig, ax1 = plt.subplots()
ax1.scatter(np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN1_simruns, axis = 1),  np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN2_2_simruns, axis = 1), s = 7);
ax1.plot([0,0], [-25,10], color = "black", linestyle = "--")
ax1.plot([-8,5], [0,0], color = "black", linestyle = "--")
ax1.plot([-25,5], [-25,5], color = "black", label = " x = y")
ax1.set_xlabel('1st - 100th uniform tax')
ax1.set_ylabel('1st - 100th percentile Luxury tax')
ax1.set_xlim(-8,5);
ax1.margins(x=0,y=0);
ax1.annotate('progressive before',xy= (-6,7), xytext =(-6,7), fontsize = 8)
ax1.annotate('regressive after',xy= (-6,5), xytext =(-6,5), fontsize = 8)
ax1.annotate('progressive before',xy= (-6,-20), xytext =(-6,-20), fontsize = 8)
ax1.annotate('progressive after',xy= (-6,-22), xytext =(-6,-22), fontsize = 8)
ax1.annotate('regressive before',xy= (1,-20), xytext =(1,-20), fontsize = 8)
ax1.annotate('progressive after',xy= (1,-22), xytext =(1,-22), fontsize = 8)
ax1.annotate('regressive before',xy= (1,7), xytext =(1,7), fontsize = 8)
ax1.annotate('regressive after',xy= (1,5), xytext =(1,5), fontsize = 8)
ax1.legend(loc = "center right", frameon = False)






df4 = pd.DataFrame(dict(x = np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN1_simruns, axis = 1) , y = np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN2_2_simruns, axis = 1), label=meta_data_countries[:,6]))
groups4 = df4.groupby('label')
markers = ['*', '^', "+", ".", "o"]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
run = 0;
fig, ax1 = plt.subplots()
for name, group in groups4:
    run = run + 1
    ax1.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=4, label=name, color = colors[run-1]);   
ax1.plot([0,0], [-25,10], color = "black", linestyle = "--")
ax1.plot([-8,5], [0,0], color = "black", linestyle = "--")
ax1.plot([-25,5], [-25,5], color = "black", label = " x = y")
ax1.set_xlabel('1st - 100th uniform tax')
ax1.set_ylabel('1st - 100th percentile Luxury tax')
ax1.set_xlim(-8,5);
ax1.margins(x=0,y=0);
ax1.annotate('progressive uniform',xy= (-5,7), xytext =(-5,7), fontsize = 8)
ax1.annotate('regressive luxury',xy= (-6,5), xytext =(-5,5), fontsize = 8)
ax1.annotate('progressive uniform',xy= (-5,-20), xytext =(-5,-20), fontsize = 8)
ax1.annotate('progressive luxury',xy= (-5,-22), xytext =(-5,-22), fontsize = 8)
ax1.annotate('regressive uniform',xy= (1,-20), xytext =(1,-20), fontsize = 8)
ax1.annotate('progressive luxury',xy= (1,-22), xytext =(1,-22), fontsize = 8)
ax1.annotate('regressive uniform',xy= (1,7), xytext =(1,7), fontsize = 8)
ax1.annotate('regressive luxury',xy= (1,5), xytext =(1,5), fontsize = 8)
ax1.legend(loc='best', bbox_to_anchor=(0.5, 0.15, 0.5, 0.5), frameon = False, fontsize = 8)   
    
    



#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  FIGURE #3  #######################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

#####################################################################################################################################
       #####################################################################################################################################
                       ############## JOINT PLOT #3  ##################   
#####################################################################################################################################
#####################################################################################################################################

#### https://matplotlib.org/stable/tutorials/intermediate/constrainedlayout_guide.html
################## JOINT PLOT #3 #######################



#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  FIGURE #3  #######################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################





fig = plt.figure(figsize=(13,7.5))
outer = gridspec.GridSpec(2,2 ,figure=fig, height_ratios = [2, 2]) 
#make nested gridspecs
gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0,0])
gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0,1], wspace = .05)
gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[1,0], wspace = .05)
gs4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec = outer[1,1], wspace = .05)

ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs2[0])
ax3 = fig.add_subplot(gs3[0])
ax4 = fig.add_subplot(gs3[1])
ax5 = fig.add_subplot(gs4[0])
ax6 = fig.add_subplot(gs4[1])



percentiles = np.arange(1,101,1)

GDP_national_total = meta_data_countries[:,3].astype(float)*meta_data_countries[:,5].astype(float);
scen1_revenue_as_percent_of_GDP = scen1_average_national_tax_revenue/GDP_national_total * 100;
scen2_1_revenue_as_percent_of_GDP = scen2_1_average_national_tax_revenue/GDP_national_total * 100;
scen2_2_revenue_as_percent_of_GDP = scen2_2_average_national_tax_revenue/GDP_national_total * 100;

scen1_average_reduced_emissions_as_percent_total = (1-np.divide(np.squeeze(scen1_average_reduced_emissions),np.squeeze(array_BIG_total_emissions_per_country)))*100
scen2_1_average_reduced_emissions_as_percent_total = (1-np.divide(np.squeeze(scen2_1_average_reduced_emissions),np.squeeze(array_BIG_total_emissions_per_country)))*100
scen2_2_average_reduced_emissions_as_percent_total = (1-np.divide(np.squeeze(scen2_2_average_reduced_emissions),np.squeeze(array_BIG_total_emissions_per_country)))*100




df = pd.DataFrame(dict(x = scen1_revenue_as_percent_of_GDP, y = scen1_average_reduced_emissions_as_percent_total, label=meta_data_countries[:,1]))
df2 = pd.DataFrame(dict(x = scen2_2_revenue_as_percent_of_GDP, y = scen2_2_average_reduced_emissions_as_percent_total, label=meta_data_countries[:,1]))
groups = df.groupby('label')
groups2 = df2.groupby('label')
#colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
#colors1 = plt.cm.get_cmap('Blues') # colors1(1-1/run/3)
#colors2 = plt.cm.get_cmap('Reds')
markers = ['.', '^', "+", "*"]

run = 0 
run2 = 0
for name, group in groups:
    run = run + 1
    ax1.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=4, label=name, color = 'b');
for name, group in groups2:
    run2 = run2 + 1
    ax1.plot(group.x, group.y, marker=markers[run2-1], linestyle='', ms=4, label=name, color = 'r');
ax1.set_yscale('log' ,basey = 2)
ax1.set_xscale('log', basex = 2)
ax1.set_xlim(1/16,8)
ax1.set_ylim(1/8,32)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.set_xlabel('Tax revenue as % of GDP')
ax1.set_ylabel('National emissions reduction %')
for i in range(0, len(scen1_revenue_as_percent_of_GDP)):
    ax1.plot([scen1_revenue_as_percent_of_GDP[i], scen2_2_revenue_as_percent_of_GDP[i]], [scen1_average_reduced_emissions_as_percent_total[i], scen2_2_average_reduced_emissions_as_percent_total[i]], color = 'black', linestyle = '-', linewidth = 1/2)
ax1.annotate("USA",(scen1_revenue_as_percent_of_GDP[87], scen1_average_reduced_emissions_as_percent_total[87]), (5,6), arrowprops =dict(arrowstyle ="-"), fontsize = 10)
ax1.annotate("CHINA",(scen1_revenue_as_percent_of_GDP[11], scen1_average_reduced_emissions_as_percent_total[11]),(0.7,18), arrowprops =dict(arrowstyle ="-"), fontsize = 10)
#https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
#https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph
#### https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
legend_elements2 = [Line2D([0], [0], marker ='^', color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='Low income $10/t'),
                    Line2D([0], [0], marker = '+', color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='Lower middle income $25/t'),
                    Line2D([0], [0], marker ='*',  color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='Upper middle income $50/t'),
                    Line2D([0], [0], marker ='.',  color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='High income $150/t'),
                      ]
ax1.legend(handles=legend_elements2, loc='lower right', title = "Symbol", fontsize = 8, frameon = False)
ax1.plot([0.1, 0.125], [16, 16],color = 'red', linestyle = '-', linewidth = 4);
ax1.plot([0.1, 0.125], [12, 12],color = 'blue', linestyle = '-', linewidth = 4);
ax1.annotate("uniform tax", (0.14, 11), (0.14, 11));
ax1.annotate("Luxury tax", (0.14, 15), (0.14, 15));
ax1.annotate("Color", (0.12, 22), (0.12, 22));
ax1.annotate('a', xy=(ax1.get_xlim()[0],ax1.get_ylim()[1]+4),fontsize=16 ,annotation_clip=False)




df4 = pd.DataFrame(dict(x = np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN1_simruns,axis=1) , y = np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN2_2_simruns, axis =1), label=meta_data_countries[:,6]))
groups4 = df4.groupby('label')
markers = ['*', '^', "+", ".", "o"]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
run = 0;

for name, group in groups4:
    run = run + 1
    ax2.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=4, label=name, color = colors[run-1]);   
ax2.plot([0,0], [-25,10], color = "black", linestyle = "--")
ax2.plot([-8,5], [0,0], color = "black", linestyle = "--")
ax2.plot([-25,5], [-25,5], color = "black", label = " x = y")
ax2.set_xlabel('1st - 100th uniform tax')
ax2.set_ylabel('1st - 100th percentile Luxury tax')
ax2.set_xlim(-8,5);
ax2.margins(x=0,y=0);
ax2.annotate('progressive uniform',xy= (-5,7), xytext =(-5,7), fontsize = 8)
ax2.annotate('regressive luxury',xy= (-6,5), xytext =(-5,5), fontsize = 8)
ax2.annotate('progressive uniform',xy= (-5,-20), xytext =(-5,-20), fontsize = 8)
ax2.annotate('progressive luxury',xy= (-5,-22), xytext =(-5,-22), fontsize = 8)
ax2.annotate('regressive uniform',xy= (1,-20), xytext =(1,-20), fontsize = 8)
ax2.annotate('progressive luxury',xy= (1,-22), xytext =(1,-22), fontsize = 8)
ax2.annotate('regressive uniform',xy= (1,7), xytext =(1,7), fontsize = 8)
ax2.annotate('regressive luxury',xy= (1,5), xytext =(1,5), fontsize = 8)
ax2.legend(loc='best', bbox_to_anchor=(0.5, 0.16, 0.5, 0.5), frameon = False, fontsize = 8)   
ax2.annotate('b', xy=(ax2.get_xlim()[0],ax2.get_ylim()[1]+1/2),fontsize=16 ,annotation_clip=False)
ax2.annotate('USA',xy= (df4.iloc[87,0],df4.iloc[87,1]), xytext =(0.2,-10), fontsize = 10, arrowprops =dict(arrowstyle ="-"))
ax2.annotate('CHINA',xy= (df4.iloc[11,0],df4.iloc[11,1]), xytext =(3,1), fontsize = 10, arrowprops =dict(arrowstyle ="-"))


ax3.plot(percentiles, USA_pre_tax_carbon_cons_pc_total/1000, color = "black", label ="emissions per capita")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax3.set_yscale('log', basey = 2)
ax3.set_ylim(0.25,2**6)
for axis in [ax3.xaxis, ax3.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax3.set_xlabel('percentile', fontsize = 8)
ax3.set_ylabel('CO2e tonnes/capita', fontsize = 8)
ax3.tick_params(axis='y')
ax7 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax7.set_ylabel('% reduction in emissions per capita', color=color, fontsize = 8)  # we already handled the x-label with ax1
ax7.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = color, label =  "reduction luxury tax")
ax7.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_mean, color = "tab:blue", label = "reduction uniform tax")
ax7.tick_params(axis='y', labelcolor=color)
ax3.legend(loc='best', bbox_to_anchor=(0.5, -.1, 0.5, 0.5), frameon = False, fontsize = 12)
ax7.legend(loc='best', bbox_to_anchor=(0.5, -0.2, 0.5, 0.5), frameon = False, fontsize = 12)
ax3.annotate('USA',xy= (45,40), xytext =(45,40), fontsize = 16)
#ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, color = color)
ax7.fill_between(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax7.fill_between(percentiles,USA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "tab:blue", alpha = 0.5)
ax7.set_ylim((4,10))
ax7.axis('off')
ax3.annotate('c', xy=(ax3.get_xlim()[0],ax3.get_ylim()[1]+8),fontsize=16 ,annotation_clip=False)

ax4.plot(percentiles, CHINA_pre_tax_carbon_cons_pc_total/1000, color = "black", label = "emissions per capita")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax4.set_yscale('log', basey = 2)
ax4.set_ylim(2**-2,2**6)
for axis in [ax4.xaxis, ax4.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax4.set_xlabel('percentile', fontsize = 8)
ax4.set_ylabel('CO2e tonnes/capita', fontsize = 8)
ax4.tick_params(axis='y')
ax8 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
color = 'black'
ax8.set_ylabel('% reduction in emissions per capita', color=color, fontsize = 8)  # we already handled the x-label with ax1
ax8.plot(percentiles, CHINA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = "tab:red", label = "reduction luxury tax")
ax8.plot(percentiles, CHINA_per_cent_reduction_emissions_pc_SCEN1_mean, color = "tab:blue", label = "reduction uniform tax")
ax8.tick_params(axis='y', labelcolor=color)
#ax4.legend(bbox_to_anchor=(0.2, 0.01, 0.5, 0.5), frameon = False)
#ax8.legend(bbox_to_anchor=(0.2, 0.08, 0.5, 0.5), frameon = False)
ax4.annotate('CHINA',xy= (45,40), xytext =(45,40), fontsize = 16)
ax8.fill_between(percentiles, CHINA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, CHINA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax8.fill_between(percentiles,CHINA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, CHINA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "tab:blue", alpha = 0.5)
ax8.set_yticks([4,5,6,7,8,9,10])
ax8.set_ylim((4,10))
ax4.get_yaxis().set_visible(False)



cm1 = plt.cm.get_cmap('summer')
cm2 = plt.cm.get_cmap('spring')
cm3 = plt.cm.get_cmap('autumn')
cm4 = plt.cm.get_cmap('winter')

cmap_bar =  [  cm1(0.8), cm1(0.6), cm1(0.4), cm1(0.2), cm1(0.05), cm2(0.2),
            cm2(0.4), cm3(0.2), cm3(0.4), cm3(0.6), cm4(0.2), cm4(0.4),
            cm4(0.6), cm4(0.8)]
x1 = 1 
x2 = 2
cum_sum_1 = 0 
cum_sum_2 = 0
for i in range(0,14):
     if i == 0:
             cum_sum_1 = cum_sum_1 + 0
             cum_sum_2 = cum_sum_2 + 0
     else:
             cum_sum_1 = cum_sum_1 + df_US_bar.iloc[0][i-1]*100
             cum_sum_2 = cum_sum_2 + df_US_bar.iloc[1][i-1]*100
     ax5.bar(x1, df_US_bar.iloc[0][i]*100, width=0.8, bottom = cum_sum_1, color = cmap_bar[i])
     ax5.bar(x2, df_US_bar.iloc[1][i]*100, width=0.8, bottom = cum_sum_2, color = cmap_bar[i])
     ax5.set_xticks([1,2])
     ax5.set_xticklabels(["uniform", "luxury"], fontsize = 8)
     ax5.set_title('USA abatement')
     ax5.yaxis.set_major_formatter(mticker.PercentFormatter())
ax5.annotate('d', xy=(ax5.get_xlim()[0]-0.1,ax5.get_ylim()[1]+5),fontsize=16 ,annotation_clip=False)
     
     

x3 = 1 
x4 = 2
cum_sum_3 = 0 
cum_sum_4 = 0
for i in range(0,14):
     if i == 0:
             cum_sum_3 = cum_sum_3 + 0
             cum_sum_4 = cum_sum_4 + 0
     else:
             cum_sum_3 = cum_sum_3 + df_CHINA_bar.iloc[0][i-1]*100
             cum_sum_4 = cum_sum_4 + df_CHINA_bar.iloc[1][i-1]*100
     ax6.bar(x1, df_CHINA_bar.iloc[0][i]*100, width=0.8, bottom = cum_sum_3, color = cmap_bar[i], label = df_CHINA_bar.columns[i])
     ax6.bar(x2, df_CHINA_bar.iloc[1][i]*100, width=0.8, bottom = cum_sum_4, color = cmap_bar[i])
     ax6.set_xticks([1,2])
     ax6.set_xticklabels(["uniform", "luxury"], fontsize = 8)
     ax6.set_title('CHINA abatement')
     ax6.legend(bbox_to_anchor=(1.05, 1), frameon = False)
     ax6.yaxis.set_major_formatter(mticker.PercentFormatter())
     ax6.get_yaxis().set_visible(False)

handles, labels = ax6.get_legend_handles_labels()
ax6.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), frameon = False)          
     

plt.tight_layout()
plt.savefig('fig3.png',bbox_inches = "tight", dpi = 300);
plt.show()




#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  FIGURE #3 END #######################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  FIGURE #3 WITH SA #######################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################






fig = plt.figure(figsize=(13,7.5))
outer = gridspec.GridSpec(2,2 ,figure=fig, height_ratios = [2, 2]) 
#make nested gridspecs
gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0,0])
gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0,1], wspace = .05)
gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[1,0], wspace = .05)
gs4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec = outer[1,1], wspace = .05)

ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs2[0])
ax3 = fig.add_subplot(gs3[0])
ax4 = fig.add_subplot(gs3[1])
ax5 = fig.add_subplot(gs4[0])
ax6 = fig.add_subplot(gs4[1])


percentiles = np.arange(1,101,1)

GDP_national_total = meta_data_countries[:,3].astype(float)*meta_data_countries[:,5].astype(float);
scen1_revenue_as_percent_of_GDP = scen1_average_national_tax_revenue/GDP_national_total * 100;
scen2_1_revenue_as_percent_of_GDP = scen2_1_average_national_tax_revenue/GDP_national_total * 100;
scen2_2_revenue_as_percent_of_GDP = scen2_2_average_national_tax_revenue/GDP_national_total * 100;

scen1_average_reduced_emissions_as_percent_total = (1-np.divide(np.squeeze(scen1_average_reduced_emissions),np.squeeze(array_BIG_total_emissions_per_country)))*100
scen2_1_average_reduced_emissions_as_percent_total = (1-np.divide(np.squeeze(scen2_1_average_reduced_emissions),np.squeeze(array_BIG_total_emissions_per_country)))*100
scen2_2_average_reduced_emissions_as_percent_total = (1-np.divide(np.squeeze(scen2_2_average_reduced_emissions),np.squeeze(array_BIG_total_emissions_per_country)))*100


#np.mean(scen1_revenue_as_percent_of_GDP)
#np.mean(scen2_2_revenue_as_percent_of_GDP)
#np.mean(scen1_average_reduced_emissions_as_percent_total)
#np.mean(scen2_2_average_reduced_emissions_as_percent_total)

#np.median(scen1_revenue_as_percent_of_GDP)
#np.median(scen2_2_revenue_as_percent_of_GDP)

df = pd.DataFrame(dict(x = scen1_revenue_as_percent_of_GDP, y = scen1_average_reduced_emissions_as_percent_total, label=meta_data_countries[:,1]))
df2 = pd.DataFrame(dict(x = scen2_2_revenue_as_percent_of_GDP, y = scen2_2_average_reduced_emissions_as_percent_total, label=meta_data_countries[:,1]))
groups = df.groupby('label')
groups2 = df2.groupby('label')
#colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
#colors1 = plt.cm.get_cmap('Blues') # colors1(1-1/run/3)
#colors2 = plt.cm.get_cmap('Reds')
markers = ['.', '^', "+", "*"]

run = 0 
run2 = 0
for name, group in groups:
    run = run + 1
    ax1.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=4, label=name, color = 'b');
for name, group in groups2:
    run2 = run2 + 1
    ax1.plot(group.x, group.y, marker=markers[run2-1], linestyle='', ms=4, label=name, color = 'r');
ax1.set_yscale('log' ,basey = 2)
ax1.set_xscale('log', basex = 2)
ax1.set_xlim(1/16,8)
ax1.set_ylim(1/8,32)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.set_xlabel('Tax revenue as % of GDP')
ax1.set_ylabel('National emissions reduction %')
for i in range(0, len(scen1_revenue_as_percent_of_GDP)):
    ax1.plot([scen1_revenue_as_percent_of_GDP[i], scen2_2_revenue_as_percent_of_GDP[i]], [scen1_average_reduced_emissions_as_percent_total[i], scen2_2_average_reduced_emissions_as_percent_total[i]], color = 'black', linestyle = '-', linewidth = 1/2)
ax1.annotate("USA",(scen1_revenue_as_percent_of_GDP[87], scen1_average_reduced_emissions_as_percent_total[87]), (5,6), arrowprops =dict(arrowstyle ="-"), fontsize = 10)
ax1.annotate("SA",(scen1_revenue_as_percent_of_GDP[47], scen1_average_reduced_emissions_as_percent_total[47]),(0.7,18), arrowprops =dict(arrowstyle ="-"), fontsize = 10)
#https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
#https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph
#### https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
legend_elements2 = [Line2D([0], [0], marker ='^', color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='Low income $10/t'),
                    Line2D([0], [0], marker = '+', color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='Lower middle income $25/t'),
                    Line2D([0], [0], marker ='*',  color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='Upper middle income $50/t'),
                    Line2D([0], [0], marker ='.',  color = 'w', markerfacecolor='w',markeredgecolor = 'black', markersize=8, label='High income $150/t'),
                      ]
ax1.legend(handles=legend_elements2, loc='lower right', title = "Symbol", fontsize = 8, frameon = False)
ax1.plot([0.1, 0.125], [16, 16],color = 'red', linestyle = '-', linewidth = 4);
ax1.plot([0.1, 0.125], [12, 12],color = 'blue', linestyle = '-', linewidth = 4);
ax1.annotate("uniform tax", (0.14, 11), (0.14, 11));
ax1.annotate("luxury tax", (0.14, 15), (0.14, 15));
ax1.annotate("Color", (0.12, 22), (0.12, 22));
ax1.annotate('a', xy=(ax1.get_xlim()[0],ax1.get_ylim()[1]+4),fontsize=16 ,annotation_clip=False)




df4 = pd.DataFrame(dict(x = np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN1_simruns,axis=1) , y = np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN2_2_simruns, axis =1), label=meta_data_countries[:,6]))
groups4 = df4.groupby('label')
markers = ['*', '^', "+", ".", "o"]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
run = 0;

for name, group in groups4:
    run = run + 1
    ax2.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=4, label=name, color = colors[run-1]);   
ax2.plot([0,0], [-25,10], color = "black", linestyle = "--")
ax2.plot([-8,5], [0,0], color = "black", linestyle = "--")
ax2.plot([-25,5], [-25,5], color = "black", label = " x = y")
ax2.set_xlabel(r'1st $-$ 100th percentile uniform tax')
ax2.set_ylabel(r'1st $-$ 100th percentile luxury tax')
ax2.set_xlim(-7,5);
ax2.set_ylim(-18,1);
ax2.margins(x=0,y=0);
#ax2.annotate('progressive uniform',xy= (-5,7), xytext =(-5,7), fontsize = 9)
#ax2.annotate('regressive luxury',xy= (-6,5), xytext =(-5,5), fontsize = 9)
ax2.annotate('progressive uniform',xy= (-6,-16), xytext =(-6,-16), fontsize = 9)
ax2.annotate('progressive luxury',xy= (-6,-17), xytext =(-6,-17), fontsize = 9)
ax2.annotate('regressive uniform',xy= (1,-16), xytext =(1,-16), fontsize = 9)
ax2.annotate('progressive luxury',xy= (1,-17), xytext =(1,-17), fontsize = 9)
#ax2.annotate('regressive uniform',xy= (1,7), xytext =(1,7), fontsize = 9)
#ax2.annotate('regressive luxury',xy= (1,5), xytext =(1,5), fontsize = 9)
ax2.legend(loc='best', bbox_to_anchor=(0.5, 0.16, 0.5, 0.5), frameon = False, fontsize = 8)   
ax2.annotate('c', xy=(ax2.get_xlim()[0],ax2.get_ylim()[1]+1/2),fontsize=16 ,annotation_clip=False)
ax2.annotate('USA',xy= (df4.iloc[87,0],df4.iloc[87,1]), xytext =(2,-5), fontsize = 10, arrowprops =dict(arrowstyle ="-"))
ax2.annotate('SA',xy= (df4.iloc[47,0],df4.iloc[47,1]), xytext =(-4,-5), fontsize = 10, arrowprops =dict(arrowstyle ="-"))

ax2.annotate('GRC',xy= (df4.iloc[67,0],df4.iloc[67,1]), xytext =(-2,-15), fontsize = 10, arrowprops =dict(arrowstyle ="-"))
ax2.annotate('CZE',xy= (df4.iloc[60,0],df4.iloc[60,1]), xytext =(1,-10), fontsize = 10, arrowprops =dict(arrowstyle ="-"))
ax2.annotate('UK',xy= (df4.iloc[85,0],df4.iloc[85,1]), xytext =(1.4,-6), fontsize = 10, arrowprops =dict(arrowstyle ="-"))
ax2.annotate('DEU',xy= (df4.iloc[61,0],df4.iloc[61,1]), xytext =(1.8,-3), fontsize = 10, arrowprops =dict(arrowstyle ="-"))
ax2.annotate('EST',xy= (df4.iloc[63,0],df4.iloc[63,1]), xytext =(4,-5), fontsize = 10, arrowprops =dict(arrowstyle ="-"))
ax2.annotate('ITA',xy= (df4.iloc[71,0],df4.iloc[71,1]), xytext =(-1,-12), fontsize = 10, arrowprops =dict(arrowstyle ="-"))
ax2.annotate('HUN',xy= (df4.iloc[69,0],df4.iloc[69,1]), xytext =(-3,-13), fontsize = 10, arrowprops =dict(arrowstyle ="-"))

ax2.annotate('BOL',xy= (df4.iloc[6,0],df4.iloc[6,1]), xytext =(-6.5,-14.5), fontsize = 10, arrowprops =dict(arrowstyle ="-"))
ax2.annotate('NAM',xy= (df4.iloc[36,0],df4.iloc[36,1]), xytext =(-5,-14.5), fontsize = 10, arrowprops =dict(arrowstyle ="-"))




ax3.plot(percentiles, USA_pre_tax_carbon_cons_pc_total/1000, color = "black", label ="emissions per capita")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax3.set_yscale('log', basey = 2)
ax3.set_ylim(0.25,2**6)
for axis in [ax3.xaxis, ax3.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax3.set_xlabel('percentile', fontsize = 8)
ax3.set_ylabel('CO2e tonnes/capita', fontsize = 8)
ax3.tick_params(axis='y')
ax7 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax7.set_ylabel('% reduction in emissions per capita', color=color, fontsize = 8)  # we already handled the x-label with ax1
ax7.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = color, label =  "reduction luxury tax")
ax7.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_mean, color = "tab:blue", label = "reduction uniform tax")
ax7.tick_params(axis='y', labelcolor=color)
ax3.legend(loc='best', bbox_to_anchor=(0.5, -.1, 0.5, 0.5), frameon = False, fontsize = 12)
ax7.legend(loc='best', bbox_to_anchor=(0.5, -0.2, 0.5, 0.5), frameon = False, fontsize = 12)
ax3.annotate('USA',xy= (45,40), xytext =(45,40), fontsize = 16)
#ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, color = color)
ax7.fill_between(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax7.fill_between(percentiles,USA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "tab:blue", alpha = 0.5)
ax7.set_ylim((3,10))
ax7.axis('off')
ax3.annotate('b', xy=(ax3.get_xlim()[0],ax3.get_ylim()[1]+8),fontsize=16 ,annotation_clip=False)

ax4.plot(percentiles, SA_pre_tax_carbon_cons_pc_total/1000, color = "black", label = "emissions per capita")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax4.set_yscale('log', basey = 2)
ax4.set_ylim(2**-2,2**6)
for axis in [ax4.xaxis, ax4.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax4.set_xlabel('percentile', fontsize = 8)
ax4.set_ylabel('CO2e tonnes/capita', fontsize = 8)
ax4.tick_params(axis='y')
ax8 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
color = 'black'
ax8.set_ylabel('% reduction in emissions per capita', color=color, fontsize = 8)  # we already handled the x-label with ax1
ax8.plot(percentiles, SA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = "tab:red", label = "reduction luxury tax")
ax8.plot(percentiles, SA_per_cent_reduction_emissions_pc_SCEN1_mean, color = "tab:blue", label = "reduction uniform tax")
ax8.tick_params(axis='y', labelcolor=color)
#ax4.legend(bbox_to_anchor=(0.2, 0.01, 0.5, 0.5), frameon = False)
#ax8.legend(bbox_to_anchor=(0.2, 0.08, 0.5, 0.5), frameon = False)
ax4.annotate('South Africa',xy= (25,40), xytext =(25,40), fontsize = 16)
ax8.fill_between(percentiles, SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax8.fill_between(percentiles,SA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, SA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "tab:blue", alpha = 0.5)
ax8.set_yticks([3,4,5,6,7,8,9,10])
ax8.set_ylim((3,10))
ax4.get_yaxis().set_visible(False)



cm1 = plt.cm.get_cmap('summer')
cm2 = plt.cm.get_cmap('spring')
cm3 = plt.cm.get_cmap('autumn')
cm4 = plt.cm.get_cmap('winter')

cmap_bar =  [  cm1(0.8), cm1(0.6), cm1(0.4), cm1(0.2), cm1(0.05), cm2(0.2),
            cm2(0.4), cm3(0.2), cm3(0.4), cm3(0.6), cm4(0.2), cm4(0.4),
            cm4(0.6), cm4(0.8)]
x1 = 1 
x2 = 2
cum_sum_1 = 0 
cum_sum_2 = 0
for i in range(0,14):
     if i == 0:
             cum_sum_1 = cum_sum_1 + 0
             cum_sum_2 = cum_sum_2 + 0
     else:
             cum_sum_1 = cum_sum_1 + df_US_bar.iloc[0][i-1]*100
             cum_sum_2 = cum_sum_2 + df_US_bar.iloc[1][i-1]*100
     ax5.bar(x1, df_US_bar.iloc[0][i]*100, width=0.8, bottom = cum_sum_1, color = cmap_bar[i])
     ax5.bar(x2, df_US_bar.iloc[1][i]*100, width=0.8, bottom = cum_sum_2, color = cmap_bar[i])
     ax5.set_xticks([1,2])
     ax5.set_xticklabels(["uniform", "luxury"], fontsize = 8)
     ax5.set_title('USA abatement')
     ax5.yaxis.set_major_formatter(mticker.PercentFormatter())
ax5.annotate('d', xy=(ax5.get_xlim()[0]-0.1,ax5.get_ylim()[1]+5),fontsize=16 ,annotation_clip=False)
     
     

x3 = 1 
x4 = 2
cum_sum_3 = 0 
cum_sum_4 = 0
for i in range(0,14):
     if i == 0:
             cum_sum_3 = cum_sum_3 + 0
             cum_sum_4 = cum_sum_4 + 0
     else:
             cum_sum_3 = cum_sum_3 + df_SA_bar.iloc[0][i-1]*100
             cum_sum_4 = cum_sum_4 + df_SA_bar.iloc[1][i-1]*100
     ax6.bar(x1, df_SA_bar.iloc[0][i]*100, width=0.8, bottom = cum_sum_3, color = cmap_bar[i], label = df_SA_bar.columns[i])
     ax6.bar(x2, df_SA_bar.iloc[1][i]*100, width=0.8, bottom = cum_sum_4, color = cmap_bar[i])
     ax6.set_xticks([1,2])
     ax6.set_xticklabels(["uniform", "luxury"], fontsize = 8)
     ax6.set_title('SA abatement')
     ax6.legend(bbox_to_anchor=(1.05, 1), frameon = False)
     ax6.yaxis.set_major_formatter(mticker.PercentFormatter())
     ax6.get_yaxis().set_visible(False)

handles, labels = ax6.get_legend_handles_labels()
ax6.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), frameon = False)     
     

plt.tight_layout()
plt.savefig('fig3v2_1SA.png',bbox_inches = "tight", dpi = 300);
plt.show()


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

### PLOT MORE PROGRESSIVE TAX VERSIONS

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


cubed_data = np.genfromtxt('more_progressive_tax_data.csv', dtype = str, delimiter=',');

fig = plt.figure(figsize=(7,4))
outer = gridspec.GridSpec(1,1 ,figure=fig, height_ratios = [2]) 
#make nested gridspecs

gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[0], wspace = .05)

ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])


ax1.plot(percentiles, USA_pre_tax_carbon_cons_pc_total/1000, color = "black", label ="emissions per capita")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax1.set_yscale('log', basey = 2)
ax1.set_ylim(0.25,2**6)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.set_xlabel('percentile', fontsize = 8)
ax1.set_ylabel('CO2e tonnes/capita', fontsize = 8)
ax1.tick_params(axis='y')
ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax3.set_ylabel('% reduction in emissions per capita', color=color, fontsize = 8)  # we already handled the x-label with ax1
ax3.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = color, label =  "reduction luxury tax")
ax3.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_mean, color = "tab:blue", label = "reduction uniform tax")
ax3.plot(percentiles, cubed_data[1:,0].astype(float), color = "tab:green", label = "cubed elasticity")


ax3.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='best', bbox_to_anchor=(0.5, -.1, 0.5, 0.5), frameon = False, fontsize = 12)
ax3.legend(loc='best', bbox_to_anchor=(0.5, -0.2, 0.5, 0.5), frameon = False, fontsize = 12)
ax1.annotate('USA',xy= (45,40), xytext =(45,40), fontsize = 16)
#ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, color = color)
ax3.fill_between(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax3.fill_between(percentiles,USA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "tab:blue", alpha = 0.5)
ax3.fill_between(percentiles,cubed_data[1:,2].astype(float), cubed_data[1:,3].astype(float), facecolor = "tab:green", alpha = 0.5)

ax3.set_ylim((0,11))
ax3.axis('off')
#ax1.annotate('c', xy=(ax3.get_xlim()[0],ax3.get_ylim()[1]+8),fontsize=16 ,annotation_clip=False)

ax2.plot(percentiles, SA_pre_tax_carbon_cons_pc_total/1000, color = "black", label = "emissions per capita")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax2.set_yscale('log', basey = 2)
ax2.set_ylim(2**-2,2**6)
for axis in [ax4.xaxis, ax4.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax2.set_xlabel('percentile', fontsize = 8)
ax2.set_ylabel('CO2e tonnes/capita', fontsize = 8)
ax2.tick_params(axis='y')
ax4 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
color = 'black'
ax4.set_ylabel('% reduction in emissions per capita', color=color, fontsize = 8)  # we already handled the x-label with ax1
ax4.plot(percentiles, SA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = "tab:red", label = "reduction luxury tax")
ax4.plot(percentiles, SA_per_cent_reduction_emissions_pc_SCEN1_mean, color = "tab:blue", label = "reduction uniform tax")
ax4.plot(percentiles, cubed_data[1:,1].astype(float), color = "tab:green", label = "cubed elasticity")
ax4.tick_params(axis='y', labelcolor=color)
#ax4.legend(bbox_to_anchor=(0.2, 0.01, 0.5, 0.5), frameon = False)
#ax8.legend(bbox_to_anchor=(0.2, 0.08, 0.5, 0.5), frameon = False)
ax2.annotate('South Africa',xy= (25,40), xytext =(25,40), fontsize = 16)
ax4.fill_between(percentiles, SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax4.fill_between(percentiles,SA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, SA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "tab:blue", alpha = 0.5)
ax4.fill_between(percentiles,cubed_data[1:,4].astype(float), cubed_data[1:,5].astype(float), facecolor = "tab:green", alpha = 0.5)

ax4.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11])
ax4.set_ylim((0,11))
ax2.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig('fig4.png',bbox_inches = "tight", dpi = 300);
plt.show()



####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

### PLOT MORE PROGRESSIVE TAX VERSIONS VERSION #2

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################



cubed_data = np.genfromtxt('more_progressive_tax_data.csv', dtype = str, delimiter=',');
price_data = np.genfromtxt('data_cubed_elasticity_plot.csv', dtype = str, delimiter=',');

fig = plt.figure(figsize=(9.35,6.6))
outer = gridspec.GridSpec(2,1 ,figure=fig, height_ratios = [2, 2]) 
#make nested gridspecs

gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[0,0], wspace = .05)
gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[1,0], wspace = .05)

ax1 = fig.add_subplot(gs1[0,0])
ax2 = fig.add_subplot(gs1[0,1])
ax3 = fig.add_subplot(gs2[0,0])
ax4 = fig.add_subplot(gs2[0,1])



ax1.plot(percentiles, USA_pre_tax_carbon_cons_pc_total/1000, color = "black", label ="emissions per capita")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, USA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax1.set_yscale('log', basey = 2)
ax1.set_ylim(0.25,2**6)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.set_xlabel('percentile', fontsize = 8)
ax1.set_ylabel('CO2e tonnes/capita', fontsize = 8)
ax1.tick_params(axis='y')
ax5 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax5.set_ylabel('% reduction in emissions per capita', color=color, fontsize = 8)  # we already handled the x-label with ax1
ax5.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = color, label =  "reduction luxury tax")
ax5.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_mean, color = "tab:blue", label = "reduction uniform tax")
ax5.plot(percentiles, cubed_data[1:,0].astype(float), color = "tab:purple", label = "cubed elasticity")
ax1.annotate('a', xy=(ax1.get_xlim()[0]-20,ax1.get_ylim()[1]+20),fontsize=16 ,annotation_clip=False)

ax5.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='best', bbox_to_anchor=(0.5, 0, 0.5, 0.5), frameon = False, fontsize = 12)
ax5.legend(loc='best', bbox_to_anchor=(0.5, -0.1, 0.5, 0.5), frameon = False, fontsize = 12)
ax1.annotate('USA',xy= (45,40), xytext =(45,40), fontsize = 16)
#ax2.plot(percentiles, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, color = color)
ax5.fill_between(percentiles, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax5.fill_between(percentiles,USA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, USA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "tab:blue", alpha = 0.5)
ax5.fill_between(percentiles,cubed_data[1:,2].astype(float), cubed_data[1:,3].astype(float), facecolor = "tab:purple", alpha = 0.5)

ax5.set_ylim((0,11))
ax5.axis('off')


ax2.plot(percentiles, SA_pre_tax_carbon_cons_pc_total/1000, color = "black", label = "emissions per capita")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN1_total/1000, color = "tab:blue")
#ax1.plot(percentiles, CHINA_post_tax_carbon_cons_pc_SCEN2_2_total/1000, color = "blue", linestyle = "--")
ax2.set_yscale('log', basey = 2)
ax2.set_ylim(2**-2,2**6)
for axis in [ax4.xaxis, ax4.yaxis]:
    axis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax2.set_xlabel('percentile', fontsize = 8)
ax2.set_ylabel('CO2e tonnes/capita', fontsize = 8)
ax2.tick_params(axis='y')
ax6 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
color = 'black'
ax6.set_ylabel('% reduction in emissions per capita', color=color, fontsize = 8)  # we already handled the x-label with ax1
ax6.plot(percentiles, SA_per_cent_reduction_emissions_pc_SCEN2_2_mean, linestyle = '--', color = "tab:red", label = "reduction luxury tax")
ax6.plot(percentiles, SA_per_cent_reduction_emissions_pc_SCEN1_mean, color = "tab:blue", label = "reduction uniform tax")
ax6.plot(percentiles, cubed_data[1:,1].astype(float), color = "tab:purple", label = "cubed elasticity")
ax6.tick_params(axis='y', labelcolor=color)
#ax4.legend(bbox_to_anchor=(0.2, 0.01, 0.5, 0.5), frameon = False)
#ax8.legend(bbox_to_anchor=(0.2, 0.08, 0.5, 0.5), frameon = False)
ax2.annotate('South Africa',xy= (25,40), xytext =(25,40), fontsize = 16)
ax6.fill_between(percentiles, SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_low, SA_per_cent_reduction_emissions_pc_SCEN2_2_99CI_high, facecolor = "red", alpha = 0.5)
ax6.fill_between(percentiles,SA_per_cent_reduction_emissions_pc_SCEN1_99CI_low, SA_per_cent_reduction_emissions_pc_SCEN1_99CI_high, facecolor = "tab:blue", alpha = 0.5)
ax6.fill_between(percentiles,cubed_data[1:,4].astype(float), cubed_data[1:,5].astype(float), facecolor = "tab:purple", alpha = 0.5)

ax6.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11])
ax6.set_ylim((0,11))
ax2.get_yaxis().set_visible(False)



#category_names = labels[0:14, 3].tolist()
#category_names = [i[:-4] for i in category_names]
#category_names = [i[1:] for i in category_names]
#category_names[5]= 'Household Appliances'
#category_names[8]= 'Vehicle Fuel'
#category_names[13]= 'Education and Luxury'
price_uniform = np.repeat(150,14)
income_elasticities_USA  = income_elasticities[1232-14:1232]
price_differentiated_cubed = np.multiply(np.expand_dims(price_uniform,axis=1),income_elasticities_USA**3)*0.938024 ### normalization constant for the USA
price_differentiated = np.multiply(np.expand_dims(price_uniform,axis=1),income_elasticities_USA)*1.15908 

y_pos = np.arange(len(category_names))
ax3.barh(y_pos, price_uniform, align='center', label = 'uniform')
ax3.barh( y_pos, np.squeeze(price_differentiated, axis =1 ), height =0.45, align='center', label = 'luxury', color = 'tab:red')
ax3.barh( y_pos, np.squeeze(price_differentiated_cubed, axis =1 ), height =0.3, align='center', label = 'cubed', color = 'tab:purple')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(category_names)
ax3.invert_yaxis()  # labels read top-to-bottom
ax3.set_xlabel('Carbon price $/tonne');
#ax3.legend(frameon = False)
ax3.annotate('b', xy=(ax3.get_xlim()[0]-90,ax3.get_ylim()[1]-1/2),fontsize=16 ,annotation_clip=False)


##### same plot for South Africa for control and comparison
price_uniform_SA = np.repeat(50,14)
income_elasticities_SA = income_elasticities[14*48-14:48*14]
price_differentiated_SA = np.multiply(np.expand_dims(price_uniform_SA,axis=1),income_elasticities_SA)*1.05383
price_differentiated_SA_cubed = np.multiply(np.expand_dims(price_uniform_SA,axis=1),income_elasticities_SA**3)*0.620955

y_pos = np.arange(len(category_names))
ax4.barh(y_pos, price_uniform_SA, align='center', label = 'uniform')
ax4.barh(y_pos, np.squeeze(price_differentiated_SA, axis =1 ), height =0.45, align='center', label = 'luxury', color = 'tab:red')
ax4.barh(y_pos, np.squeeze(price_differentiated_SA_cubed, axis =1 ), height =0.3, align='center', label = 'cubed', color = 'tab:purple')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(category_names)
ax4.invert_yaxis()  # labels read top-to-bottom
ax4.set_xlabel('Carbon price $/tonne');
ax4.legend(frameon = False)
ax4.get_yaxis().set_visible(False)



plt.tight_layout()
plt.savefig('fig4_v2.png',bbox_inches = "tight", dpi = 300);
plt.show()


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################




#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
############################################## Plotting carbon elasticity national vs. progressivity measurement ################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


#### calculate national carbon elasticity 
national_carbon_elas = np.zeros((88,1))

for i in range(1,89):
    national_carbon_elas[i-1,0] = lin_fit(df_BIG_exp_2019_pc.iloc[14*i-14:14*i,:].sum(), df_BIG_carbon_2019_pc.iloc[14*i-14:14*i,:].sum())[1]



plt.scatter(national_carbon_elas, np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN1_simruns,axis=1))
plt.show()
plt.scatter(national_carbon_elas, np.mean(diff_first_and_last_percentile_reduction_emissions_SCEN2_2_simruns,axis=1))
plt.show()



### compute standard deviation of income elasticities 
std_income_elas = np.zeros((88,1))
mean_income_elas = np.zeros((88,1))
coeffv_income_elas = np.zeros((88,1))

for i in range(1,89):
   std_income_elas[i-1] = np.std(income_elasticities[14*i-14:14*i])
   mean_income_elas[i-1] = np.mean(income_elasticities[14*i-14:14*i])
   
   
coeffv_income_elas = std_income_elas/mean_income_elas

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
############################################## IMPORTANT ################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
############################################## CALCULATE HOW MUCH EXPENDITURE IS REDUCDED PER PERCENTILE PER COUNTRY ################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


##### build total difference between 2019 consumption and post tax one ######
scen1_consumption_reduction_pc = df_BIG_exp_2019_pc - scen1_df_post_tax_consumption_volume_pc
scen1_consumption_reduction_total = df_BIG_exp_2019_total - scen1_df_post_tax_consumption_volume_total
scen1_consumption_reduction_pc_relative = scen1_df_post_tax_consumption_volume_pc/df_BIG_exp_2019_pc
scen1_consumption_reduction_total_relative = scen1_df_post_tax_consumption_volume_total/df_BIG_exp_2019_total 

scen2_2_consumption_reduction_pc = df_BIG_exp_2019_pc - scen2_2_df_post_tax_consumption_volume_pc
scen2_2_consumption_reduction_total = df_BIG_exp_2019_total - scen2_2_df_post_tax_consumption_volume_total
scen2_2_consumption_reduction_pc_relative = scen2_2_df_post_tax_consumption_volume_pc/df_BIG_exp_2019_pc
scen2_2_consumption_reduction_total_relative = scen2_2_df_post_tax_consumption_volume_total/df_BIG_exp_2019_total 

scen1_diff_revenue_reduction = scen1_tax_revenue - scen1_consumption_reduction_total
scen2_2_diff_revenue_reduction = scen2_2_tax_revenue - scen2_2_consumption_reduction_total



scen1_consumption_reduction_total_percentile_total = pd.DataFrame(columns = labels2, index = meta_data_countries[:,0])
scen2_2_consumption_reduction_total_percentile_total = pd.DataFrame(columns = labels2, index = meta_data_countries[:,0])

for i in range(1,89):
    scen1_consumption_reduction_total_percentile_total.iloc[i-1,:] = scen1_consumption_reduction_total[i*14-14:i*14].sum(axis = 0)
    scen2_2_consumption_reduction_total_percentile_total.iloc[i-1,:] = scen2_2_consumption_reduction_total[i*14-14:i*14].sum(axis = 0)



###### compute total revenue vs. reduction in consumption total per country and for the
#### lower 50% in high income and upper middle income and lower 80% in lower middle income and low income
##### having national tax revenues already
scen1_average_national_tax_revenue;
scen2_2_average_national_tax_revenue;

########## now set up consumption gaps casued by the tax 

scen1_missing_consumption_national = np.zeros((88,1));
scen1_missing_consumption_per_percentile = np.zeros((88,100))


scen2_2_missing_consumption_national  = np.zeros((88,1));
scen2_2_missing_consumption_per_percentile = np.zeros((88,100));

for i in range (1,89):
    scen1_missing_consumption_national[i-1] = sum(scen1_consumption_reduction_total[i*14-14:i*14].sum());
    scen2_2_missing_consumption_national[i-1] = sum(scen1_consumption_reduction_total[i*14-14:i*14].sum());
    
    scen1_missing_consumption_per_percentile[i-1, :] = scen1_consumption_reduction_total[i*14-14:i*14].sum();
    scen2_2_missing_consumption_per_percentile[i-1, :] = scen2_2_consumption_reduction_total[i*14-14:i*14].sum();
    
    print("iteration is "+str(i))
    
    
    
#### these scen1_missing_consumption_per_percentile can now be used to calculate how much from total tax revenue has to be redistributed if
### we want to protect X% of the population from impact of the tax. this can also be done on a global level
#### but what we are really intersted is in how protection from poverty or even poverty alleviation compares against revenue investment in 
### carbon abatemet technology e.g. retrofitting or low carbon mobility. therefore we need a model of those 
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
############################################## LOAD Number of HOUSEHOLDs DATA  ################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
number_of_households_2019 = np.genfromtxt('household_numbers.csv', dtype = str, delimiter=',')




#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
############################################## LOAD ENERGY INENSITIES AND Transform into 1232x1 vector  ################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


energy_intensities_array = np.genfromtxt('energy_intensities.csv', dtype = str, delimiter=',').astype(float)

final_energy_intensities_estimate_2019 = np.zeros((1232,1));

for i in range(1,89):
   final_energy_intensities_estimate_2019[i*14-14:i*14,0] =  np.transpose(energy_intensities_array[i-1,:])

test_total_final_energy_total_sample = sum(df_BIG_exp_2019_total.multiply(final_energy_intensities_estimate_2019).sum()) 
### ~220 exajoule for hh energy demand in 88 countries of concern sounds reasonable


total_final_energy_2019 = df_BIG_exp_2019_total.multiply(final_energy_intensities_estimate_2019) 
total_final_energy_scen1 = scen1_average_consumption_granular.multiply(final_energy_intensities_estimate_2019)
total_final_energy_scen2_2 = scen2_2_average_consumption_granular.multiply(final_energy_intensities_estimate_2019)### unit megajoule
### ~220 exajoule for hh energy demand in 88 countries of concern sounds reasonable

#
##### test whether total expenditure including tax is more or less after introducing the tax
#### if this is positive than the original expenditure is more than the expenditure after introducing the tax i.e. that is the physical consumption 
#### which remains plus the tax rate on top. 
scen1_rest_function_of_consumption_money = df_BIG_exp_2019_total - (scen1_average_consumption_granular + scen1_average_tax_revenue_granular)
scen2_2_rest_function_of_consumption_money = df_BIG_exp_2019_total - (scen2_2_average_consumption_granular + scen2_2_average_tax_revenue_granular)

scen1_net_change_expenditure_percentile = np.zeros((88,100));
scen2_2_net_change_expenditure_percentile = np.zeros((88,100));

for i in range(1,89):
              scen1_net_change_expenditure_percentile[i-1,:] = scen1_rest_function_of_consumption_money.iloc[14*i-14:i*14,:].sum(axis = 0);
              scen2_2_net_change_expenditure_percentile[i-1,:]  = scen2_2_rest_function_of_consumption_money.iloc[14*i-14:i*14,:].sum(axis = 0);
              #plt.plot(percentiles, scen1_net_change_expenditure_percentile[i-1,:])
              plt.plot(percentiles, scen2_2_net_change_expenditure_percentile[i-1,:])

### these scen1_missing_consumption_per_percentile can now be used to calculate how much from total tax revenue has to be redistributed if
### we want to protect X% of the population from impact of the tax. this can also be done on a global level
#### but what we are really intersted is in how protection from poverty or even poverty alleviation compares against revenue investment in 
### carbon abatemet technology e.g. retrofitting or low carbon mobility. therefore we need a model of those 

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
############################################## FIRST REVENUE RECYCLING ANALYSIS  ################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

coverable_by_tax_revenue_scen1 = np.zeros((88,100));
coverable_by_tax_revenue_scen2_2 = np.zeros((88,100));
for k in range(1,89):
        for i in range(1,101):
              coverable_by_tax_revenue_scen1[k-1,i-1] = int((scen1_average_national_tax_revenue[k-1] - scen1_consumption_reduction_total_percentile_total.iloc[k-1,0:i].sum())>0)
              coverable_by_tax_revenue_scen2_2[k-1,i-1] =  int((scen2_2_average_national_tax_revenue[k-1] - scen2_2_consumption_reduction_total_percentile_total.iloc[k-1,0:i].sum())>0)


revenue_left_scen1_share_country = np.zeros((88,100));
revenue_left_scen1_share_global = np.zeros((1,100));

revenue_left_scen2_2_share_country = np.zeros((88,100));
revenue_left_scen2_2_share_global = np.zeros((1,100));

##### plot revenue left vs. percentile covered

#### we do not need factor in the tax rate for the government paying for the consumption of the percentiles. 
### Because the government so to speak is freed from the tax, it does not pay the tax to itself and generates recursively revenue again. 

for k in range(1,89):
    revenue_left_scen1 = scen1_average_national_tax_revenue[k-1]
    revenue_left_scen2_2 = scen2_2_average_national_tax_revenue[k-1]
    for i in range(1,101):
               #### compute the percentage share of revenue left 
               revenue_left_scen1 = revenue_left_scen1 - scen1_consumption_reduction_total_percentile_total.iloc[k-1,i-1]
               revenue_left_scen2_2 = revenue_left_scen2_2 - scen2_2_consumption_reduction_total_percentile_total.iloc[k-1,i-1]
               if revenue_left_scen1 > 0:
                        revenue_left_scen1_share_country[k-1,i-1] = revenue_left_scen1/scen1_average_national_tax_revenue[k-1]
               else:
                        revenue_left_scen1_share_country[k-1,i-1] = 0; 
               if revenue_left_scen2_2 > 0:
                        revenue_left_scen2_2_share_country[k-1,i-1] = revenue_left_scen2_2/scen2_2_average_national_tax_revenue[k-1]
               else:
                        revenue_left_scen2_2_share_country[k-1,i-1] = 0; 
                        
                        

### shape per country is genereally concave
for i in range(1,89):
   plt.plot(percentiles, revenue_left_scen1_share_country[i-1,:], color = "tab:blue", alpha = 0.5)
plt.xlabel("percentiles covered tax impact")
plt.ylabel("tax revenue left")
plt.show()

for i in range(1,89):
   plt.plot(percentiles, revenue_left_scen2_2_share_country[i-1,:], color = "tab:red", alpha = 0.5)
plt.xlabel("percentiles covered tax impact")
plt.ylabel("tax revenue left")
plt.show()



#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  NEW TEST PROGRESSIVITY ##################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


##################### check actual progressivity in the sense that share of total money spend how much is tax in relation
#### to total spends/disposable income


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
############################################## NEW TEST PROGRESSIVITY ##################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################



for i in range(1,89):
        plt.plot(percentiles, scen2_2_average_tax_incidence_granular[i-1,:])
plt.show()


for i in range(1,89):
        plt.plot(percentiles, scen1_average_tax_incidence_granular[i-1,:])
 

## negative if progressiv
diff_top_bottom_scen2_2 =  scen2_2_average_tax_incidence_granular[:,0]-  scen2_2_average_tax_incidence_granular[:,99]

### positive if regressive
diff_top_bottom_scen1 = scen1_average_tax_incidence_granular[:,0]- scen1_average_tax_incidence_granular[:,99]





#%%

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
############################################## 2nd REVENUE RECYCLING ANALYSIS  INCLUDING RETROFIT INVESTMENT and redistribution #####
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##############################################  ALGORITHM ##################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

##### preparation####
#### calculate energy and emissions (from heat and electricity per household) per scenario ####

### we need number of hh per percentile to compute energy used per household/dwelling from heat and electricity
number_of_households_percentile = number_of_households_2019[:,1].astype(float)/100;


scen1_average_emissions_granular_total = scen1_average_consumption_granular.multiply(carbon_intensities_2019_estimate, axis = 0)
scen2_2_average_emissions_granular_total = scen2_2_average_consumption_granular.multiply(carbon_intensities_2019_estimate, axis = 0)

domestic_energy_scen1 = total_final_energy_scen1.iloc[4::14]
domestic_energy_spends_scen1 = scen1_average_consumption_granular.iloc[4::14]
domestic_emissions_scen1 = copy.deepcopy(scen1_average_emissions_granular_total.iloc[4::14])

per_dwelling_energy_scen1 = domestic_energy_scen1.multiply(1/number_of_households_percentile,axis = 0)
per_dwelling_domestic_energy_spends_scen1 = domestic_energy_spends_scen1.multiply(1/number_of_households_percentile,axis = 0)


domestic_energy_scen2_2 = total_final_energy_scen2_2.iloc[4::14]
domestic_energy_spends_scen2_2 = scen2_2_average_consumption_granular.iloc[4::14]
domestic_emissions_scen2_2 = copy.deepcopy(scen2_2_average_emissions_granular_total.iloc[4::14])

per_dwelling_energy_scen2_2 = domestic_energy_scen2_2.multiply(1/number_of_households_percentile,axis = 0)
per_dwelling_domestic_energy_spends_scen2_2 = domestic_energy_spends_scen2_2.multiply(1/number_of_households_percentile,axis = 0)


##### MAIN PART n####
### Model a 50% energy reduction deep retrofit based on 0.77$/MJ cost of reduction derived from US meta study




emission_reductions_sensitivity_to_protected_percent_scen1 =  np.zeros((101,88))
emission_reductions_sensitivity_to_protected_percent_scen2_2 = np.zeros((101,88))

emissions_per_country_sensitivity_to_protected_percent_scen1 = np.zeros((101,88))
emissions_per_country_sensitivity_to_protected_percent_scen2_2 = np.zeros((101,88))

budget_redist_per_country_sensitivity_to_protected_percent_scen1 = np.zeros((101,88))
budget_redist_per_country_sensitivity_to_protected_percent_scen2_2 = np.zeros((101,88))

budget_total_per_country_scen1 = np.zeros((101,88))
budget_total_per_country_scen2_2 = np.zeros((101,88))

costs_per_megajoule_retrofit_differentiated = np.zeros((88,1));
for i in range(1,89):
    if meta_data_countries[i-1,7] == 'North':
        costs_per_megajoule_retrofit_differentiated[i-1,0] = 0.77
    else: 
        costs_per_megajoule_retrofit_differentiated[i-1,0] = 0.77*0.55
    

percentiles = np.linspace(0,100,101)

#### only for luxury scenario calculated fitting to the figure #3
#### extra emissions from redistribution
extra_emissions_from_redistribution = np.zeros((101,88))
#### extra abated emissions from retrofit

extra_abated_emissions_from_retrofit = np.zeros((101,88))

cost_sensitivity_param = 1 ### if 1 then no sensitivity analysis. if 2 then double as in lower bound fig 3 panel c, if 1/2 upper bound 

##### loop over all countries ######
print("start revenue recycling simulation")
for m in range (1,89):

        
        for p in range(0,101):

                            
                    costs_per_retrofit_per_income_group_scen1 = (per_dwelling_energy_scen1.iloc[m-1,:]/2)*costs_per_megajoule_retrofit_differentiated[m-1,0]*cost_sensitivity_param
                    costs_per_retrofit_per_income_group_scen2_2 = (per_dwelling_energy_scen2_2.iloc[m-1,:]/2)*costs_per_megajoule_retrofit_differentiated[m-1,0]*cost_sensitivity_param
                                    
                    emissions_pre_tax = sum(df_BIG_carbon_2019_total.iloc[m*14-14:m*14].sum())
                    emissions_post_tax_scen1 = scen1_average_reduced_emissions[m-1] 
                    emissions_post_tax_scen2_2 = scen2_2_average_reduced_emissions[m-1] 
                                   
                    reduced_cons_scen1 = scen1_average_reduced_consumption_granular.iloc[m*14-14:m*14,:]
                    reduced_cons_scen2_2 = scen2_2_average_reduced_consumption_granular.iloc[m*14-14:m*14,:]
                                    
                    carbon_intensities = carbon_intensities_2019_estimate[m*14-14:m*14]
                    energy_intensities = final_energy_intensities_estimate_2019[m*14-14:m*14]
                                    
                    tax_revenue_scen1 = copy.deepcopy(scen1_average_national_tax_revenue[m-1])
                    tax_revenue_scen2_2 = copy.deepcopy(scen2_2_average_national_tax_revenue[m-1])
                                    
                    number_of_households_percentile = copy.deepcopy(number_of_households_2019[m-1,1].astype(float)/100)
                                    
                    domestic_emissions_percentile_scen1 = copy.deepcopy(domestic_emissions_scen1.iloc[m-1,:])
                    domestic_emissions_percentile_pre_retrofit_scen1 = copy.deepcopy(domestic_emissions_scen1.iloc[m-1,:])
                                    
                    domestic_emissions_percentile_scen2_2 = copy.deepcopy(domestic_emissions_scen2_2.iloc[m-1,:])
                    domestic_emissions_percentile_pre_retrofit_scen2_2 = copy.deepcopy(domestic_emissions_scen2_2.iloc[m-1,:])
                                    
                    budget_scen1 = copy.deepcopy(tax_revenue_scen1)
                    budget_scen2_2 = copy.deepcopy(tax_revenue_scen2_2)
                    
                    #### full budget save ####
                    budget_total_per_country_scen2_2[p,m-1] = budget_scen2_2
                    budget_total_per_country_scen1[p,m-1] = budget_scen1
                                    
                    additional_emissions_scen1 = 0 
                    additional_emissions_scen2_2 = 0 
                    
        
                    for i in range(1,p + 1):
                          for c in range(1,15):
                              
                              if budget_scen1 - reduced_cons_scen1.iloc[c-1,i-1] > 0:
                                    budget_scen1 = budget_scen1 - reduced_cons_scen1.iloc[c-1,i-1]
                                    additional_emissions_scen1 = additional_emissions_scen1 + reduced_cons_scen1.iloc[c-1,i-1]*carbon_intensities[c-1]               
                              
                              if budget_scen2_2 - reduced_cons_scen2_2.iloc[c-1,i-1] > 0:
                                    budget_scen2_2 = budget_scen2_2 - reduced_cons_scen2_2.iloc[c-1,i-1]
                                    additional_emissions_scen2_2 = additional_emissions_scen2_2 + reduced_cons_scen2_2.iloc[c-1,i-1]*carbon_intensities[c-1]               
                    
                    #### after redistr.  before retrofit budget save####
                    budget_redist_per_country_sensitivity_to_protected_percent_scen2_2[p,m-1] = copy.deepcopy(budget_scen2_2)
                    budget_redist_per_country_sensitivity_to_protected_percent_scen1[p,m-1] = copy.deepcopy(budget_scen1)
                    
                    
                    extra_emissions_from_redistribution[p,m-1] = copy.deepcopy(additional_emissions_scen2_2)
                    
                    
                    print("budget left before retrofit invest " + str((round(budget_scen2_2)/10**9)) + " billions")
                    for i in range(1,101):
                        if budget_scen1 > 0:
                                           if (budget_scen1 - (costs_per_retrofit_per_income_group_scen1[i-1] * number_of_households_percentile)) > 0:
                                               budget_scen1 = budget_scen1 - (costs_per_retrofit_per_income_group_scen1[i-1] * number_of_households_percentile)
                                               domestic_emissions_percentile_scen1.iloc[i-1] = domestic_emissions_percentile_scen1.iloc[i-1]/2
                                           else:
                                                 break
                        if budget_scen2_2 > 0:
                                           if (budget_scen2_2 - (costs_per_retrofit_per_income_group_scen2_2[i-1] * number_of_households_percentile)) > 0:
                                               budget_scen2_2 = budget_scen2_2 - (costs_per_retrofit_per_income_group_scen2_2[i-1] * number_of_households_percentile)
                                               domestic_emissions_percentile_scen2_2.iloc[i-1] = domestic_emissions_percentile_scen2_2.iloc[i-1]/2
                                           else:
                                                 break 
                      
                     
                                               
                    print("budget left before bisect to use up rest " + str((round(budget_scen2_2)/10**9)) + " billions")   
                    
                    ########### for scenario #1 retrofit ####
                    ### initiate the interval we want to bisect
                    upper_bound_scen1 = number_of_households_percentile
                    lower_bound_scen1 = 0
                    g = 0
                    NMAX = 100
                    
                    while g < NMAX:
                        g = g + 1
                        if (budget_scen1 - (costs_per_retrofit_per_income_group_scen1[i-1] * (upper_bound_scen1+lower_bound_scen1)/2) < 0): 
                            upper_bound_scen1 = (upper_bound_scen1+lower_bound_scen1)/2
                        else:
                            lower_bound_scen1 = (upper_bound_scen1+lower_bound_scen1)/2
                    budget_scen1 = budget_scen1 - costs_per_retrofit_per_income_group_scen1[i-1] * (upper_bound_scen1+lower_bound_scen1)/2
                    ### initiate correct percentile segment
                    help_em_per_dwelling_scen1 = domestic_emissions_percentile_scen1[i-1]/number_of_households_percentile
                    ### compute weighted average for left over segment
                    domestic_emissions_percentile_scen1[i-1] = help_em_per_dwelling_scen1/2*upper_bound_scen1 + (number_of_households_percentile-upper_bound_scen1)*help_em_per_dwelling_scen1             
                    extra_abated_through_retrofit_scen1 = domestic_emissions_percentile_pre_retrofit_scen1 - domestic_emissions_percentile_scen1                     
                    emissions_post_tax_scen1_post_revenue_recycling_scen1 = emissions_post_tax_scen1 + additional_emissions_scen1 - sum(extra_abated_through_retrofit_scen1)
                    final_reduction_emissions_post_revenue_recycling_scen1 = round((1 - emissions_post_tax_scen1_post_revenue_recycling_scen1/emissions_pre_tax)*100,2)  ### in percent
                    print("emission reduction after tax impact and revenue recycling is " + str(final_reduction_emissions_post_revenue_recycling_scen1) + " %")
                   
                    emission_reductions_sensitivity_to_protected_percent_scen1[p,m-1] = final_reduction_emissions_post_revenue_recycling_scen1   

                    emissions_per_country_sensitivity_to_protected_percent_scen1[p,m-1] = emissions_post_tax_scen1_post_revenue_recycling_scen1
                    
                    ########### for scenario #2 retrofit ####
                    ### initiate the interval we want to bisect
                    upper_bound_scen2_2 = number_of_households_percentile
                    lower_bound_scen2_2 = 0
                    g = 0
                    NMAX = 100
                    while g < NMAX:
                        g = g + 1
                        if (budget_scen2_2 - (costs_per_retrofit_per_income_group_scen2_2[i-1] * (upper_bound_scen2_2+lower_bound_scen2_2)/2) < 0): 
                            upper_bound_scen2_2 = (upper_bound_scen2_2+lower_bound_scen2_2)/2
                        else:
                            lower_bound_scen2_2 = (upper_bound_scen2_2+lower_bound_scen2_2)/2
                    budget_scen2_2 = budget_scen2_2 - costs_per_retrofit_per_income_group_scen2_2[i-1] * (upper_bound_scen2_2+lower_bound_scen2_2)/2
                    ### initiate correct percentile segment
                    help_em_per_dwelling_scen2_2 = domestic_emissions_percentile_scen2_2[i-1]/number_of_households_percentile
                    ### compute weighted average for left over segment
                    domestic_emissions_percentile_scen2_2[i-1] = help_em_per_dwelling_scen2_2/2*upper_bound_scen2_2 + (number_of_households_percentile-upper_bound_scen2_2)*help_em_per_dwelling_scen2_2            
                    extra_abated_through_retrofit_scen2_2 = domestic_emissions_percentile_pre_retrofit_scen2_2 - domestic_emissions_percentile_scen2_2                     
                    emissions_post_tax_scen2_2_post_revenue_recycling_scen2_2 = emissions_post_tax_scen2_2 + additional_emissions_scen2_2 - sum(extra_abated_through_retrofit_scen2_2)
                    final_reduction_emissions_post_revenue_recycling_scen2_2 = round((1 - emissions_post_tax_scen2_2_post_revenue_recycling_scen2_2/emissions_pre_tax)*100,2)  ### in percent
                    print("emission reduction after tax impact and revenue recycling is " + str(final_reduction_emissions_post_revenue_recycling_scen2_2) + " %")
                   
                    emission_reductions_sensitivity_to_protected_percent_scen2_2[p,m-1] = final_reduction_emissions_post_revenue_recycling_scen2_2   
                    
                    emissions_per_country_sensitivity_to_protected_percent_scen2_2[p,m-1] = emissions_post_tax_scen2_2_post_revenue_recycling_scen2_2
                    
                    
                    extra_abated_emissions_from_retrofit[p,m-1] = copy.deepcopy(sum(extra_abated_through_retrofit_scen2_2))
                    
                    
        print("iteration is " + str(m))
        
#### compute pre tax national total emissions
total_national_emissions_2019 = np.zeros((88,1));
for i in range(1,89):    
   total_national_emissions_2019[i-1, 0] = sum(df_BIG_carbon_2019_total.iloc[i*14-14:i*14].sum())


##### compute global emissions reductions in revenue recycling scenarios ####

scen1_global_em_sensitivity_rev_recycling = 1-np.sum(emissions_per_country_sensitivity_to_protected_percent_scen1,axis = 1)/total_national_emissions_2019.sum()
scen2_2_global_em_sensitivity_rev_recycling = 1-np.sum(emissions_per_country_sensitivity_to_protected_percent_scen2_2,axis = 1)/total_national_emissions_2019.sum()

plt.plot(percentiles, scen1_global_em_sensitivity_rev_recycling*100)
plt.plot(percentiles, scen2_2_global_em_sensitivity_rev_recycling*100)
plt.plot(percentiles, np.repeat((1-sum(scen1_average_reduced_emissions)/sum(total_national_emissions_2019))*100,101), linestyle = '--')
plt.plot(percentiles, np.repeat((1-sum(scen2_2_average_reduced_emissions)/sum(total_national_emissions_2019))*100,101), linestyle = '--')
plt.ylim((0,10))
plt.xlabel("tax protected percentile")
plt.ylabel("CO2e reduction % post revenue recycling")
plt.margins(x=0,y=0);
plt.show()




##### compute total global budget allocation to redist. vs. retrofit in simulation above

### the numerator measured how much budget is left BEFORE RETROFIT, so it measured how much of the total proportion goes to retrfoti
budget_global_going_to_retrofit_scen2_2 = np.sum(budget_redist_per_country_sensitivity_to_protected_percent_scen2_2, axis = 1)/np.sum(budget_total_per_country_scen2_2, axis = 1)
budget_global_going_to_redistr_scen2_2 = 1 - budget_global_going_to_retrofit_scen2_2

budget_global_going_to_retrofit_scen1 = np.sum(budget_redist_per_country_sensitivity_to_protected_percent_scen1, axis = 1)/np.sum(budget_total_per_country_scen1, axis = 1)
budget_global_going_to_redistr_scen1 = 1 - budget_global_going_to_retrofit_scen1


plt.plot(percentiles, budget_global_going_to_redistr_scen2_2, label = "redistribution")
plt.plot(percentiles, budget_global_going_to_retrofit_scen2_2, label = "retrofit")
plt.plot(percentiles, budget_global_going_to_redistr_scen1, label = "redistribution1")
plt.plot(percentiles, budget_global_going_to_retrofit_scen1, label = "retrofit1")
plt.xlabel("paid out percentiles")
plt.ylabel("% of revenue allocated")
plt.legend(frameon = False)
plt.margins(x=0,y=0);
plt.show()



#### compute marginal retrofit extra abated emissions and marginal redistribution emissions (emissions from redist.)
### compute this also as a share 

extra_emissions_from_redistribution_GLOBAL = np.sum(extra_emissions_from_redistribution, axis = 1)
extra_abated_emissions_from_retrofit_GLOBAL = np.sum(extra_abated_emissions_from_retrofit, axis = 1)

extra_emissions_from_redistribution_GLOBAL_SHARE = extra_emissions_from_redistribution_GLOBAL/sum(total_national_emissions_2019)
extra_abated_emissions_from_retrofit_GLOBAL_SHARE = extra_abated_emissions_from_retrofit_GLOBAL/sum(total_national_emissions_2019)


MARGINAL_extra_emissions_from_redistribution_GLOBAL = np.zeros((100,1))

MARGINAL_extra_abated_emissions_from_retrofit_GLOBAL = np.zeros((100,1))

for i in range(0,100):
      MARGINAL_extra_emissions_from_redistribution_GLOBAL[i] = extra_emissions_from_redistribution_GLOBAL[i+1]-extra_emissions_from_redistribution_GLOBAL [i] 
      MARGINAL_extra_abated_emissions_from_retrofit_GLOBAL[i] = extra_abated_emissions_from_retrofit_GLOBAL[i]- extra_abated_emissions_from_retrofit_GLOBAL[i+1]
      if i == 100:
          MARGINAL_extra_abated_emissions_from_retrofit_GLOBAL[i] = extra_abated_emissions_from_retrofit_GLOBAL[i] ### because marginal is here in relation to 0 which is not in the vector
          
####### PLOT global concave trade off between tax exemption and emissions abatement #####


plot_data_trade_off = np.genfromtxt('plot_data_trade_off.csv', dtype = str, delimiter=',').astype(float)



plt.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,1]*100, label = "sensitivity")
plt.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,2]*100, label = "best estimate")
plt.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,3]*100, color = "tab:blue")
plt.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,4], color = "black", linestyle = '--', label = 'no rev. recycling')
plt.plot([34.5, 34.5],[0, 6], color = "black" , alpha = 0.5)
plt.margins(x=0,y=0);
plt.xlabel("paid out percentiles")
plt.ylabel("Global CO2e reduction %")
plt.legend(frameon= False, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
plt.fill_between(plot_data_trade_off[:,0],plot_data_trade_off[:,1]*100,plot_data_trade_off[:,3]*100, facecolor = "blue", alpha = 0.2)
plt.annotate('Retrofit', xy=(78,7),fontsize=10)
plt.annotate('Redistribution', xy=(78,5.0),fontsize=10)
plt.annotate('No trade-off', xy=(6,3),fontsize=11)
plt.arrow(75, 6.2, 0, 1, head_width = 3, head_length = 0.5)
plt.arrow(75, 5.8, 0, -1, head_width = 3, head_length = 0.5)
plt.show()
#plt.savefig('fig4.png',bbox_inches = "tight", dpi = 300);







####### PLOT POTENTIAL PANEL D WHERE NO TRADE_OFF point from panel c in figure 4 is illustrated per country instead of globally


trade_off_zero_point_array = np.zeros((88,1))
for i in range(0,88):
    a = abs(emissions_per_country_sensitivity_to_protected_percent_scen2_2[:,i]/scen2_2_average_reduced_emissions[i]-1) ### find minimum distance to trade_off zero point
    index = np.where(a == a.min()) ### find index of minimum distance to trade_off zero point
    trade_off_zero_point = index ### not plus one because where index = 0 means no percentile is paid back, so index matches the percentiles
    trade_off_zero_point_array[i,0] = trade_off_zero_point[0][0]

### try to find correlates o the trade_off_zero_point_array

### 1st try Gini index of consumption 2019 

### calculate total consumption per percentile

Gini_consumption_per_country = np.zeros((88,1))
for i in range(1,89): 
   Gini_consumption_per_country[i-1,0] = giniold(np.transpose(stacked_population_percentiles[100*i-100:100*i,:]), np.expand_dims(np.transpose(np.array(df_BIG_exp_2019_total.iloc[14*i-14:14*i,:].sum())),axis=0))

plt.scatter(Gini_consumption_per_country, trade_off_zero_point_array)
plt.xlabel("Gini consumption")
plt.ylabel("no trade-off point")
plt.show()

###2nd try absolute spending level on Heat&Elect. that is absolute average spending per capita

(df_BIG_exp_2019_pc.sum(axis = 1)/100)[4::14]

plt.scatter((df_BIG_exp_2019_pc.sum(axis = 1)/100)[4::14], trade_off_zero_point_array)
plt.xlabel("absolute spending on residential energy")
plt.ylabel("no trade-off point")
plt.show()


###3rd try prproption of spending level on Heat&Elect.

residential_energy_spends_proportion = np.zeros((88,1))
for i in range(1,89):
  residential_energy_spends_proportion[i-1,0] =  (df_BIG_exp_2019_pc.sum(axis = 1)/100)[4+(i-1)*14]/sum((df_BIG_exp_2019_pc.sum(axis = 1)/100)[i*14-14:i*14])


plt.scatter(residential_energy_spends_proportion, trade_off_zero_point_array)
plt.xlabel("relative spending on residential energy")
plt.ylabel("no trade-off point")
plt.show()



plt.hist(trade_off_zero_point_array)
plt.show()

##### make graph which plots the trade_off points 
A  = np.append(meta_data_countries,(trade_off_zero_point_array).astype(float), axis = 1)
A  = np.append(A,Gini_consumption_per_country, axis = 1)
for i in range(0,88):
    A[i,8] =  A[i,8].zfill(4)
B = A[A[:, 8].argsort()]


fig, ax = plt.subplots(figsize=(4,10))
ax.scatter(B[:, 8].astype(float), np.linspace(1,88,88))
ax.plot(np.repeat(33,93), np.linspace(-1,92,93), color = "black", label = "world level")
ax.margins(x=0.05,y=0.05);
ax.set_ylim((0,88));
ax.set_yticks(np.linspace(1,88,88))
ax.set_yticklabels(B[:,0], fontsize = 5)

plt.show()


trade_off_zero_point_array
meta_data_countries

##############################################################################################################################


##################### PLOT NEW FIGURE #4 INCLUDING PANEL A, B AND C ################

fig5_a_b = np.genfromtxt('data_fig5_a_b.csv', dtype = str, delimiter=',')
plot_data_trade_off = np.genfromtxt('plot_data_trade_off.csv', dtype = str, delimiter=',').astype(float)


percentiles = np.arange(0,101,1)


widths_grid = [2, 2, 2]
heights_grid = [1, 1, 1]
fig = plt.figure(figsize=(5,8))

gs = GridSpec(nrows = 3, ncols = 1, figure=fig)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])



ax1.plot(percentiles, fig5_a_b[1:102,0].astype(float)*100, label = "redistribution")
ax1.plot(percentiles, fig5_a_b[1:102,1].astype(float)*100, label = "retrofit")
ax1.set_xlabel("percentiles paid back")
ax1.set_ylabel("% of revenue spend on")
ax1.get_xaxis().set_visible(False)
ax1.plot([33, 33],[0, 100], color = "black" , alpha = 0.5)
ax1.margins(x=0,y=0);
ax1.legend(frameon = False, bbox_to_anchor=(0.4, 0.45, 0.5, 0.5))
ax1.annotate('a', xy=(-10,ax1.get_ylim()[1]+8),fontsize=14 ,annotation_clip=False)



ax2.plot(percentiles, fig5_a_b[1:102,2].astype(float)*100, label = "redistribution")
ax2.plot(percentiles, fig5_a_b[1:102,3].astype(float)*100, label = "retrofit")
ax2.set_ylabel("% of emissions affected", color = "black")
ax2.get_xaxis().set_visible(False)
ax2.plot([33, 33],[0, 8], color = "black" , alpha = 0.5)
ax2.margins(x=0,y=0);
ax2.set_ylim((0,6))
#ax4 = ax2.twinx()  # in
#ax4.set_ylabel("% of emissions reduced", color = "tab:orange")
ax2.legend(frameon = False, bbox_to_anchor=(0.38, 0.45, 0.5, 0.5))
#ax4.tick_params(axis='y', labelcolor="tab:orange")
ax2.tick_params(axis='y', labelcolor="black")
ax2.annotate('b', xy=(-10,ax2.get_ylim()[1]+0.2),fontsize=14 ,annotation_clip=False)

ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,1]*100, label = "sensitivity", color = "tab:green")
ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,2]*100, label = "best estimate", color = "tab:brown")
ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,3]*100, color = 'tab:green')
ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,4], color = "black", linestyle = '--', label = 'no rev. recycling')
ax3.plot([33, 33],[0, 10], color = "black" , alpha = 0.5)
ax3.margins(x=0,y=0);
ax3.set_xlabel("percentiles paid back with revenue")
ax3.set_ylabel("CO2e reduction % achieved")
ax3.set_ylim((0,9))
ax3.legend(frameon= False, bbox_to_anchor=(0.35, 0., 0.5, 0.5), fontsize = 9)
ax3.fill_between(plot_data_trade_off[:,0],plot_data_trade_off[:,1]*100,plot_data_trade_off[:,3]*100, facecolor = 'green', alpha = 0.2)
ax3.annotate('Retrofit', xy=(78,7),fontsize=8)
ax3.annotate('Redistribution', xy=(78,5.0),fontsize=8)
ax3.annotate('No trade-off', xy=(4,3),fontsize=9)
ax3.arrow(75, 6.2, 0, 1, head_width = 3, head_length = 0.5, color = "tab:orange")
ax3.arrow(75, 5.8, 0, -1, head_width = 3, head_length = 0.5, color = "tab:blue")
ax3.annotate('c', xy=(-10,ax3.get_ylim()[1]+0.2),fontsize=14 ,annotation_clip=False)


plt.tight_layout()
#plt.savefig('fig4.png',bbox_inches = "tight", dpi = 300);
plt.show()

###################################### ###################################### ###################################### ###################################### 
###################################### NEW VERSION OF THE ABOVE PLOT ################################################################ 
###################################### ###################################### ###################################### ###################################### 

fig = plt.figure(figsize=(9,11))

gs = GridSpec(nrows = 9, ncols = 3, figure=fig)

ax1 = fig.add_subplot(gs[0:2, :2])
ax2 = fig.add_subplot(gs[2:4, :2])
ax3 = fig.add_subplot(gs[4:6, :2])
ax4 = fig.add_subplot(gs[0:9, 2])
ax5 = fig.add_subplot(gs[6:9, :2])
#gs.update(wspace=0.025, hspace=0.05)


ax1.plot(percentiles, fig5_a_b[1:102,0].astype(float)*100, label = "redistribution")
ax1.plot(percentiles, fig5_a_b[1:102,1].astype(float)*100, label = "retrofit")
ax1.set_xlabel("percentiles paid back")
ax1.set_ylabel("% of revenue spend on")
ax1.get_xaxis().set_visible(False)
ax1.plot([33, 33],[0, 100], color = "black" , alpha = 0.5)
ax1.margins(x=0,y=0);
ax1.legend(frameon = False, bbox_to_anchor=(0.4, 0.45, 0.5, 0.5))
ax1.annotate('a', xy=(-15,ax1.get_ylim()[1]+8),fontsize=14 ,annotation_clip=False)



ax2.plot(percentiles, fig5_a_b[1:102,2].astype(float)*100, label = "redistribution")
ax2.plot(percentiles, fig5_a_b[1:102,3].astype(float)*100, label = "retrofit")
ax2.set_ylabel("% of emissions affected", color = "black")
ax2.get_xaxis().set_visible(False)
ax2.plot([33, 33],[0, 8], color = "black" , alpha = 0.5)
ax2.margins(x=0,y=0);
ax2.set_ylim((0,6))
#ax4 = ax2.twinx()  # in
#ax4.set_ylabel("% of emissions reduced", color = "tab:orange")
ax2.legend(frameon = False, bbox_to_anchor=(0.38, 0.45, 0.5, 0.5))
#ax4.tick_params(axis='y', labelcolor="tab:orange")
ax2.tick_params(axis='y', labelcolor="black")
ax2.annotate('b', xy=(-15,ax2.get_ylim()[1]+0.6),fontsize=14 ,annotation_clip=False)

ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,1]*100, label = "sensitivity", color = "tab:green")
ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,2]*100, label = "best estimate", color = "tab:brown")
ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,3]*100, color = 'tab:green')
ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,4], color = "black", linestyle = '--', label = 'no rev. recycling')
ax3.plot([33, 33],[0, 10], color = "black" , alpha = 0.5)
ax3.margins(x=0,y=0);
ax3.set_xlabel("percentiles paid back with revenue")
ax3.set_ylabel("CO2e reduction % achieved")
ax3.set_ylim((0,9))
ax3.legend(frameon= False, bbox_to_anchor=(0.35, 0., 0.5, 0.5), fontsize = 9)
ax3.fill_between(plot_data_trade_off[:,0],plot_data_trade_off[:,1]*100,plot_data_trade_off[:,3]*100, facecolor = 'green', alpha = 0.2)
ax3.annotate('Retrofit', xy=(78,7),fontsize=8)
ax3.annotate('Redistribution', xy=(78,5.0),fontsize=8)
ax3.annotate('Zero trade-off', xy=(4,3),fontsize=9)
ax3.arrow(75, 6.2, 0, 1, head_width = 3, head_length = 0.5, color = "tab:orange")
ax3.arrow(75, 5.8, 0, -1, head_width = 3, head_length = 0.5, color = "tab:blue")
ax3.annotate('c', xy=(-15,ax3.get_ylim()[1]+1.2),fontsize=14 ,annotation_clip=False)



A  = np.append(meta_data_countries,(trade_off_zero_point_array).astype(float), axis = 1)
A  = np.append(A,Gini_consumption_per_country, axis = 1)
for i in range(0,88):
    A[i,8] =  A[i,8].zfill(4)
B = A[A[:, 8].argsort()]


#ax4.scatter(B[:, 8].astype(float), np.linspace(1,88,88))

df8 = pd.DataFrame(dict(x = B[:, 8].astype(float) , y = np.linspace(1,88,88), label=B[:, 6]))
groups8 = df8.groupby('label')
markers = ['*', '^', "+", ".", "o"]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
run = 0;

for name, group in groups8:
    run = run + 1
    ax4.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=4, label=name, color = colors[run-1]); 
    ax4.margins(x=0.05,y=0.05);
    ax4.set_ylim((0,89));
    ax4.set_yticks(np.linspace(1,88,88))
    ax4.set_yticklabels(B[:,0], fontsize = 6.5)
    ax4.set_xlabel("Zero trade-off point")
    ax4.locator_params(nbins=10, axis='x')
    #ax4.set_xticklabels(B[:,8], fontsize = 6.5)
    #ax4.set_xlim((0,80))
    #ax4.set_xticks(np.linspace(0,80,3))

ax4.set_xlim((0,80))
#ax4.set_xticks(np.linspace(0,80,3))
        
#ax4.plot(np.repeat(33,93), np.linspace(-1,92,93), color = "black", label = "world level")
#ax4.margins(x=0.05,y=0.05);
#ax4.set_ylim((0,89));
#ax4.set_yticks(np.linspace(1,88,88))
#ax4.set_yticklabels(B[:,0], fontsize = 6.5)
#ax4.set_xlabel("Zero trade-off point")
ax4.annotate('e', xy=(-0.005,ax4.get_ylim()[1]+1),fontsize=14 ,annotation_clip=False)
#ax4.legend(frameon = False, fontsize = 14)
ax4.legend(loc='best', markerscale=2, bbox_to_anchor=(0.15, 0.15, 0.7, 0), frameon = False, fontsize = 12)  


df9 = pd.DataFrame(dict(x = np.squeeze((Gini_consumption_per_country*100).astype(float)) , y = np.squeeze(trade_off_zero_point_array), label=meta_data_countries[:,6]))
groups9 = df9.groupby('label')
markers = ['*', '^', "+", ".", "o"]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
run = 0;

for name, group in groups9:
    run = run + 1
    ax5.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=5, label=name, color = colors[run-1]); 

#ax5.scatter(Gini_consumption_per_country*100, trade_off_zero_point_array)
ax5.set_xlabel("Gini index consumption")
ax5.set_ylabel("Zero trade-off point")
ax5.annotate('d', xy=(-10,ax5.get_ylim()[1]+6),fontsize=14 ,annotation_clip=False)
b = lin_fit_non_log(Gini_consumption_per_country*100, trade_off_zero_point_array)[0][1]
a = lin_fit_non_log(Gini_consumption_per_country*100, trade_off_zero_point_array)[0][0]
new_x = np.linspace(0,1,100)*100
new_y = a + b*new_x
ax5.plot(new_x,new_y, '--', color = "black", linewidth = 5 )
ax5.set_xlim(0,60)
ax5.set_ylim(0,80)
ax5.text(5, 60, r'$y=4.7+x$', fontsize=12.5)
ax5.text(5, 70, r'$R^2$= 0.35', fontsize=12.5)

plt.tight_layout()
plt.savefig('fig5.png',bbox_inches = "tight", dpi = 300);
plt.show()



labels = np.genfromtxt('labels.csv', dtype = str, delimiter=',');
############# COMPRESS AND PREPARE DATA FOR DYNAMIC ANALYSIS
df_BIG_exp_2019_pc_quintiles = pd.DataFrame(columns = [1,2,3,4,5], index = labels);
for i in range(1,6):
    df_BIG_exp_2019_pc_quintiles.iloc[:,i-1] = df_BIG_exp_2019_pc.iloc[:,i*20-20:i*20].sum(axis = 1)/20

pop_quintile_2019 = pop_percentile_2019*20


#df_BIG_exp_2019_pc_quintiles
#pop_quintile_2019
#carbon_intensities_2019_estimate
#energy_intensities
#final_energy_intensities_estimate_2019




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AGAIN NEW VERSION !!!!!!!!!!!!!!!!!!###########
###################################### ###################################### ###################################### ###################################### 
###################################### !!!!NEW VERSION OF THE ABOVE PLOT #2!!!! ################################################################ 
###################################### ###################################### ###################################### ###################################### 



img=mpimg.imread('Figure5_panel_e.png')

img2=mpimg.imread('Figure5_panel_e_v2.png')



fig = plt.figure(figsize=(8,8))

gs = GridSpec(nrows = 3, ncols = 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[:2, 1])
ax5 = fig.add_subplot(gs[2, 1])
#gs.update(wspace=0.025, hspace=0.05)


ax1.plot(percentiles, fig5_a_b[1:102,0].astype(float)*100, label = "redistribution")
ax1.plot(percentiles, fig5_a_b[1:102,1].astype(float)*100, label = "retrofit")
ax1.set_xlabel("percentiles paid back")
ax1.set_ylabel("% of revenue spent")
ax1.get_xaxis().set_visible(False)
ax1.plot([33, 33],[0, 100], color = "black" , alpha = 0.5)
ax1.margins(x=0,y=0);
#ax1.legend(frameon = False, bbox_to_anchor=(0.4, 0.45, 0.5, 0.5))
ax1.annotate('a', xy=(-15,ax1.get_ylim()[1]+8),fontsize=14 ,annotation_clip=False)



ax2.plot(percentiles, fig5_a_b[1:102,2].astype(float)*100, label = "redistribution")
ax2.plot(percentiles, fig5_a_b[1:102,3].astype(float)*100, label = "retrofit")
ax2.set_ylabel("% of emissions affected", color = "black")
ax2.get_xaxis().set_visible(False)
ax2.plot([33, 33],[0, 8], color = "black" , alpha = 0.5)
ax2.margins(x=0,y=0);
ax2.set_ylim((0,6))
#ax4 = ax2.twinx()  # in
#ax4.set_ylabel("% of emissions reduced", color = "tab:orange")
ax2.legend(frameon = False, bbox_to_anchor=(0.38, 0.45, 0.5, 0.5))
#ax4.tick_params(axis='y', labelcolor="tab:orange")
ax2.tick_params(axis='y', labelcolor="black")
ax2.annotate('b', xy=(-15,ax2.get_ylim()[1]+0.6),fontsize=14 ,annotation_clip=False)

ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,1]*100, label = "sensitivity", color = "tab:green")
ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,2]*100, label = "best estimate", color = "tab:brown")
ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,3]*100, color = 'tab:green')
ax3.plot(plot_data_trade_off[:,0],plot_data_trade_off[:,4], color = "black", linestyle = '--', label = 'no rev. recycling')
ax3.plot([33, 33],[0, 10], color = "black" , alpha = 0.5)
ax3.margins(x=0,y=0);
ax3.set_xlabel("percentiles paid back with revenue")
ax3.set_ylabel("CO2e reduction % achieved")
ax3.set_ylim((0,9))
ax3.legend(frameon= False, bbox_to_anchor=(0.35, 0., 0.5, 0.5), fontsize = 9)
ax3.fill_between(plot_data_trade_off[:,0],plot_data_trade_off[:,1]*100,plot_data_trade_off[:,3]*100, facecolor = 'green', alpha = 0.2)
ax3.annotate('Retrofit', xy=(77,7),fontsize=8)
ax3.annotate('Redistribution', xy=(76,5.0),fontsize=8)
ax3.annotate('Zero trade-off', xy=(4,3),fontsize=9)
ax3.arrow(75, 6.2, 0, 1, head_width = 3, head_length = 0.5, color = "tab:orange")
ax3.arrow(75, 5.8, 0, -1, head_width = 3, head_length = 0.5, color = "tab:blue")
#https://stackoverflow.com/questions/15971768/drawing-arrow-in-x-y-coordinate-in-python
ax3.plot([20,33], [4,6], color = "black")


ax3.annotate('c', xy=(-15,ax3.get_ylim()[1]+1.2),fontsize=14 ,annotation_clip=False)



A  = np.append(meta_data_countries,(trade_off_zero_point_array).astype(float), axis = 1)
A  = np.append(A,Gini_consumption_per_country, axis = 1)
for i in range(0,88):
    A[i,8] =  A[i,8].zfill(4)
B = A[A[:, 8].argsort()]



df9 = pd.DataFrame(dict(x = np.squeeze((Gini_consumption_per_country*100).astype(float)) , y = np.squeeze(trade_off_zero_point_array), label=meta_data_countries[:,6]))
groups9 = df9.groupby('label')
markers = ['*', '^', "+", ".", "o"]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
run = 0;

for name, group in groups9:
    run = run + 1
    ax4.plot(group.x, group.y, marker=markers[run-1], linestyle='', ms=5, label=name, color = colors[run-1]); 

#ax5.scatter(Gini_consumption_per_country*100, trade_off_zero_point_array)
ax4.set_xlabel("Gini index consumption (x 100)")
ax4.set_ylabel("Zero trade-off point (in percentiles)")
ax4.annotate('d', xy=(-10,ax4.get_ylim()[1]+8.5),fontsize=14 ,annotation_clip=False)
b = lin_fit_non_log(Gini_consumption_per_country*100, trade_off_zero_point_array)[0][1]
a = lin_fit_non_log(Gini_consumption_per_country*100, trade_off_zero_point_array)[0][0]
new_x = np.linspace(0,1,100)*100
new_y = a + b*new_x
ax4.plot(new_x,new_y, '--', color = "black", linewidth = 4 )
ax4.set_xlim(0,60)
ax4.set_ylim(0,80)
ax4.text(42, 20, r'$y=4.7+x$', fontsize=12.5)
ax4.text(42, 14, r'$R^2$= 0.35', fontsize=12.5)
ax4.text(42, 8, r'N = 88', fontsize=12.5)
ax4.legend(loc='best', markerscale=2, frameon = False, fontsize = 10)  
ax4.annotate("USA",(Gini_consumption_per_country[87]*100, trade_off_zero_point_array[87]), (10,50), arrowprops =dict(arrowstyle ="-"), fontsize = 10)
ax4.annotate("SA",(Gini_consumption_per_country[47]*100, trade_off_zero_point_array[47]),(50,40), arrowprops =dict(arrowstyle ="-"), fontsize = 10)


imgplot = ax5.imshow(img2) 
ax5.annotate('e', xy=(-50,ax5.get_ylim()[1]+2),fontsize=14 ,annotation_clip=False)
ax5.axis('off')


plt.tight_layout()
plt.savefig('fig5.png',bbox_inches = "tight", dpi = 300);
plt.show()



labels = np.genfromtxt('labels.csv', dtype = str, delimiter=',');
############# COMPRESS AND PREPARE DATA FOR DYNAMIC ANALYSIS
df_BIG_exp_2019_pc_quintiles = pd.DataFrame(columns = [1,2,3,4,5], index = labels);
for i in range(1,6):
    df_BIG_exp_2019_pc_quintiles.iloc[:,i-1] = df_BIG_exp_2019_pc.iloc[:,i*20-20:i*20].sum(axis = 1)/20

pop_quintile_2019 = pop_percentile_2019*20


#df_BIG_exp_2019_pc_quintiles
#pop_quintile_2019
#carbon_intensities_2019_estimate
#energy_intensities
#final_energy_intensities_estimate_2019






























savetxt('df_BIG_exp_2019_pc_quintiles.csv', df_BIG_exp_2019_pc_quintiles, delimiter=',')
savetxt('final_energy_intensities_estimate_2019.csv', final_energy_intensities_estimate_2019, delimiter=',')
savetxt('pop_quintile_2019.csv', pop_quintile_2019)



#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
################################ CALCULATE AND SET UP TABLE #1 AND SAVE IT AS A RESULT ##############################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


column_names = [ 'USA', 'China', 'India', 'Europe', 'SSA+Mena', 'Latin_America', 'Rest of Asia']
rows1 = ['emission_reductions', 'revenue_top10', 'revenue_luxury']
rows2 = [ 'luxury', 'uniform']
iterables = [rows1, 
             rows2]
pd.MultiIndex.from_product(iterables, names=["first", "second"])
df_table1 = pd.DataFrame(columns = column_names, index = pd.MultiIndex.from_product(iterables, names=["first", "second"]))

####################### COMPUTE total emissions reductions for regions

scen2_2_average_reduced_emissions
scen1_average_reduced_emissions
total_national_emissions_2019
meta_data_countries[:,6]

####### calculate % reduced emissions for both scenarios
###USA
df_table1.loc[("emission_reductions", "luxury"), "USA"] = float(1- scen2_2_average_reduced_emissions[87]/total_national_emissions_2019[87])*100
df_table1.loc[("emission_reductions", "uniform"), "USA"] = float(1- scen1_average_reduced_emissions[87]/total_national_emissions_2019[87])*100
##China
df_table1.loc[("emission_reductions", "luxury"), "China"] = float(1- scen2_2_average_reduced_emissions[11]/total_national_emissions_2019[11])*100
df_table1.loc[("emission_reductions", "uniform"), "China"] = float(1- scen1_average_reduced_emissions[11]/total_national_emissions_2019[11])*100
###India
df_table1.loc[("emission_reductions", "luxury"), "India"] = float(1- scen2_2_average_reduced_emissions[21]/total_national_emissions_2019[21])*100
df_table1.loc[("emission_reductions", "uniform"), "India"] = float(1- scen1_average_reduced_emissions[21]/total_national_emissions_2019[21])*100

## Europe
df_table1.loc[("emission_reductions", "luxury"), "Europe"] = (1-sum(np.multiply((meta_data_countries[:,6] == "Europe"),scen2_2_average_reduced_emissions))/sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(total_national_emissions_2019))))*100
df_table1.loc[("emission_reductions", "uniform"), "Europe"] = (1-sum(np.multiply((meta_data_countries[:,6] == "Europe"),scen1_average_reduced_emissions))/sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(total_national_emissions_2019))))*100

## SSA+Mena
df_table1.loc[("emission_reductions", "luxury"), "SSA+Mena"] = (1-sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),scen2_2_average_reduced_emissions))/sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(total_national_emissions_2019))))*100
df_table1.loc[("emission_reductions", "uniform"), "SSA+Mena"] = (1-sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),scen1_average_reduced_emissions))/sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(total_national_emissions_2019))))*100

## Latin America
df_table1.loc[("emission_reductions", "luxury"), "Latin_America"] = (1-sum(np.multiply((meta_data_countries[:,6] == "Latin America"),scen2_2_average_reduced_emissions))/sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(total_national_emissions_2019))))*100
df_table1.loc[("emission_reductions", "uniform"), "Latin_America"] = (1-sum(np.multiply((meta_data_countries[:,6] == "Latin America"),scen1_average_reduced_emissions))/sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(total_national_emissions_2019))))*100

## Rest of Asia
###roa = rest of asia
roa_1 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),scen2_2_average_reduced_emissions))-scen2_2_average_reduced_emissions[11]-scen2_2_average_reduced_emissions[21]
roa_2 = float(sum(np.multiply((meta_data_countries[:,6] == "Asia"),np.squeeze(total_national_emissions_2019)))-total_national_emissions_2019[11]- total_national_emissions_2019[21])
roa_3 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),scen1_average_reduced_emissions))-scen1_average_reduced_emissions[11]-scen1_average_reduced_emissions[21]

df_table1.loc[("emission_reductions", "luxury"), "Rest of Asia"] = float((1-roa_1/roa_2)*100)
df_table1.loc[("emission_reductions", "uniform"), "Rest of Asia"] = float((1-roa_3/roa_2)*100)



################### COMPUTE TAX REVENUE GENERATED BY THE TOP 10% within countries (but across regions as in table #1)

scen1_average_national_tax_revenue;
scen2_2_average_national_tax_revenue;

scen1_average_tax_revenue_granular;
scen2_2_average_tax_revenue_granular;

scen1_tax_revenue_top10_national = np.zeros((88,1))
scen2_2_tax_revenue_top10_national = np.zeros((88,1))

for j in range(1,89):
    scen1_tax_revenue_top10_national[j-1,0] = sum(scen1_average_tax_revenue_granular.iloc[j*14-14:j*14,90:100].sum())
    scen2_2_tax_revenue_top10_national[j-1,0] = sum(scen2_2_average_tax_revenue_granular.iloc[j*14-14:j*14,90:100].sum())
    sum(scen1_average_tax_revenue_granular.iloc[j*14-14:j*14,90:100].sum())/scen1_average_national_tax_revenue[j-1]
    sum(scen2_2_average_tax_revenue_granular.iloc[j*14-14:j*14,90:100].sum())/scen1_average_national_tax_revenue[j-1]



###USA
df_table1.loc[("revenue_top10", "luxury"), "USA"] = float(scen2_2_tax_revenue_top10_national[87]/scen2_2_average_national_tax_revenue[87])*100
df_table1.loc[("revenue_top10", "uniform"), "USA"] = float(scen1_tax_revenue_top10_national[87]/scen1_average_national_tax_revenue[87])*100

##China
df_table1.loc[("revenue_top10", "luxury"), "China"] = float(scen2_2_tax_revenue_top10_national[11]/scen2_2_average_national_tax_revenue[11])*100
df_table1.loc[("revenue_top10", "uniform"), "China"] = float(scen1_tax_revenue_top10_national[11]/scen1_average_national_tax_revenue[11])*100

##India
df_table1.loc[("revenue_top10", "luxury"), "India"] = float(scen2_2_tax_revenue_top10_national[21]/scen2_2_average_national_tax_revenue[21])*100
df_table1.loc[("revenue_top10", "uniform"), "India"] = float(scen1_tax_revenue_top10_national[21]/scen1_average_national_tax_revenue[21])*100

## Europe
df_table1.loc[("revenue_top10", "luxury"), "Europe"] = sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(scen2_2_tax_revenue_top10_national)))/sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(scen2_2_average_national_tax_revenue)))*100
df_table1.loc[("revenue_top10", "uniform"), "Europe"] = sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(scen1_tax_revenue_top10_national)))/sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(scen1_average_national_tax_revenue)))*100


## SSA + MENA
df_table1.loc[("revenue_top10", "luxury"), "SSA+Mena"] = sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(scen2_2_tax_revenue_top10_national)))/sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(scen2_2_average_national_tax_revenue)))*100
df_table1.loc[("revenue_top10", "uniform"), "SSA+Mena"] = sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(scen1_tax_revenue_top10_national)))/sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(scen1_average_national_tax_revenue)))*100


## Latin America
df_table1.loc[("revenue_top10", "luxury"), "Latin_America"] = sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(scen2_2_tax_revenue_top10_national)))/sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(scen2_2_average_national_tax_revenue)))*100
df_table1.loc[("revenue_top10", "uniform"), "Latin_America"] = sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(scen1_tax_revenue_top10_national)))/sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(scen1_average_national_tax_revenue)))*100


roa1 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),np.squeeze(scen1_tax_revenue_top10_national))) - scen1_tax_revenue_top10_national[11] - scen1_tax_revenue_top10_national[21] 
roa2 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),np.squeeze(scen2_2_tax_revenue_top10_national))) - scen2_2_tax_revenue_top10_national[11] - scen2_2_tax_revenue_top10_national[21] 
roa3 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),np.squeeze(scen1_average_national_tax_revenue))) - scen1_average_national_tax_revenue[11] - scen1_average_national_tax_revenue[21]
roa4 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),np.squeeze(scen2_2_average_national_tax_revenue))) - scen2_2_average_national_tax_revenue[11] - scen2_2_average_national_tax_revenue[21]


## Rest of Asia
df_table1.loc[("revenue_top10", "luxury"), "Rest of Asia"] = float(roa2/roa4)*100
df_table1.loc[("revenue_top10", "uniform"), "Rest of Asia"] = float(roa1/roa3)*100





################### COMPUTE TAX REVENUE GENERATED BY luxury consumption (but across regions as in table #1) ######

luxury_yes = np.intc((income_elasticities > 1))

scen1_revenue_from_luxury = np.zeros((88,1))
scen2_2_revenue_from_luxury = np.zeros((88,1))

for j in range(1,89):
   scen1_revenue_from_luxury[j-1,0] = (luxury_yes[j*14-14:j*14,0]*scen1_average_tax_revenue_granular.iloc[j*14-14:j*14,:].sum(axis =1)).sum()
   scen2_2_revenue_from_luxury[j-1,0] = (luxury_yes[j*14-14:j*14,0]*scen2_2_average_tax_revenue_granular.iloc[j*14-14:j*14,:].sum(axis =1)).sum()

###USA
df_table1.loc[("revenue_luxury", "luxury"), "USA"] = float(scen2_2_revenue_from_luxury[87]/scen2_2_average_national_tax_revenue[87])*100
df_table1.loc[("revenue_luxury", "uniform"), "USA"] = float(scen1_revenue_from_luxury[87]/scen1_average_national_tax_revenue[87])*100

###China
df_table1.loc[("revenue_luxury", "luxury"), "China"] = float(scen2_2_revenue_from_luxury[11]/scen2_2_average_national_tax_revenue[11])*100
df_table1.loc[("revenue_luxury", "uniform"), "China"] = float(scen1_revenue_from_luxury[11]/scen1_average_national_tax_revenue[11])*100

###India
df_table1.loc[("revenue_luxury", "luxury"), "India"] = float(scen2_2_revenue_from_luxury[21]/scen2_2_average_national_tax_revenue[21])*100
df_table1.loc[("revenue_luxury", "uniform"), "India"] = float(scen1_revenue_from_luxury[21]/scen1_average_national_tax_revenue[21])*100


## Europe
df_table1.loc[("revenue_luxury", "luxury"), "Europe"] = sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(scen2_2_revenue_from_luxury)))/sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(scen2_2_average_national_tax_revenue)))*100
df_table1.loc[("revenue_luxury", "uniform"), "Europe"] = sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(scen1_revenue_from_luxury)))/sum(np.multiply((meta_data_countries[:,6] == "Europe"),np.squeeze(scen1_average_national_tax_revenue)))*100

## SSA + MENA
df_table1.loc[("revenue_luxury", "luxury"), "SSA+Mena"] = sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(scen2_2_revenue_from_luxury)))/sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(scen2_2_average_national_tax_revenue)))*100
df_table1.loc[("revenue_luxury", "uniform"), "SSA+Mena"] = sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(scen1_revenue_from_luxury)))/sum(np.multiply((meta_data_countries[:,6] == "SSA + MENA"),np.squeeze(scen1_average_national_tax_revenue)))*100


## Latin America
df_table1.loc[("revenue_luxury", "luxury"), "Latin_America"] = sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(scen2_2_revenue_from_luxury)))/sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(scen2_2_average_national_tax_revenue)))*100
df_table1.loc[("revenue_luxury", "uniform"), "Latin_America"] = sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(scen1_revenue_from_luxury)))/sum(np.multiply((meta_data_countries[:,6] == "Latin America"),np.squeeze(scen1_average_national_tax_revenue)))*100

## Rest of Asia
roa1 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),np.squeeze(scen1_revenue_from_luxury))) - scen1_revenue_from_luxury[11] - scen1_revenue_from_luxury[21] 
roa2 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),np.squeeze(scen2_2_revenue_from_luxury))) - scen2_2_revenue_from_luxury[11] - scen2_2_revenue_from_luxury[21] 
roa3 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),np.squeeze(scen1_average_national_tax_revenue))) - scen1_average_national_tax_revenue[11] - scen1_average_national_tax_revenue[21]
roa4 = sum(np.multiply((meta_data_countries[:,6] == "Asia"),np.squeeze(scen2_2_average_national_tax_revenue))) - scen2_2_average_national_tax_revenue[11] - scen2_2_average_national_tax_revenue[21]


df_table1.loc[("revenue_luxury", "luxury"), "Rest of Asia"] = float(roa2/roa4)*100
df_table1.loc[("revenue_luxury", "uniform"), "Rest of Asia"] = float(roa1/roa3)*100

#%%


####  test how much is the total global tax revenue compared to total global consumption in the luxury scenario and uniform scenario
sum(np.sum(scen2_2_tax_revenue))/sum(scen2_2_df_post_tax_consumption_volume_total.sum())
sum(np.sum(scen1_tax_revenue))/sum(scen1_df_post_tax_consumption_volume_total.sum())



#%%


###### test gini coefficient scen2_2_average_emissions_granular_total
### scen1_average_emissions_granular_total

scen2_2_average_emissions_granular_total.stack()
scen1_average_emissions_granular_total.sum(axis = 0)

df_pc_yearly_emissions_total_2019

df_yearly_emissions_total_2019_post_luxury_tax = pd.DataFrame(columns = labels2, index = meta_data_countries[:,0])
df_yearly_emissions_total_2019_post_uniform_tax = pd.DataFrame(columns = labels2, index = meta_data_countries[:,0])
for i in range(1,89):  
     for j in range(0,100):
            df_yearly_emissions_total_2019_post_luxury_tax.iloc[i-1,j] = scen2_2_average_emissions_granular_total.iloc[i*14-14:i*14,j].sum();
            df_yearly_emissions_total_2019_post_uniform_tax.iloc[i-1,j]= scen1_average_emissions_granular_total.iloc[i*14-14:i*14,j].sum();
            
mega_test = df_yearly_emissions_total_2019_post_luxury_tax.stack()
mega_test2 = df_yearly_emissions_total_2019_post_uniform_tax.stack()

df_pc_yearly_emissions_total_2019.stack().iloc[100*i-100:100*i]*np.squeeze(stacked_population_percentiles[100*i-100:100*i])
#### this is all about emissions here
national_gini_post_luxury = np.zeros((88,1))
national_gini_post_uniform = np.zeros((88,1))
national_gini_before_tax = np.zeros((88,1))

for i in range(1,89):
    national_gini_post_luxury[i-1,0] = gini_array_version(np.squeeze(stacked_population_percentiles[100*i-100:100*i]), np.array(df_yearly_emissions_total_2019_post_luxury_tax.stack().iloc[100*i-100:100*i]))
    national_gini_post_uniform[i-1,0] = gini_array_version(np.squeeze(stacked_population_percentiles[100*i-100:100*i]), np.array(df_yearly_emissions_total_2019_post_uniform_tax.stack().iloc[100*i-100:100*i]))
    national_gini_before_tax[i-1,0] = gini_array_version(np.squeeze(stacked_population_percentiles[100*i-100:100*i]), np.array(df_pc_yearly_emissions_total_2019.stack().iloc[100*i-100:100*i]*np.squeeze(stacked_population_percentiles[100*i-100:100*i])))



#%%

### make new figure 4 for illustrating relationship between GDPpc and impact of luxury tax design
diff_in_diff_data_vs_GDP = np.genfromtxt('diff_in_diff_scenarios_fig4_data.csv', dtype = str, delimiter=',');

x1 = diff_in_diff_data_vs_GDP[2:,2].astype(float)
y1 = diff_in_diff_data_vs_GDP[2:,6].astype(float)*100 ## transform to %
y2 = diff_in_diff_data_vs_GDP[2:,9].astype(float)*100 ## 


widths_grid = [2, 2]
heights_grid = [1]
fig = plt.figure(constrained_layout=True, figsize=(7,3.5))


gs = GridSpec(1, 2, figure=fig ,width_ratios = widths_grid, height_ratios = heights_grid)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

#fig.suptitle('Progressivity gain luxury design compared to uniform design', fontsize=14)

df10 = pd.DataFrame(dict(x = x1, y = y1, label=diff_in_diff_data_vs_GDP[2:,3]))
df11 = pd.DataFrame(dict(x = x1, y = y2, label=diff_in_diff_data_vs_GDP[2:,3]))
groups = df10.groupby('label')
groups2 = df11.groupby('label')
#colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
#colors1 = plt.cm.get_cmap('Blues') # colors1(1-1/run/3)
#colors2 = plt.cm.get_cmap('Reds')
markers = ['*', '^', "+", ".", "o"]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']


#### fit number 1 for left panel a 
results_lin_fit_1 = lin_fit((x1),(y1)) 
x_model = np.linspace((10**2),(10**6),100)
y_model = np.exp(results_lin_fit_1[0])*x_model**results_lin_fit_1[1]

runY1 = 0 
runY2 = 0
for name, group in groups:
    runY1 = runY1 + 1
    ax1.plot(group.x, group.y, marker=markers[runY1-1], linestyle='', ms=4, label=name, color = colors[runY1-1]);
    
ax1.set_yscale('log' ,basey = 10)
ax1.set_xscale('log', basex = 10)
ax1.set_ylim((10**-1.5,10**1.1))
ax1.set_xlim((10**2.9,10**5.1))
ax1.set_xlabel('GDPpc PPP 2019');
ax1.set_ylabel('% change in (1st - 100th percentile)');
ax1.plot(x_model, y_model, linewidth=3.0, c = 'black', linestyle = '--');
ax1.set_title(label = "Individual emission reductions")
ax1.set_yticks([0.1,1,10])
ax1.set_yticklabels(["0.1%","1%","10%"])
ax1.text(10**4.2, 10**-1.1, r'$y=0.005x^{0.6}$', fontsize=10)
ax1.text(10**4.2, 10**-1.1*1.5, r'$R^2$= 0.29', fontsize=10)
ax1.text(10**4.2, 10**-1.1/1.5, r'N = 88', fontsize=10)
 

#### fit number 1 for left panel a 
results_lin_fit_2 = lin_fit((x1),(y2)) 
x2_model = np.linspace((10**2),(10**6),100)
y2_model = np.exp(results_lin_fit_2[0])*x2_model**results_lin_fit_2[1]


for name, group in groups2:
    runY2 = runY2 + 1
    ax2.plot(group.x, group.y, marker=markers[runY2-1], linestyle='', ms=4, label=name, color = colors[runY2-1]);
ax2.set_yscale('log' ,basey = 10)
ax2.set_xscale('log', basex = 10)
ax2.set_ylim((10**-1.5,10**1.1))
ax2.set_xlim((10**2.9,10**5.1))
ax2.set_xlabel('GDPpc PPP 2019');
ax2.plot(x2_model, y2_model, linewidth=3.0, c = 'black', linestyle = '--');
ax2.set_title(label = "Financial burden")
ax2.get_yaxis().set_visible(False)
ax2.text(10**4.2, 10**-1.1, r'$y=0.001x^{0.64}$', fontsize=10)
ax2.text(10**4.2, 10**-1.1*1.5, r'$R^2$= 0.51', fontsize=10)
ax2.text(10**4.2, 10**-1.1/1.5, r'N = 88', fontsize=10)
ax2.legend(loc='best', markerscale=1.5, frameon = False, fontsize = 9)  


plt.savefig('fig4_new_new_new.png',bbox_inches = "tight", dpi = 300);
