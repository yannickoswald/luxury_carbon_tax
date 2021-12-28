
"""
Created on Tue Jun  1 14:01:28 2021

@author: eeyo
"""
#%%
##### 'infrastructure' loading ######
import os
os.getcwd()
#work 
#os.chdir("C:/Users/eeyo/Dropbox/Bildung/PhD/3. paper")
#home 
os.chdir("C:/Users/y-osw/Dropbox/Bildung/PhD/3. paper")


import pandas as pd 
import math 
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
import scipy as scipy
import scipy.special as ssp
import copy
from gini_dynamic import *
from gini import *
from giniold import *
from gini_array_version import *
from lin_fit import *
from lin_fit_non_log import *
from scipy.stats import gamma
import math as math
from matplotlib import rc
import matplotlib.gridspec as gridspec
from scipy.special import erfinv
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
from collections import OrderedDict

### set base variable space ####

carbon_intensities_2019_estimate = np.genfromtxt('carbon_intensities_2019_estimate.csv', dtype = float, delimiter=',');
final_energy_intensities_estimate_2019 = np.genfromtxt('final_energy_intensities_estimate_2019.csv', dtype = float, delimiter=',');
income_elasticities = np.expand_dims(np.genfromtxt('income_elasticities.csv', dtype = float, delimiter=','),axis=1); ###e-01 so it is not wrong just notated that way
income_elasticities_SE = np.expand_dims(np.genfromtxt('income_elasticities_SE.csv', dtype = float, delimiter=','),axis=1);#
weight_for_elasticities = np.expand_dims(np.genfromtxt('weight_for_elasticities.csv', dtype = float, delimiter=','),axis=1);#
### elasticities that have been interpolated/assumed to be 1 have no standard error variation but is assumed to be securely 0 because even if some reasonable stochastic term
### would be assumed it would be extremely minor effect on results.
income_elasticities_SE = np.nan_to_num(income_elasticities_SE)
population_WB_2019 = np.genfromtxt('population_WB_2019.csv', dtype = str, delimiter=',');
labels = np.genfromtxt('labels.csv', dtype = str, delimiter=',');
array_BIG_exp_2019_pc_quintiles = np.genfromtxt('df_BIG_exp_2019_pc_quintiles.csv', dtype = float, delimiter=',');
pop_quintile_2019 = np.genfromtxt('pop_quintile_2019.csv', dtype = float, delimiter=',');
years = np.linspace(2020,2100,81);
years2 = np.linspace(2019,2100,82);
meta_data_countries = np.genfromtxt('country_national_level_meta.csv', dtype = str, delimiter=',');
Gini_cons_national = np.genfromtxt('Gini_consumption_national_output.csv', dtype = float, delimiter=',');
### for consumption projections set in. elast. to 1 if < 0 
#(because we do not expect cons to shrink with growing income/GDP over time actually, only for alc tobacco but for others definitely unrealistic)
income_elasticities = np.where(income_elasticities  < 0, 1, income_elasticities )

### base scenario evolution parameters  ####
ssp2_cons_pc_growth = np.genfromtxt('consumption_growth_rates_forecast.csv', dtype = float, delimiter=',');
ssp2_pop_growth = np.genfromtxt('pop_growth_rates_forecast.csv', dtype = float, delimiter=',');
standard_em_intensity_evolution = np.genfromtxt('ssp2_standard_scen_em_evolution.csv', dtype = float, delimiter=',');

### other
number_of_households_2019 = np.genfromtxt('household_numbers.csv', dtype = str, delimiter=',')
households_per_capita = (number_of_households_2019[:,1]).astype(float)/(population_WB_2019[:,1]).astype(float)




############ ROBUSTNESS SIMPLIFICATION, REIGNING IN OUTLIERS WITH ASSUMPTIONS #####################
income_elasticities = np.where(income_elasticities < 0, 0.1, income_elasticities) #### replace negative elasticities with concave engel curve.
np.nan_to_num(income_elasticities_SE, copy = False, nan = 0)
##### we clip standard error of elasticities to avoid hypersensitivity of model to parameters, only an extremely minor share of parameters is affected by this. 
SE_as_fraction_of_elas = income_elasticities_SE/abs(income_elasticities) #### coefficient_of_variation SE/mean https://en.wikipedia.org/wiki/Coefficient_of_variation
SE_as_fraction_of_elas_2 = SE_as_fraction_of_elas.clip(max = 1) ### restrict coefficient of variation to 1 , https://www.readyratios.com/reference/analysis/coefficient_of_variation.html
SE_as_fraction_of_elas_2 = np.where(SE_as_fraction_of_elas_2 == 0, 1, SE_as_fraction_of_elas_2) ### assuming maximum uncertainty where we interpolated data with an elasticity of 1, i.e. set coefficient of variation to 1.
income_elasticities_SE = SE_as_fraction_of_elas_2 * abs(income_elasticities)
carbon_intensities_2019_estimate = carbon_intensities_2019_estimate.clip(max = 13.89)  ###we clip carbon intensities of consumption to avoid hypersensitivity of model to parameters, only one value, heating and electricity in Belarus, is affected by this.

### compute weighted standard deviation of income elasticities

#def weighted_avg_and_std(values, weights):
 #   """
  #  Return the weighted average and standard deviation.
#
 #   values, weights -- Numpy ndarrays with the same shape.
  #  """
   # average = np.average(values, weights=weights)
    ## Fast and numerically precise:
    #variance = np.average((values-average)**2, weights=weights)
    #return (average, math.sqrt(variance))

#weighted_avg_and_std_elasticities = np.zeros((88,2))
#spread_of_consumption_shares= np.zeros((88,2))
#unity_weight = np.ones((14,1))
#for n in range(1,89):
 #  spread_of_consumption_shares[n-1,:] =  weighted_avg_and_std(weight_for_elasticities[14*n-14:14*n,0], weight_for_elasticities[14*n-14:14*n,0]) 

#%%

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
##################################   BAU - SSP2 standard scenario    ###################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


#### make carbon intensity assumptions for countries which did increase emissions intensity historically, we assume decrease of -0.01 per year for the next decade and the double the pace

standard_em_intensity_evolution = np.where(standard_em_intensity_evolution == -0.01, -0.02, standard_em_intensity_evolution )
standard_em_intensity_evolution = np.where(standard_em_intensity_evolution > 0, -0.01, standard_em_intensity_evolution )
df_BIG_exp_2019_pc_quintiles = pd.DataFrame(data = array_BIG_exp_2019_pc_quintiles, columns = [1,2,3,4,5], index = labels);
standard_em_intensity_evolution_extended = np.zeros((1232, 82)); 
ssp2_pop_growth_extended = np.zeros((1232, 82)); 
ssp2_cons_pc_growth_extended_differentiated_not_normed = np.zeros((1232, 82)); 
ssp2_cons_pc_growth_extended_aggregate = np.zeros((1232, 82)); 

ssp2_cons_pc_growth_extended_differentiated_normed = np.zeros((1232, 82)); 
for i in range(1,89):
    standard_em_intensity_evolution_extended[i*14-14:i*14, :] = np.tile(standard_em_intensity_evolution[i-1,:],(14,1)); 
    ssp2_pop_growth_extended[i*14-14:i*14, :] = np.tile(ssp2_pop_growth[i-1,:],(14,1)); 
    ssp2_cons_pc_growth_extended_differentiated_not_normed[i*14-14:i*14, :] = income_elasticities[i*14-14:i*14, :]*ssp2_cons_pc_growth[i-1,:]
    ssp2_cons_pc_growth_extended_aggregate[i*14-14:i*14, :]  = np.tile(ssp2_cons_pc_growth[i-1,:],(14,1))

################## COMPUTE BASE CASE FORECAST/ BAU/SSP2 SCENARIO ############################
###set up carbon_intensities 2020 to 2101
### the "dummy" year 2101 is created so that in the MAIN SIMULATION (further down) the housing stocks are accordingly updated to 2100. 

carbon_intensities_over_time = np.zeros((1232, 82)); ## begin 2020, end 2101
for i in range(0, len(years)+1):
    if i == 0: 
          carbon_intensities_over_time[:,0] = carbon_intensities_2019_estimate*(1+standard_em_intensity_evolution_extended[:,0])
    else:
          carbon_intensities_over_time[:,i] = carbon_intensities_over_time[:,i-1]*(1+standard_em_intensity_evolution_extended[:,i])
   
### set up population per quintile 2020 to 2101 
#### this population vector is set up so that it can be multiplied directly with consumption per product category. It contains
### population per quintile. 
population_over_time =  np.zeros((1232, 82)); ## begin 2020, end 2101 
for i in range(0, len(years)+1):
    if i == 0: 
          population_over_time[:,0] = pop_quintile_2019*(1+ssp2_pop_growth_extended[:,0])
    else:
          population_over_time[:,i] = population_over_time[:,i-1]*(1+ssp2_pop_growth_extended[:,i])
   
###set up consumption per capita and total consumption 2020 to 2101
#### first compute with aggregate growth rates, so product differentiated product growth rates can be normed on this 

consumption_over_time_aggregate = []
consumption_over_time_aggregate.append(df_BIG_exp_2019_pc_quintiles) 
consumption_over_time_differentiated_not_normed = []
consumption_over_time_differentiated_not_normed.append(df_BIG_exp_2019_pc_quintiles) 
consumption_over_time_differentiated_normed = []
consumption_over_time_differentiated_normed.append(df_BIG_exp_2019_pc_quintiles) 
cons_aggregate_change_over_time = np.zeros((88,82))
cons_differentiated_not_normed_change_over_time = np.zeros((88,82))
cons_differentiated_normed_change_over_time = np.zeros((88,82))

for i in range(0,len(years)+1):
    consumption_over_time_aggregate.append(consumption_over_time_aggregate[i] + consumption_over_time_aggregate[i].multiply(ssp2_cons_pc_growth_extended_aggregate[:,i], axis = 0))
    #consumption_over_time_differentiated_not_normed.append(consumption_over_time_differentiated_not_normed[i] + consumption_over_time_differentiated_not_normed[i].multiply(ssp2_cons_pc_growth_extended_differentiated_not_normed[:,i], axis = 0))
    for j in range(1,89):
        cons_aggregate_change_over_time[j-1, i] = sum(consumption_over_time_aggregate[i].multiply(ssp2_cons_pc_growth_extended_aggregate[:,i], axis = 0)[14*j-14:j*14].sum());
        #cons_differentiated_not_normed_change_over_time[j-1, i] = sum(consumption_over_time_differentiated_not_normed[i].multiply(ssp2_cons_pc_growth_extended_differentiated_not_normed[:,i], axis = 0)[14*j-14:j*14].sum(axis = 1));
    print("iteration is " + str(i))
    
#####extremely important code which creates projected time series of consumption per quintile per cons. category per country. from 2020 to 2100
#### loop over years
for i in range(0,82):
    ### loop over countries
    for j in range(1,89):  
            ### take current consumption over time before correction is applied. list starts with its only element being the starting point 2019 
            test1 = sum(consumption_over_time_differentiated_normed[i].multiply(ssp2_cons_pc_growth_extended_differentiated_not_normed[:,i], axis = 0)[14*j-14:j*14].sum(axis = 1));            
            ### create correction factor (normalization factor)
            test1_1 = cons_aggregate_change_over_time[j-1,i]/test1    
            ### apply correction factor
            test1_2 = consumption_over_time_differentiated_normed[i].multiply(ssp2_cons_pc_growth_extended_differentiated_not_normed[:,i], axis = 0)[14*j-14:j*14]*test1_1  
            #### compute factual normalized but product differentiated growth rate
            test1_3 = test1_2/consumption_over_time_differentiated_normed[i][14*j-14:j*14]       
            ### store growth rates from step above
            ssp2_cons_pc_growth_extended_differentiated_normed[14*j-14:j*14,i] = test1_3.iloc[:,0]
    
    ## add newly normalized but product differeniated projected consumption to list. this element is used in the next iteration as a starting point                
    consumption_over_time_differentiated_normed.append(consumption_over_time_differentiated_normed[i] + consumption_over_time_differentiated_normed[i].multiply(ssp2_cons_pc_growth_extended_differentiated_normed[:,i], axis = 0))
    
### compute projected total cons. and emissions of world 2019 to 2101 for base case scenario (based on ssp2)
ssp2_global_total_cons = np.zeros((1,83))
ssp2_emissions_total = np.zeros((1,83))
ssp2_emissions_total_granular_list = []
for i in range(0,83):
    ssp2_global_total_cons[:,i] = sum(consumption_over_time_differentiated_normed[i].sum())
    if i == 0:
        ssp2_global_total_cons[:,i] = sum(consumption_over_time_differentiated_normed[0].multiply(pop_quintile_2019, axis = 0).sum())
        ssp2_emissions_total[:,i] = sum(consumption_over_time_differentiated_normed[0].multiply(pop_quintile_2019, axis = 0).multiply(carbon_intensities_2019_estimate, axis = 0).sum());
        ssp2_emissions_total_granular_list.append(consumption_over_time_differentiated_normed[0].multiply(pop_quintile_2019, axis = 0).multiply(carbon_intensities_2019_estimate, axis = 0))
    else:
        ssp2_global_total_cons[:,i] = sum(consumption_over_time_differentiated_normed[i].multiply(population_over_time[:,i-1], axis = 0).sum())
        ssp2_emissions_total[:,i] =  sum(consumption_over_time_differentiated_normed[i].multiply(population_over_time[:,i-1], axis = 0).multiply(carbon_intensities_over_time[:,i-1], axis = 0).sum());
        ssp2_emissions_total_granular_list.append(consumption_over_time_differentiated_normed[i].multiply(population_over_time[:,i-1], axis = 0).multiply(carbon_intensities_over_time[:,i-1], axis = 0))

    
#### compute emission share of global top 1%     

labels2 = labels[0::14][:,0:3]    
population_over_time2 = population_over_time[0::14] ### population per country (country = rows) per quintile per year (yr = columns)
ssp2_dict_pc_em_quintiles = OrderedDict()
ssp2_dict_total_em_quintiles = OrderedDict()

#### https://stackoverflow.com/questions/6181935/how-do-you-create-different-variable-names-while-in-a-loop
for i in range(1,83):
        ssp2_dict_pc_em_quintiles["{0}".format(2019+i)] = pd.DataFrame(columns = [1,2,3,4,5], index = labels2);
        ssp2_dict_total_em_quintiles["{0}".format(2019+i)] = pd.DataFrame(columns = [1,2,3,4,5], index = labels2);
        
        for j in range(1,89):
            ssp2_dict_pc_em_quintiles[str(2019+i)].iloc[j-1,:] = consumption_over_time_differentiated_normed[i].multiply(carbon_intensities_over_time[:,i-1], axis = 0).iloc[j*14-14:j*14].sum();
        
        ssp2_dict_total_em_quintiles[str(2019+i)] = ssp2_dict_pc_em_quintiles[str(2019+i)].multiply(population_over_time2[:,i-1], axis = 0)
                                       
        print("iteration is " + str(i))

population_over_time3 = np.zeros((88*5,82))
for i in range(1,89):
      for j in range(0,82):
         population_over_time3[i*5-5:i*5, j] = population_over_time[i*14-14:i*14-9, j]

### tuple list starts at t = 2020, not at t_0 = 2019 see code below 2019 + i, i element (1,82)
tuple_list1 = []
for i in range(1,83):        
        step1 = ssp2_dict_pc_em_quintiles[str(2019+i)].stack()
        step2 = ssp2_dict_total_em_quintiles[str(2019+i)].stack()
        step3 = population_over_time3[:,i-1]            
        step4 = pd.concat([step1, step2], axis = 1, ignore_index = True)
        step4.insert(2, "2", step3, True)       
        #step5 = step4.sort_values(0)
        resulttuple = gini_dynamic(step4.iloc[:,2], step4.iloc[:,1])
        tuple_list1.append(resulttuple)
        print("iteration is "+ str(i))

### concatenate all Gini measurements to one array
ssp2_Gini_array = np.zeros((82,1))
for i in range(1,82):
   ssp2_Gini_array[i-1] = tuple_list1[i-1][0]    
   
### ALGORITHM TO SMOOTHLY INTERPOLATE CHUNKY LORENZ CURVE, WITH LINEAR INTERPOLATION SO THAT EXACT cumulative cut off for
### top 1% can be found.

sign_arr_BIG = np.zeros((441,82))

for j in range(0,len(tuple_list1)):
        dist_to_99 = tuple_list1[j][1]-0.99
        sign_arr = np.zeros((1,len(dist_to_99)))
        
        for i in range(1,len(dist_to_99)-1):
             if (i == 0) or (i == len(dist_to_99)-1):
                  pass 
             else:
                     if np.sign(dist_to_99[i-1]) == np.sign(dist_to_99[i+1]):
                         sign_arr[:,i] = 0
                     else:
                         sign_arr[:,i] = 1
        sign_arr_BIG[:,j] = np.squeeze(np.transpose(sign_arr))
                 
#### extract data on to be (linearly) interpolated lorenz curve sections
data_arr = np.zeros((164,2)) ### always 2x2 elements belong together
index = np.expand_dims(np.linspace(0,440, 441), axis = 1)
index_arr = np.multiply(sign_arr_BIG, index)
helparr1 = np.floor((np.sum(index_arr, axis = 0)/2))
helparr2 = np.ceil((np.sum(index_arr, axis = 0)/2))
index_arr_new  = index_arr[index_arr  > 0]

for j in range(1,len(tuple_list1)+1):
    data_arr[j*2-2,0] = tuple_list1[j-1][1][int(helparr1[j-1])]
    data_arr[j*2-1,0] = tuple_list1[j-1][1][int(helparr2[j-1])]
    data_arr[j*2-2,1] = tuple_list1[j-1][2][int(helparr1[j-1])]
    data_arr[j*2-1,1] = tuple_list1[j-1][2][int(helparr2[j-1])]
    
share_99_arr_cum = np.zeros((83,1))   
for j in range(1,83):
    result_fit = lin_fit_non_log(data_arr[j*2-2:j*2,0], data_arr[j*2-2:j*2,1])
    share_99_arr_cum[j-1,0] = result_fit[0][0]+ result_fit[0][1]*0.99

share_top1_arr = 1 - share_99_arr_cum

#############
#### compute share of luxury and basic emissions over time

luxury = np.where(income_elasticities  <= 1, 0, income_elasticities )
luxury = np.where(luxury > 0, 1, luxury)

basic = np.where(income_elasticities  <= 1, 1, income_elasticities )
basic = np.where(basic > 1 , 0, basic)

luxury_emissions_over_time = np.zeros((83,1))
basic_emissions_over_time = np.zeros((83,1))
for i in range(0,83):
     luxury_emissions_over_time[i,:] = sum((ssp2_emissions_total_granular_list[i]*luxury).sum())/ssp2_emissions_total[:,i]
     basic_emissions_over_time[i,:] = sum((ssp2_emissions_total_granular_list[i]*basic).sum())/ssp2_emissions_total[:,i]
     
        
#### compute households over time for non-BAU scenarios already
#### using a constant ratio of households to population and assuming that households are equal to dwellings(which they are not but they are very strongly correlated)
     
### this population over time per country (total population, not quintile number like population_over_time2)
population_over_time4 = population_over_time2 * 5

households_over_time = np.zeros((88,82));
for i in range(0,82):
    households_over_time[:,i] = population_over_time4[:,i]*households_per_capita
    
households_over_time_quintile = households_over_time/5    


##### calculate energy intensities over time for retrofit investment scenario. take very simple CAGR of -1.1% for all energy categories but in line with the 
#### overall global energy intensity evolution in SSP2

final_energy_intensities_over_time = np.zeros((1232,82))

for i in range(0,82):
    if i == 0:
       final_energy_intensities_over_time[:,i] = np.squeeze(final_energy_intensities_estimate_2019*(1-0.011))
    else:
       final_energy_intensities_over_time[:,i] = final_energy_intensities_over_time[:,i-1]*(1-0.011)
       
###### create and plot SSP2 BAU total final energy of households 2020 to 2101
final_energy_over_time_BAU = []
final_energy_over_time_total = np.zeros((1,82))
for i in range(1,83):
    final_energy_over_time_BAU.append(consumption_over_time_differentiated_normed[i].multiply(population_over_time[:,i-1], axis = 0).multiply(final_energy_intensities_over_time[:,i-1], axis = 0))
    final_energy_over_time_total[:,i-1] = sum(consumption_over_time_differentiated_normed[i].multiply(population_over_time[:,i-1], axis = 0).multiply(final_energy_intensities_over_time[:,i-1], axis = 0).sum())
    
    
######## plot ########

plt.plot(years, np.squeeze(final_energy_over_time_total/10**12)[:-1] )   
plt.ylabel("EJ/yr")
plt.show()
    
###### calculate costs vector for retrofitting dependent on country (same like in static model line 2125 etc.)

costs_per_megajoule_retrofit_differentiated = np.zeros((88,1));
for i in range(1,89):
    if meta_data_countries[i-1,7] == 'North':
        costs_per_megajoule_retrofit_differentiated[i-1,0] = 0.77
    else: 
        costs_per_megajoule_retrofit_differentiated[i-1,0] = 0.77*0.55
        
        
### calculate baseline residential emissions (that is residential heat and electr. use)
### because it is important for the retrofit policy to compare baseline (without any retrofit) vs. with retrofits over time

##########################################################
##########################################################
##########################################################
##########################################################
##### PLOT ALL CATEGORIES EMISSION EVOLUTION BAU #########
##########################################################
##########################################################
##########################################################
##########################################################

category_labels = copy.deepcopy(labels[0:14,3])
for i in range(0,14): category_labels[i] = category_labels[i][:-4]


rows = 4; cols = 4;
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 12), squeeze=0, sharex=True, sharey=True)
axes = np.array(axes)

category_emissions = np.zeros((82,1))

for c, ax in enumerate(axes.reshape(-1)):
  if c < 14:
          ax.set_ylabel("GT/yr")
          ax.set_title(category_labels[c])
          for i in range(1,83):
              category_emissions[i-1] = sum(ssp2_emissions_total_granular_list[i-1][c::14].sum())/10**12
          ax.plot(years, category_emissions[1:], label = "test")  ### from 2020 to 2100
  else:
          pass 
plt.show()    
#plt.savefig('emissions_over_time_categories.png',bbox_inches = "tight", dpi = 300);    


########################################################################################################################################
########################################################################################################################################
##################################################### END ##############################################################################
########################################################################################################################################
############################## END    BAU - SSP2 standard scenario   END ###############################################################
########################################################################################################################################
##################################################### END ##############################################################################
########################################################################################################################################
########################################################################################################################################
     
#%%  

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
##################################   scenario #2 = scen_2 (parameters are flexible so different scenarios can be created ###################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


retrofit_impact = 0.5 ### variable that determines how much a deep retrofit is going to abate. the literature says around 50%
### 0.5 = 50%, 0.1 = 10%, 0.9 = 90% 
### e.g. Less and Walker 2014 but this parameter is extremely uncertain especially around the globe. 
# we just assume deep retrofit impact of X %

scen2_consumption_over_time = []
scen2_consumption_over_time.append(consumption_over_time_differentiated_normed[0])
scen2_consumption_over_time.append(consumption_over_time_differentiated_normed[1])
scen2_consumption_over_time.append(consumption_over_time_differentiated_normed[2])


###inititate retrofitted consumption profile of retrofitted households
consumption_over_time_differentiated_normed_0_retrofits = copy.deepcopy(consumption_over_time_differentiated_normed[0])
consumption_over_time_differentiated_normed_1_retrofits = copy.deepcopy(consumption_over_time_differentiated_normed[1])
consumption_over_time_differentiated_normed_2_retrofits = copy.deepcopy(consumption_over_time_differentiated_normed[2])
consumption_over_time_differentiated_normed_0_retrofits[4::14] = consumption_over_time_differentiated_normed_0_retrofits[4::14]*(1-retrofit_impact)
consumption_over_time_differentiated_normed_1_retrofits[4::14] = consumption_over_time_differentiated_normed_1_retrofits[4::14]*(1-retrofit_impact)
consumption_over_time_differentiated_normed_2_retrofits[4::14] = consumption_over_time_differentiated_normed_2_retrofits[4::14]*(1-retrofit_impact)
scen2_consumption_over_time_retrofits = []
scen2_consumption_over_time_retrofits.append(consumption_over_time_differentiated_normed_0_retrofits)
scen2_consumption_over_time_retrofits.append(consumption_over_time_differentiated_normed_1_retrofits)
scen2_consumption_over_time_retrofits.append(consumption_over_time_differentiated_normed_2_retrofits)



#### set up tax revenue collection over time for control tracking

scen2_revenue_over_time = []
scen2_revenue_over_time_retrofits = []


scen2_revenue_over_time.append(pd.DataFrame(data = 0, columns = [1,2,3,4,5], index = labels))
scen2_revenue_over_time.append(pd.DataFrame(data = 0, columns = [1,2,3,4,5], index = labels))
scen2_revenue_over_time.append(pd.DataFrame(data = 0, columns = [1,2,3,4,5], index = labels))


scen2_revenue_over_time_retrofits.append(pd.DataFrame(data = 0, columns = [1,2,3,4,5], index = labels))
scen2_revenue_over_time_retrofits.append(pd.DataFrame(data = 0, columns = [1,2,3,4,5], index = labels))
scen2_revenue_over_time_retrofits.append(pd.DataFrame(data = 0, columns = [1,2,3,4,5], index = labels))


#### first thing set up carbon price time series 
### including for each country, carbon price can be constant, linearly growing or exponentially growing, focus on exponential increase in price

###### MAIN PARAMETER SET UP ######
###################################
low_income_country_price = 5*2
lower_middle_income_country_price = 12.5*2
upper_middle_income_country_price = 25*2
high_income_country_price = 75*2

low_income_country_price_2100 = 500*2
lower_middle_income_country_price_2100 = 500*2
upper_middle_income_country_price_2100 = 500*2
high_income_country_price_2100 = 1000*2

per_product_yes_no = "yes" ### yes or no as option
carbon_price_regime = "linear" ### exponential or linear as option
retrofit_policy = "yes" ### yes or no 
redistribution_policy = "yes" ## yes or no
tech_bounce_back = "no" ## yes or no
low_growth = "no" ## yes or no


###################################

##### SET UP EXPONENTIAL CARBON PRICE EVOLUTION #####

carbon_price_over_time_per_country_exponential = np.zeros((88,82)) ### 82 because 2020 to including 2101

for i in range(0,88):
    if meta_data_countries[i,1] == "Low income": carbon_price_over_time_per_country_exponential[i,0] = low_income_country_price
    if meta_data_countries[i,1] == "Lower middle income": carbon_price_over_time_per_country_exponential[i,0] = lower_middle_income_country_price
    if meta_data_countries[i,1] == "Upper middle income": carbon_price_over_time_per_country_exponential[i,0] = upper_middle_income_country_price
    if meta_data_countries[i,1] == "High income": carbon_price_over_time_per_country_exponential[i,0] = high_income_country_price 

###1/t for compound growth rate formula and t = 80 because we compound from 2020 to 2100
carbon_price_growth_rates_country_type_exp = np.zeros((4,1))
carbon_price_growth_rates_country_type_exp[0] = (low_income_country_price_2100/low_income_country_price)**(1/80)-1
carbon_price_growth_rates_country_type_exp[1] = (lower_middle_income_country_price_2100/lower_middle_income_country_price)**(1/80)-1
carbon_price_growth_rates_country_type_exp[2] = (upper_middle_income_country_price_2100/upper_middle_income_country_price)**(1/80)-1
carbon_price_growth_rates_country_type_exp[3] = (high_income_country_price_2100 /high_income_country_price)**(1/80)-1

for i in range(0,88):
    for j in range(0,81):
            if meta_data_countries[i,1] == "Low income": carbon_price_over_time_per_country_exponential[i,j+1] = carbon_price_over_time_per_country_exponential[i,j]*(1+carbon_price_growth_rates_country_type_exp[0])
            if meta_data_countries[i,1] == "Lower middle income": carbon_price_over_time_per_country_exponential[i,j+1] = carbon_price_over_time_per_country_exponential[i,j]*(1+carbon_price_growth_rates_country_type_exp[1])
            if meta_data_countries[i,1] == "Upper middle income": carbon_price_over_time_per_country_exponential[i,j+1] = carbon_price_over_time_per_country_exponential[i,j]*(1+carbon_price_growth_rates_country_type_exp[2])
            if meta_data_countries[i,1] == "High income": carbon_price_over_time_per_country_exponential[i,j+1] = carbon_price_over_time_per_country_exponential[i,j]*(1+carbon_price_growth_rates_country_type_exp[3])

carbon_price_CHANGE_over_time_per_country_exponential = np.zeros((88,81))
carbon_price_CHANGE_over_time_per_country_exponential[:,0] = carbon_price_over_time_per_country_exponential[:,0]

for i in range(0,88):
    for j in range(1,81):
        carbon_price_CHANGE_over_time_per_country_exponential[i,j] = carbon_price_over_time_per_country_exponential[i,j] - carbon_price_over_time_per_country_exponential[i,j-1]

##### SET UP LINEAR CARBON PRICE EVOLUTION #####

carbon_price_over_time_per_country_linear = np.zeros((88,82)) ### 8

for i in range(0,88):
    if meta_data_countries[i,1] == "Low income": carbon_price_over_time_per_country_linear[i,0] = low_income_country_price
    if meta_data_countries[i,1] == "Lower middle income": carbon_price_over_time_per_country_linear[i,0] = lower_middle_income_country_price
    if meta_data_countries[i,1] == "Upper middle income": carbon_price_over_time_per_country_linear[i,0] = upper_middle_income_country_price
    if meta_data_countries[i,1] == "High income": carbon_price_over_time_per_country_linear[i,0] = high_income_country_price 

###1/t for linear growth rate formula and t = 80 because we compound from 2020 to 2100
carbon_price_growth_rates_country_type_linear = np.zeros((4,1))
carbon_price_growth_rates_country_type_linear[0] = (low_income_country_price_2100-low_income_country_price)/78
carbon_price_growth_rates_country_type_linear[1] = (lower_middle_income_country_price_2100-lower_middle_income_country_price)/78
carbon_price_growth_rates_country_type_linear[2] = (upper_middle_income_country_price_2100-upper_middle_income_country_price)/78
carbon_price_growth_rates_country_type_linear[3] = (high_income_country_price_2100-high_income_country_price)/78


for i in range(0,88):
    for j in range(0,81):
            if meta_data_countries[i,1] == "Low income": carbon_price_over_time_per_country_linear[i,j+1] = carbon_price_over_time_per_country_linear[i,j]+carbon_price_growth_rates_country_type_linear[0]
            if meta_data_countries[i,1] == "Lower middle income": carbon_price_over_time_per_country_linear[i,j+1] = carbon_price_over_time_per_country_linear[i,j]+carbon_price_growth_rates_country_type_linear[1]
            if meta_data_countries[i,1] == "Upper middle income": carbon_price_over_time_per_country_linear[i,j+1] = carbon_price_over_time_per_country_linear[i,j]+carbon_price_growth_rates_country_type_linear[2]
            if meta_data_countries[i,1] == "High income": carbon_price_over_time_per_country_linear[i,j+1] = carbon_price_over_time_per_country_linear[i,j]+carbon_price_growth_rates_country_type_linear[3]

carbon_price_CHANGE_over_time_per_country_linear = np.zeros((88,81))
carbon_price_CHANGE_over_time_per_country_linear[:,0] = carbon_price_over_time_per_country_linear[:,0]

for i in range(0,88):
    for j in range(1,81):
        carbon_price_CHANGE_over_time_per_country_linear[i,j] = carbon_price_over_time_per_country_linear[i,j] - carbon_price_over_time_per_country_linear[i,j-1]


#%%
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
##################################   MAIN SIMULATION      ###################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
### prepare bounce-back effect similar to excel toy model principle


### make square array for tax decay effect per category over time, time is on both dimensions/axes from 2020-2100
square_array = np.zeros((81, 81));
carbon_intensities_over_time;
carbon_price_CHANGE_over_time_per_country_linear;
list_of_square_arrays = []

for j in range(0,1232):
      square_array = np.zeros((81, 81));
      for i in range(0,len(square_array)):
            ### -1 for carbon intensities because they go till 2101 not just 2100. 
           square_array[i,i:] = carbon_price_CHANGE_over_time_per_country_linear[math.floor(j/14),i]*carbon_intensities_over_time[j,i:-1] 
      list_of_square_arrays.append(square_array)

### the list of square arrays is a list of data related to all 1232 consumption categories (14 types *88 countries)
### every list item contains a square matrix (or array) that the fundamentals for calculating the diminishing/decay of the tax impact over time
### if you had a world without economic growth let us say you consume 1000 $PPP for sth. then the tax reduces this to 850. and there is no further taxation 
### then, if technoloyg does not change i.e. carbon intensity stays constant, nothing happens
### but if carbon intensity changes/reduces over time, you expect the tax effects from the past to diminish and consumption to bounce back


### set up storage matrix/array for hypothetical consumption over time with future carbon intensities
consumption_decay_tax_rate_big_array = np.zeros((14*80,80*5))

stored_multiplier_bounce_back_effect = []


consumption_decay_tax_rate_big_array_list = []
consumption_decay_tax_rate_big_array_list_retrofits = []
for i in range(0,88):
    consumption_decay_tax_rate_big_array_list.append(copy.deepcopy(consumption_decay_tax_rate_big_array))
    consumption_decay_tax_rate_big_array_list_retrofits.append(copy.deepcopy(consumption_decay_tax_rate_big_array))
    stored_multiplier_bounce_back_effect.append(copy.deepcopy(consumption_decay_tax_rate_big_array))

#for y in range(1,15):
 #       for k in range(1,6):      
  #                  df_price_elasticities.iloc[y-1,k-1] * list_of_square_arrays[j*14-15+y][:i-1,i-2]
   #                 list_of_square_arrays[j*14-15+y][:i-1,i-2]
            



#tax_revenue_granular 

def foo(x,y):
    column_number = int(x.size/len(x))
    row_number = len(x)
    z = np.zeros((len(x), column_number));
    for i in range(0,len(x)):
        for j in range(0, column_number):
            try:
                z[i,j] = x.iloc[i,j]/y.iloc[i,j]
            except ZeroDivisionError:         
                z[i,j]  = 0 
    return z

### IMPORTANT INFO!!!!!!!! #### we need to distinguish between retrofits and non retrofits, So basically the whole simulation will run for 2 different types of households
### the once that already received retrofits and the ones that did not. thus most variables need intitation across both types

physical_cons_time_arr = np.zeros((88,83))
tax_revenue_over_time_arr = np.zeros((88,83))

physical_cons_time_arr_retrofits = np.zeros((88,83))
tax_revenue_over_time_arr_retrofits = np.zeros((88,83))


### track the available budget over years and countries for control purposes

budget_available_over_time = np.zeros((88,83))
budget_enough_for_redistribution = np.zeros((88,83))

###this array is supposed to record (as a memory array) how many dwellings/houses still have to be retrofitted
### it depends on the households that are present minus the ones that are covered by the budget and also the new households that originate every year. 

households_not_retrofitted_yet = np.zeros((88*5,83)) 
households_retrofitted_already = np.zeros((88*5,83)) 


### track average national consumption per capita and national emissions gini coefficient
### as control for how the policies impact society

average_pc_cons_over_time = np.zeros((88,83))
national_gini_coefficient_over_time = np.zeros((88,83))


###### preparation for redistribution policy #########

###translating Gini_cons_national into national_zero_trade_off_point
#### zero trade off points will be the ones from static model and not change dynamically throughout the simulation but be adopted and fixed

zero_trade_off_points_modelled = 4.7 + Gini_cons_national * 100 

### now round zero_trade_off_points_modelled to nearest quintile. Because points are in percentiles while in the dynamic simulatio
### only quintiles are modelled 

zero_trade_off_points_modelled_quintiles =np.round(zero_trade_off_points_modelled/20)

budget_available = 0 ### budget needs to be intialized once, because it is now defined iteratively in the beginning of
### revenue recycling block in main simulation

if low_growth == "yes": ssp2_cons_pc_growth_extended_aggregate = ssp2_cons_pc_growth_extended_aggregate/2

print('main simulation')
#####extremely important code which creates projected time series of consumption per quintile per cons. category per country. from 2020 to 2100
#### loop over years
for i in range(2,82): #### because starting in 2021, first thing it does is starting from 2021 and calculating the year 2022... until 2100 is computed. 
#### so last run through is the index for the year 2099
    #### set up yearly dataframe

    df_yearly = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
    df_yearly_tax_revenue = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
    if retrofit_policy == "yes": 
        df_yearly_retrofits = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
        df_yearly_tax_revenue_retrofits = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
    
    ### loop over countries
    for j in range(1,89):  
                
            ####################################
            ##### BLOCK #1 ECONOMIC GROWTH #####
            ####################################
        
            ### NOT YET RETROFITTED HOUSEHOLDS ####
            
            ### take current consumption over time before correction is applied. list starts with its only element being the starting point 2019 
            step1 = sum(scen2_consumption_over_time[i].multiply(ssp2_cons_pc_growth_extended_differentiated_not_normed[:,i], axis = 0)[14*j-14:j*14].sum(axis = 1));            
            step2 = sum(scen2_consumption_over_time[i].multiply(ssp2_cons_pc_growth_extended_aggregate[:,i], axis = 0)[14*j-14:j*14].sum());
            ### create correction factor (normalization factor)
            step3 = step2/step1  
            ### apply correction factor
            step4 = scen2_consumption_over_time[i].multiply(ssp2_cons_pc_growth_extended_differentiated_not_normed[:,i], axis = 0)[14*j-14:j*14]*step3
            #### compute factual normalized product differentiated growth rate before tax
            step5 = step4/scen2_consumption_over_time[i][14*j-14:j*14]      
            #### compute factual normalized product differentiated consumption after growth BEFORE TAX
            step6 = step4 + scen2_consumption_over_time[i][14*j-14:j*14] 
            
            
            ### ALREADY RETROFITTED HOUSEHOLDS ####
            if retrofit_policy == "yes":
                                            
            ### take current consumption over time before correction is applied. list starts with its only element being the starting point 2019 
                step1_retrofits = sum(scen2_consumption_over_time_retrofits[i].multiply(ssp2_cons_pc_growth_extended_differentiated_not_normed[:,i], axis = 0)[14*j-14:j*14].sum(axis = 1));            
                step2_retrofits = sum(scen2_consumption_over_time_retrofits[i].multiply(ssp2_cons_pc_growth_extended_aggregate[:,i], axis = 0)[14*j-14:j*14].sum());
                ### create correction factor (normalization factor)
                step3_retrofits = step2_retrofits/step1_retrofits  
                ### apply correction factor
                step4_retrofits = scen2_consumption_over_time_retrofits[i].multiply(ssp2_cons_pc_growth_extended_differentiated_not_normed[:,i], axis = 0)[14*j-14:j*14]*step3
                #### compute factual normalized product differentiated growth rate before tax
                step5_retrofits = step4_retrofits/scen2_consumption_over_time_retrofits[i][14*j-14:j*14]      
                #### compute factual normalized product differentiated consumption after growth BEFORE TAX
                step6_retrofits = step4_retrofits + scen2_consumption_over_time_retrofits[i][14*j-14:j*14] 
                
            
            #### compute factual normalized product differentiated consumption after growth AFTER TAX --> next blocks
            
            ################################################
            ##### BLOCK #2 PRICE ELASTICITY ESTIMATION #####
            ################################################
            
            
            ### NOT YET RETROFITTED HOUSEHOLDS ####
            
            #### whole df_price elasticity procedure first
            #### compute budget share
            #### make price elasticity dataframe #### based on Sabetelli 2016 mapping model, method section 
            roh = -1.26 + np.random.normal(0,0.05) #### elasticity of the marginal utility of income, #### one std ~0.05 based on layard et al. 95% CI
            #roh_upper_bound = -1.19
            #roh_lower_bound = -1.34 a
            ####according to Laynard et al. 2008 so therefore 0.1~ 2 standard deviations i.e. 95% CI of normal dist.
            step7_budget_share = step6/step6.sum(axis = 0)
            df_first_term_0 = (-1/roh)*step7_budget_share 
            
            ## no distinction between retrofit and non-retrofitted households in terms of income elasticities and their stochastic terms
            income_elasticity_error = np.random.normal(0,income_elasticities_SE[14*j-14:j*14])
            
            df_first_term = df_first_term_0.multiply((income_elasticities[14*j-14:j*14]+income_elasticity_error)**2, axis = 1)
            df_second_term_0 = (1/roh)-step7_budget_share 
            df_second_term = df_second_term_0.multiply(income_elasticities[14*j-14:j*14]+income_elasticity_error, axis = 1)
            df_price_elasticities = df_first_term + df_second_term
            

            ### ALREADY RETROFITTED HOUSEHOLDS ####
            if retrofit_policy == "yes":
                                     
            #roh = -1.26 + np.random.normal(0,0.05) #### elasticity of the marginal utility of income, #### one std ~0.05 based on layard et al. 95% CI
            #roh_upper_bound = -1.19
            #roh_lower_bound = -1.34 a
            ####according to Laynard et al. 2008 so therefore 0.1~ 2 standard deviations i.e. 95% CI of normal dist.
                step7_budget_share_retrofits = step6_retrofits/step6_retrofits.sum(axis = 0)
                df_first_term_0_retrofits = (-1/roh)*step7_budget_share_retrofits 
                df_first_term_retrofits = df_first_term_0_retrofits.multiply((income_elasticities[14*j-14:j*14]+income_elasticity_error)**2, axis = 1)
                df_second_term_0_retrofits = (1/roh)-step7_budget_share_retrofits 
                df_second_term_retrofits = df_second_term_0_retrofits.multiply(income_elasticities[14*j-14:j*14]+income_elasticity_error, axis = 1)
                df_price_elasticities_retrofits = df_first_term_retrofits + df_second_term_retrofits
                

            
            ########################################
            #### BLOCK #3 TAX RATE CALCULATION #####
            ########################################
            
            ### NOT YET RETROFITTED HOUSEHOLDS ####
            
            #### per product differentiated yes or no ####
            
            ### decision on price evolution regime
            if carbon_price_regime == "exponential":
            
                    ### decision on taxation regime (luxury or uniform pricing) 
                    if per_product_yes_no == "yes":
                        ### divide price by 1000 to go from price per ton to price per kg
                        price_increase_through_tax = carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * income_elasticities[14*j-14:j*14] 
                        #### compute embodied carbon tax without and with price adjustment so that the *average* carbon costs are preserved
                        embodied_carbon_costs_luxury = sum((carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * income_elasticities[14*j-14:j*14] * step6).sum()) 
                        embodied_carbon_costs_blanket = sum((carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * step6).sum()) 
                        
                        correction_factor = embodied_carbon_costs_blanket/embodied_carbon_costs_luxury
                        ### corrected price increase_through tax luxury scenario
                        price_increase_through_tax = price_increase_through_tax*correction_factor
                        
                        ### -2 in index i because starting in 2021 see line 424
                    elif per_product_yes_no == "no":
                        price_increase_through_tax = carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1)
                                   
                    decreased_consumption = (1+price_increase_through_tax*df_price_elasticities)
                    
            elif carbon_price_regime == "linear":
                                
                    if per_product_yes_no == "yes":
                        ### divide price by 1000 to go from price per ton to price per kg
                        price_increase_through_tax = carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * income_elasticities[14*j-14:j*14] 
                        #### compute embodied carbon tax without and with price adjustment so that the *average* carbon costs are preserved
                        embodied_carbon_costs_luxury = sum((carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * income_elasticities[14*j-14:j*14] * step6).sum()) 
                        embodied_carbon_costs_blanket = sum((carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * step6).sum()) 
                        
                        correction_factor = embodied_carbon_costs_blanket/embodied_carbon_costs_luxury
                        ### corrected price increase_through tax luxury scenario
                        price_increase_through_tax = price_increase_through_tax*correction_factor
                        
                        ### -2 in index i because starting in 2021 see line 424
                        
                    elif per_product_yes_no == "no":
                        price_increase_through_tax = carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1)
                                   
                    decreased_consumption = (1+price_increase_through_tax*df_price_elasticities)
                
            ### ALREADY RETROFITTED HOUSEHOLDS ####
            if retrofit_policy == "yes":
                                     
            #### per product differentiated yes or no ####
                
                ### decision on price evolution regime
                if carbon_price_regime == "exponential":
                
                        ### decision on taxation regime (luxury or uniform pricing) 
                        if per_product_yes_no == "yes":
                            ### divide price by 1000 to go from price per ton to price per kg
                            price_increase_through_tax = carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * income_elasticities[14*j-14:j*14] 
                            #### compute embodied carbon tax without and with price adjustment so that the *average* carbon costs are preserved
                            embodied_carbon_costs_luxury = sum((carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * income_elasticities[14*j-14:j*14] * step6_retrofits).sum()) 
                            embodied_carbon_costs_blanket = sum((carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * step6_retrofits).sum()) 
                            
                            correction_factor = embodied_carbon_costs_blanket/embodied_carbon_costs_luxury
                            ### corrected price increase_through tax luxury scenario
                            price_increase_through_tax = price_increase_through_tax*correction_factor
                            
                            ### -2 in index i because starting in 2021 see line 424
                        elif per_product_yes_no == "no":
                            price_increase_through_tax = carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1)
                                       
                        decreased_consumption_retrofits = (1+price_increase_through_tax*df_price_elasticities_retrofits)
                        
                elif carbon_price_regime == "linear":
                                    
                        if per_product_yes_no == "yes":
                            ### divide price by 1000 to go from price per ton to price per kg
                            price_increase_through_tax = carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * income_elasticities[14*j-14:j*14] 
                            #### compute embodied carbon tax without and with price adjustment so that the *average* carbon costs are preserved
                            embodied_carbon_costs_luxury = sum((carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * income_elasticities[14*j-14:j*14] * step6_retrofits).sum()) 
                            embodied_carbon_costs_blanket = sum((carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1) * step6_retrofits).sum()) 
                            
                            correction_factor = embodied_carbon_costs_blanket/embodied_carbon_costs_luxury
                            ### corrected price increase_through tax luxury scenario
                            price_increase_through_tax = price_increase_through_tax*correction_factor
                            
                            ### -2 in index i because starting in 2021 see line 424
                            
                        elif per_product_yes_no == "no":
                            price_increase_through_tax = carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1)
                                       
                        decreased_consumption_retrofits = (1+price_increase_through_tax*df_price_elasticities_retrofits)
            
 
        
            
            ####################################################################################################
            #### BLOCK #4 HYPOTHETICAL TAX RATE CALCULATION WITH TECH from t+n for tax rate decay factor #######
            ####################################################################################################
            
            ### the list of square arrays is a list of data related to all 1232 consumption categories (14 types *88 countries)
            ### every list item contains a square matrix (or array) that the fundamentals for calculating the diminishing/decay of the tax impact over time
            ### if you had a world without economic growth let us say you consume 1000 $PPP for sth. then the tax reduces this to 850. and there is no further taxation 
            ### then, if technoloyg does not change i.e. carbon intensity stays constant, nothing happens
            ### but if carbon intensity changes/reduces over time, you expect the tax effects from the past to diminish and consumption to bounce back

            if tech_bounce_back == "yes":
                
                for k in range(i-1,81):
                        #print("iteration is " + str(k))
                        ### NOT YET RETROFITTED HOUSEHOLDS ####                                    
                        ### decision on price evolution regime
                        if carbon_price_regime == "exponential":
                                
                    
                                    ### decision on taxation regime (luxury or uniform pricing) 
                                if per_product_yes_no == "yes":
                                    ### divide price by 1000 to go from price per ton to price per kg
                                    price_increase_through_tax_HYPO = carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * income_elasticities[14*j-14:j*14] 
                                    #### compute embodied carbon tax without and with price adjustment so that the *average* carbon costs are preserved
                                    embodied_carbon_costs_luxury_HYPO = sum((carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * income_elasticities[14*j-14:j*14] * step6).sum()) 
                                    embodied_carbon_costs_blanket_HYPO = sum((carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * step6).sum()) 
                                        
                                    correction_factor_HYPO = embodied_carbon_costs_blanket_HYPO/embodied_carbon_costs_luxury_HYPO
                                        ### corrected price increase_through tax luxury scenario
                                    price_increase_through_tax_HYPO = price_increase_through_tax_HYPO*correction_factor_HYPO
                                        
                                        ### -2 in index i because starting in 2021 see line 424
                                elif per_product_yes_no == "no":
                                    price_increase_through_tax_HYPO = carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1)
                                                   
                                decreased_consumption_HYPO = (1+price_increase_through_tax_HYPO*df_price_elasticities)
                                    
                                consumption_decay_tax_rate_big_array_list[j-1][14*(i-1)-14:14*(i-1),5*k-5:5*k] =  decreased_consumption_HYPO
                                
                        if carbon_price_regime == "linear":
                                            
                                if per_product_yes_no == "yes":
                                    ### divide price by 1000 to go from price per ton to price per kg
                                    price_increase_through_tax_HYPO = carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * income_elasticities[14*j-14:j*14] 
                                    #### compute embodied carbon tax without and with price adjustment so that the *average* carbon costs are preserved
                                    embodied_carbon_costs_luxury_HYPO = sum((carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * income_elasticities[14*j-14:j*14] * step6).sum()) 
                                    embodied_carbon_costs_blanket_HYPO = sum((carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * step6).sum()) 
                                    
                                    correction_factor_HYPO = embodied_carbon_costs_blanket_HYPO/embodied_carbon_costs_luxury_HYPO
                                    ### corrected price increase_through tax luxury scenario
                                    price_increase_through_tax_HYPO = price_increase_through_tax_HYPO*correction_factor_HYPO
                                    
                                    ### -2 in index i because starting in 2021 see line 424
                                    
                                elif per_product_yes_no == "no":
                                    price_increase_through_tax_HYPO = carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1)
                                               
                                decreased_consumption_HYPO = (1+price_increase_through_tax_HYPO*df_price_elasticities)
                                
                                consumption_decay_tax_rate_big_array_list[j-1][14*(i-1)-14:14*(i-1),5*k-5:5*k] =  decreased_consumption_HYPO
                            
                        ### ALREADY RETROFITTED HOUSEHOLDS ####
                        if retrofit_policy == "yes":
                                                 
                        #### per product differentiated yes or no ####
                            
                            ### decision on price evolution regime
                            if carbon_price_regime == "exponential":
                            
                                    ### decision on taxation regime (luxury or uniform pricing) 
                                    if per_product_yes_no == "yes":
                                        ### divide price by 1000 to go from price per ton to price per kg
                                        price_increase_through_tax_HYPO = carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * income_elasticities[14*j-14:j*14] 
                                        #### compute embodied carbon tax without and with price adjustment so that the *average* carbon costs are preserved
                                        embodied_carbon_costs_luxury_HYPO = sum((carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * income_elasticities[14*j-14:j*14] * step6_retrofits).sum()) 
                                        embodied_carbon_costs_blanket_HYPO = sum((carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * step6_retrofits).sum()) 
                                        
                                        correction_factor_HYPO = embodied_carbon_costs_blanket_HYPO/embodied_carbon_costs_luxury_HYPO
                                        ### corrected price increase_through tax luxury scenario
                                        price_increase_through_tax_HYPO = price_increase_through_tax_HYPO*correction_factor_HYPO
                                        
                                        ### -2 in index i because starting in 2021 see line 424
                                    elif per_product_yes_no == "no":
                                        price_increase_through_tax_HYPO = carbon_price_CHANGE_over_time_per_country_exponential[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1)
                                                   
                                    decreased_consumption_retrofits_HYPO = (1+price_increase_through_tax_HYPO*df_price_elasticities_retrofits)
                                    
                                    consumption_decay_tax_rate_big_array_list_retrofits[j-1][14*(i-1)-14:14*(i-1),5*k-5:5*k] =  decreased_consumption_retrofits_HYPO
                                    
                            elif carbon_price_regime == "linear":
                                                
                                    if per_product_yes_no == "yes":
                                        ### divide price by 1000 to go from price per ton to price per kg
                                        price_increase_through_tax_HYPO = carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * income_elasticities[14*j-14:j*14] 
                                        #### compute embodied carbon tax without and with price adjustment so that the *average* carbon costs are preserved
                                        embodied_carbon_costs_luxury_HYPO = sum((carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * income_elasticities[14*j-14:j*14] * step6_retrofits).sum()) 
                                        embodied_carbon_costs_blanket_HYPO = sum((carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1) * step6_retrofits).sum()) 
                                        
                                        correction_factor_HYPO = embodied_carbon_costs_blanket_HYPO/embodied_carbon_costs_luxury_HYPO
                                        ### corrected price increase_through tax luxury scenario
                                        price_increase_through_tax_HYPO = price_increase_through_tax_HYPO*correction_factor_HYPO
                                        
                                        ### -2 in index i because starting in 2021 see line 424
                                        
                                    elif per_product_yes_no == "no":
                                        price_increase_through_tax_HYPO = carbon_price_CHANGE_over_time_per_country_linear[j-1,i-2]/1000 * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,k-1],axis =1)
                                                   
                                    decreased_consumption_retrofits_HYPO = (1+price_increase_through_tax_HYPO*df_price_elasticities_retrofits)
                                    
                                    consumption_decay_tax_rate_big_array_list_retrofits[j-1][14*(i-1)-14:14*(i-1),5*k-5:5*k] =  decreased_consumption_retrofits_HYPO
                        
                 
                  
                  
                   ## transform zeros to 1 for ease of computation
                for l in range(0,88):
                      consumption_decay_tax_rate_big_array_list[j-1] = np.where(consumption_decay_tax_rate_big_array_list[j-1] != 0, consumption_decay_tax_rate_big_array_list[j-1], 1)
                      #consumption_decay_tax_rate_big_array_list[j-1] = np.where(consumption_decay_tax_rate_big_array_list[j-1] >= 0, consumption_decay_tax_rate_big_array_list[j-1], 1)
                      if retrofit_policy == "yes": consumption_decay_tax_rate_big_array_list_retrofits[j-1] = np.where(consumption_decay_tax_rate_big_array_list_retrofits[j-1] != 0, consumption_decay_tax_rate_big_array_list_retrofits[j-1], 1)
                      #print("iteration is " + str(l)) 
                      
                      
                      
                if i > 2: 
                          ### divide whole columns ("one column" here is 5 columns wide)
                          multiplier_1 = consumption_decay_tax_rate_big_array_list[j-1][:,5*(i-1)-5:5*(i-1)]/consumption_decay_tax_rate_big_array_list[j-1][:,5*(i-2)-5:5*(i-2)]
                          if retrofit_policy == "yes": multiplier_1r = consumption_decay_tax_rate_big_array_list_retrofits[j-1][:,5*(i-1)-5:5*(i-1)]/consumption_decay_tax_rate_big_array_list_retrofits[j-1][:,5*(i-2)-5:5*(i-2)]
                          ### multiplier correction because entries >0 but <1 need to be set 1 because they are not supposed to be multipliers at that time step.
                     
                          
                          multiplier_2 =  np.where(multiplier_1>1,  multiplier_1, 1)
                          multiplier_3 =  np.where(multiplier_2<1.1,  multiplier_2, 1.001)
                          stored_multiplier_bounce_back_effect[j-1][:,5*(i-1)-5:5*(i-1)] = multiplier_3
                          if retrofit_policy == "yes": 
                              multiplier_2r =  np.where(multiplier_1r>1,  multiplier_1r, 1)
                              multiplier_3r =  np.where(multiplier_2r<1.1,  multiplier_2r, 1.001)
                          
                      
                          #### finish of the multiplier per time step calculation by multiplying across decay effects from all previously introduced carbon price increases
                      
                          tax_decay_multiplier_finished = np.zeros((14,5))
                          if retrofit_policy == "yes": tax_decay_multiplier_finished_retrofits = np.zeros((14,5))
                        
                          for e in range(0,14):
                             tax_decay_multiplier_finished[e,:] = np.prod(multiplier_3[e::14], axis = 0)
                             if retrofit_policy == "yes": tax_decay_multiplier_finished_retrofits[e,:] = np.prod(multiplier_3r[e::14], axis = 0)

                #np.histogram(stored_multiplier_bounce_back_effect[4], bins = 20)
            
            
            
            
            
            #####################################################
            ##### BLOCK #5 TAX IMPACT AND TOTAL CONSUMPTION #####
            #####################################################
            
            if i == 2: 
                    households_not_retrofitted_yet[j*5-5:j*5,i-1] = households_over_time_quintile[j-1,i-1]
                    households_not_retrofitted_yet[j*5-5:j*5,i-2] = households_over_time_quintile[j-1,i-1]
                    if retrofit_policy == "yes":
                        households_retrofitted_already[j*5-5:j*5,i-1] = 0
                        households_retrofitted_already[j*5-5:j*5,i-2] = 0
            
            
            ### memory array that stores how many dwellings still have to be retrofitted in each quintile, thus rows = 88*5 , countries * quintiles            
            #if i == 2: households_not_retrofitted_yet[j*5-5:j*5,i-1] = households_over_time_quintile[j-1,i-1]
            ### also set up tracker array of households that are already retrofitted because we need to distinguish between retrofitted and non-retrofitted over time
            #if i == 2: households_retrofitted_already[j*5-5:j*5,i-1] = 0
            ### take into account population growth and thus growth of the number of households/dwellings 
            ###new households (can be negative) but if negative and household section equal zero already set to zero
            if i >= 2:
                    new_hhs = (population_over_time4[j-1,i-1]-population_over_time4[j-1,i-2])*households_per_capita[j-1]
                    new_hhs_per_quintile = new_hhs/5
                    households_not_retrofitted_yet[j*5-5:j*5,i] = households_not_retrofitted_yet[j*5-5:j*5,i-1] + new_hhs_per_quintile 
                    ### set negative households amount to zero, in case population growth is negative and therefore growth of number of households is negative 
                    ### this can happen if all hh already are retrofitted in a quintile
                    if retrofit_policy == "yes":                 
                        households_not_retrofitted_yet[j*5-5:j*5,i-1][households_not_retrofitted_yet[j*5-5:j*5,i-1] < 0] = 0              
                        households_retrofitted_already[j*5-5:j*5,i-1] = households_over_time_quintile[j-1,i-1] - households_not_retrofitted_yet[j*5-5:j*5,i-1]
                                    
                
            #### NOT YET RETROFITTED HOUSEHOLDS ####
             
            if tech_bounce_back ==  "yes" and i > 2: pc_consumption_after_tax = copy.deepcopy(step6) * decreased_consumption * tax_decay_multiplier_finished
            else:
                pc_consumption_after_tax = copy.deepcopy(step6) * decreased_consumption
               
            pc_consumption_after_tax[pc_consumption_after_tax<0]=1            
            total_consumption_after_tax = pc_consumption_after_tax / households_per_capita[j-1] * households_not_retrofitted_yet[j*5-5:j*5,i-1]
            df_yearly.iloc[14*j-14:j*14,:] = pc_consumption_after_tax 
            
           
            
            ### ALREADY RETROFITTED HOUSEHOLDS ####
            if retrofit_policy == "yes":
                                     
                if tech_bounce_back == "yes" and i > 2: pc_consumption_after_tax_retrofits = copy.deepcopy(step6_retrofits) * decreased_consumption_retrofits * tax_decay_multiplier_finished_retrofits
                else:
                    pc_consumption_after_tax_retrofits = copy.deepcopy(step6_retrofits) * decreased_consumption_retrofits
                    
                    
                pc_consumption_after_tax_retrofits[pc_consumption_after_tax_retrofits<0]=1            
                total_consumption_after_tax_retrofits = pc_consumption_after_tax_retrofits / households_per_capita[j-1] * households_retrofitted_already[j*5-5:j*5,i-1]
                df_yearly_retrofits.iloc[14*j-14:j*14,:] = pc_consumption_after_tax_retrofits 
                
                #df_yearly_total.iloc[14*j-14:j*14,:] = total_consumption_after_tax
                        

            
            ####################################
            ##### BLOCK REVENUE CALCULATION ####
            ####################################
            
            #### NOT YET RETROFITTED HOUSEHOLDS ####
        
                       
             
            ##revenue needs to be calculated on the basis of the *total* carbon price not just the marginal increases as above for the *physical* consumption volume 
            ### these are actually cumulative numbers so to speak based on the total/cumulative carbon price 
            embodied_carbon_costs_total = total_consumption_after_tax *np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1)*sum(carbon_price_CHANGE_over_time_per_country_exponential[j-1,:i-2+1])/1000
            total_tax_rate = foo(embodied_carbon_costs_total,total_consumption_after_tax) 
            tax_revenue_granular = total_tax_rate * total_consumption_after_tax            
            tax_revenue_total = sum(tax_revenue_granular.sum())         
            physical_cons_time_arr[j-1,i+1] = sum(total_consumption_after_tax.sum())
            tax_revenue_over_time_arr[j-1,i+1] =  tax_revenue_total
            
            df_yearly_tax_revenue.iloc[14*j-14:j*14,:] = total_tax_rate * pc_consumption_after_tax
            
            
            ### ALREADY RETROFITTED HOUSEHOLDS ####
            if retrofit_policy == "yes":
                                     
            ### these are actually cumulative numbers so to speak based on the total/cumulative carbon price 
                embodied_carbon_costs_total_retrofits = total_consumption_after_tax_retrofits * np.expand_dims(carbon_intensities_over_time[14*j-14:j*14,i],axis =1)*sum(carbon_price_CHANGE_over_time_per_country_exponential[j-1,:i-2+1])/1000
                total_tax_rate_retrofits = foo(embodied_carbon_costs_total_retrofits, total_consumption_after_tax_retrofits) 
                tax_revenue_granular_retrofits = total_tax_rate_retrofits * total_consumption_after_tax_retrofits            
                tax_revenue_total_retrofits = sum(tax_revenue_granular_retrofits.sum())         
                physical_cons_time_arr_retrofits[j-1,i+1] = sum(total_consumption_after_tax_retrofits.sum())
                tax_revenue_over_time_arr_retrofits[j-1,i+1] =  tax_revenue_total_retrofits
                
                df_yearly_tax_revenue_retrofits.iloc[14*j-14:j*14,:] =  total_tax_rate_retrofits * pc_consumption_after_tax_retrofits
            
                ## sum of tax revenue across no yet retrofitted households and retrofitted households
                tax_revenue_total_sum = tax_revenue_total + tax_revenue_total_retrofits
            else:    
                tax_revenue_total_sum = tax_revenue_total    
                 
            ####################################
            #### BLOCK REVENUE RECYCLING ####### 
            ####################################
            
            ### add budget from previous year in case budget is available. should be equal to zero if gone through retrofitting. 
            budget_available = copy.deepcopy(tax_revenue_total_sum) + copy.deepcopy(budget_available) 
            
            budget_available_over_time[j-1,i-2] = copy.deepcopy(budget_available) 
            
            ####################################
            #### REDISTRIBUTION POLICY ######### 
            ####################################
            
            
            #### we calculate what happens with the budget if the intention is to redistribute revenues back to low-income households
            if redistribution_policy == "yes": 
                
                if retrofit_policy == "no":
                        
                    pc_consumption_before_tax = copy.deepcopy(step6)
                    
                    ####################################
                    #### Tech bounce back effect ####### 
                    ####################################
                    #### introduce tax decay multiplier effect before redistribution otherwise it is false
                    #if tech_bounce_back == "yes" and i > 2:  pc_consumption_after_tax = copy.deepcopy(pc_consumption_after_tax)*tax_decay_multiplier_finished
                    
                    pc_reduction_consumption = pc_consumption_before_tax - pc_consumption_after_tax
                    quintiles_paid_back = int(zero_trade_off_points_modelled_quintiles[j-1])                
                    quintiles_paid_back_consumption = pc_reduction_consumption.iloc[0:14,0:quintiles_paid_back]                
                    total_consumption_paid_back = quintiles_paid_back_consumption*households_not_retrofitted_yet[j*5-5:j*5, i][0:quintiles_paid_back]                
                    ## this is a highly simplified procedure which only pays back low incomes households if 
                    ## the budget is large enough to pay back all quintiles below zero trade off point
                    if budget_available - sum(total_consumption_paid_back.sum()) > 0:
                          budget_available = budget_available - sum(total_consumption_paid_back.sum())
                          
                          for column in range(1, (5 - quintiles_paid_back)+1):
                               quintiles_paid_back_consumption[str(quintiles_paid_back + column)] = float(0) 
                               
                          quintiles_paid_back_consumption.columns = quintiles_paid_back_consumption.columns.astype(int)    
                          
                          pc_consumption_after_tax = pc_consumption_after_tax + quintiles_paid_back_consumption
                          df_yearly.iloc[14*j-14:j*14,:] = pc_consumption_after_tax 
                    else:
                         pass 
                     
                if retrofit_policy == "yes":
                    ### here everything twice again because of the distinction between retrofitted and non retrofitted households
                    pc_consumption_before_tax = copy.deepcopy(step6)
                    pc_consumption_before_tax_retrofits = copy.deepcopy(step6_retrofits)
                    
                                   
                    pc_reduction_consumption = pc_consumption_before_tax - pc_consumption_after_tax
                    pc_reduction_consumption_retrofits = pc_consumption_before_tax_retrofits - pc_consumption_after_tax_retrofits
                    quintiles_paid_back = int(zero_trade_off_points_modelled_quintiles[j-1])   
                                 
                    quintiles_paid_back_consumption = pc_reduction_consumption.iloc[0:14,0:quintiles_paid_back]    
                    quintiles_paid_back_consumption_retrofits = pc_reduction_consumption_retrofits.iloc[0:14,0:quintiles_paid_back]
                    
                    
                    
                    total_consumption_paid_back_non_retrofits = quintiles_paid_back_consumption*households_not_retrofitted_yet[j*5-5:j*5, i][0:quintiles_paid_back]     
                    total_consumption_paid_back_retrofits = quintiles_paid_back_consumption_retrofits*households_retrofitted_already[j*5-5:j*5, i][0:quintiles_paid_back] 
                    
                    total_consumption_paid_back = total_consumption_paid_back_non_retrofits + total_consumption_paid_back_retrofits
                    
                    
                    
                    ## this is a highly simplified procedure which only pays back low incomes households if 
                    ## the budget is large enough to pay back all quintiles below zero trade off point
                    if budget_available - sum(total_consumption_paid_back.sum()) > 0:
                          budget_available = budget_available - sum(total_consumption_paid_back.sum())
                          
                          for column in range(1, (5 - quintiles_paid_back)+1):
                               quintiles_paid_back_consumption[str(quintiles_paid_back + column)] = float(0) 
                               quintiles_paid_back_consumption_retrofits[str(quintiles_paid_back + column)] = float(0)
                               
                          quintiles_paid_back_consumption.columns = quintiles_paid_back_consumption.columns.astype(int)    
                          quintiles_paid_back_consumption_retrofits.columns = quintiles_paid_back_consumption_retrofits.columns.astype(int) 
                          pc_consumption_after_tax = pc_consumption_after_tax + quintiles_paid_back_consumption
                          
                          pc_consumption_after_tax_retrofits = pc_consumption_after_tax_retrofits + quintiles_paid_back_consumption_retrofits
                          df_yearly.iloc[14*j-14:j*14,:] = pc_consumption_after_tax 
                          df_yearly_retrofits.iloc[14*j-14:j*14,:] = pc_consumption_after_tax_retrofits 
                    else:
                         pass 

            
            



            ####################################
            #### RETROFIT POLICY ############### 
            ####################################
            
            if retrofit_policy == "yes":
                                     
                    ### set up the budget available
                    
                    

                    ### set up retrofit costs calculation for not yet retrofitted households
                    ### first find out how much energy is used in residential energy 
                    ### subsetting heat and elect. from df. multiplied by the corresponding energy intensities
                    var1 = total_consumption_after_tax.iloc[4]
                    var2 = final_energy_intensities_over_time[14*j-14:14*j,i-1][4] #### this array starts in 2020 so i-1 = 2021, if i = 2 in loop over yrs equals 2021            
                    residential_energy_per_quintile = var1*var2      
                                    
                    
                    ##### now calculate how much energy is used per each single dwelling (approximated by households)
                    per_dwelling_energy = residential_energy_per_quintile.multiply(1/households_not_retrofitted_yet[j*5-5:j*5,i-1],axis = 0)    
                                        
                                         
                    #### now calculate the costs it would take to retrofit one dwelling i.e. reduce its energy use (thus it carbon emissions as well) by 50%.
                    retrofit_costs_per_quintile = per_dwelling_energy * costs_per_megajoule_retrofit_differentiated[j-1,0] * retrofit_impact

                    
                    ### how many houses can be retrofitted given a certain budget? 
                    
                    #### TEST TEST TEST ####
                    #np.min(np.where((v1 > 0) == True))
                    
                    ###determine number of retrofits made based on budget available
                    v1 = copy.deepcopy(households_not_retrofitted_yet[j*5-5:j*5,i-1])
                    v3 = copy.deepcopy(budget_available)
                    total_number_that_can_be_retrofitted = np.zeros((5)) 
                    if sum(v1) > 0: 
                        for p in range(np.min(np.where((v1 > 0) == True))+1,len(v1)+1):
                               v2 = v3/retrofit_costs_per_quintile[p] ### number of dwellings that can be retrofitted with budget in given quintile
                               if v2 > v1[p-1]:
                                  v2 = copy.deepcopy(v1[p-1])
                               v3 = v3 - v2*retrofit_costs_per_quintile[p] ### budget left
                               total_number_that_can_be_retrofitted[p-1] = total_number_that_can_be_retrofitted[p-1] + v2
    
                    budget_available = 0 #### budget after retrofitting is always used up
                    
                    #### still need to put in changes to total number that are not retrofitted and connect loop back to beginning 
                    
                    households_not_retrofitted_yet[j*5-5:j*5,i] = households_not_retrofitted_yet[j*5-5:j*5,i-1] - total_number_that_can_be_retrofitted                  
                    households_retrofitted_already[j*5-5:j*5,i] = households_over_time_quintile[j-1,i-1] - households_not_retrofitted_yet[j*5-5:j*5,i]
        
            else:
                pass 
            
    ## add newly normalized but product differeniated projected consumption to list. this element is used in the next iteration as a starting point 
    print("iteration is " + str(i))                  
    scen2_consumption_over_time.append(df_yearly)
    scen2_revenue_over_time.append(df_yearly_tax_revenue)
    if retrofit_policy == "yes": 
        scen2_consumption_over_time_retrofits.append(df_yearly_retrofits)
        scen2_revenue_over_time_retrofits.append(df_yearly_tax_revenue_retrofits)
    


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
##################################   MAIN SIMULATION  END    ###################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################





#### Now that per capita consumption over time has been calculated, calculate carbon emissions over time.

### first compute total consumption over time for retrofitted and non retrofitted households

if retrofit_policy == "yes": 
    
        scen2_total_cons_non_retrofits = []
        scen2_total_cons_retrofits = []
        scen2_total_cons_both_hh_types = []
        
        
        for i in range(0,82):
              if i == 0: 
                  scen2_total_cons_both_hh_types.append(consumption_over_time_differentiated_normed[0].multiply(pop_quintile_2019, axis = 0))
              elif i <=2 and i > 0:
                  scen2_total_cons_both_hh_types.append(consumption_over_time_differentiated_normed[i].multiply(population_over_time[:,i-1], axis = 0))
              else:
                  df_total_yearly_non_retrofits = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
                  df_total_yearly_retrofits = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
                  for j in range(1,89):
                          df_total_yearly_non_retrofits.iloc[j*14-14:j*14,:] = scen2_consumption_over_time[i].iloc[j*14-14:j*14,:]/households_per_capita[j-1]*households_not_retrofitted_yet[j*5-5:j*5,i]
                          df_total_yearly_retrofits.iloc[j*14-14:j*14,:] = scen2_consumption_over_time_retrofits[i].iloc[j*14-14:j*14,:]/households_per_capita[j-1]*households_retrofitted_already[j*5-5:j*5,i]
                  scen2_total_cons_non_retrofits.append(df_total_yearly_non_retrofits)
                  scen2_total_cons_retrofits.append(df_total_yearly_retrofits)
                  scen2_total_cons_both_hh_types.append(df_total_yearly_non_retrofits+df_total_yearly_retrofits)
              print("iteration is " + str(i))
              
              
        
        scen2_pc_emissions_over_time = []
        ## total emissions over time is product granular
        scen2_total_emissions_over_time = []
        ## total global is an aggregate number
        scen2_total_global_emissions_over_time = []
        for i in range(0,82): 
            
            if i == 0:
          
                #scen2_pc_emissions_over_time.append(scen2_consumption_over_time[i].multiply(carbon_intensities_2019_estimate, axis = 0))
                scen2_total_emissions_over_time.append(scen2_total_cons_both_hh_types[i].multiply(carbon_intensities_2019_estimate, axis = 0))
                scen2_total_global_emissions_over_time.append(sum(scen2_total_cons_both_hh_types[i].multiply(carbon_intensities_2019_estimate, axis = 0).sum()))   
                
            else:
                
                #scen2_pc_emissions_over_time.append(scen2_consumption_over_time[i].multiply(carbon_intensities_over_time[:,i-1], axis = 0))
                scen2_total_emissions_over_time.append(scen2_total_cons_both_hh_types[i].multiply(carbon_intensities_over_time[:,i-1], axis = 0))
                scen2_total_global_emissions_over_time.append(sum(scen2_total_cons_both_hh_types[i].multiply(carbon_intensities_over_time[:,i-1], axis = 0).sum()))
            


            for c in range(1,89): 
                 average_pc_cons_over_time[c-1,i] = sum(scen2_total_cons_both_hh_types[i][14*c-14:14*c].sum())/population_over_time4[c-1, i]
                 national_gini_coefficient_over_time[c-1,i] = gini_array_version(population_over_time3[c*5-5:c*5, i],np.array(scen2_total_emissions_over_time[i][14*c-14:14*c].sum()))

 


          
if retrofit_policy == "no": 
    
    
        scen2_total_cons_non_retrofits = []
        scen2_total_cons_both_hh_types = []
        
        
        for i in range(0,82):
              if i == 0: 
                  scen2_total_cons_both_hh_types.append(consumption_over_time_differentiated_normed[0].multiply(pop_quintile_2019, axis = 0))
              elif i <=2 and i > 0:
                  scen2_total_cons_both_hh_types.append(consumption_over_time_differentiated_normed[i].multiply(population_over_time[:,i-1], axis = 0))
              else:
                  df_total_yearly_non_retrofits = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
                  
                  for j in range(1,89):
                          df_total_yearly_non_retrofits.iloc[j*14-14:j*14,:] = scen2_consumption_over_time[i].iloc[j*14-14:j*14,:]/households_per_capita[j-1]*households_not_retrofitted_yet[j*5-5:j*5,i]
                          
                  scen2_total_cons_non_retrofits.append(df_total_yearly_non_retrofits)
                  
                  scen2_total_cons_both_hh_types.append(df_total_yearly_non_retrofits)
              print("iteration is " + str(i))


        
        scen2_pc_emissions_over_time = []
        ## total emissions over time is product granular
        scen2_total_emissions_over_time = []
        ## total global is an aggregate number
        scen2_total_global_emissions_over_time = []
        for i in range(0,82): 
            
            if i == 0:
          
                #scen2_pc_emissions_over_time.append(scen2_consumption_over_time[i].multiply(carbon_intensities_2019_estimate, axis = 0))
                scen2_total_emissions_over_time.append(scen2_total_cons_both_hh_types[i].multiply(carbon_intensities_2019_estimate, axis = 0))
                scen2_total_global_emissions_over_time.append(sum(scen2_total_cons_both_hh_types[i].multiply(carbon_intensities_2019_estimate, axis = 0).sum()))   
                
            else:
                
                #scen2_pc_emissions_over_time.append(scen2_consumption_over_time[i].multiply(carbon_intensities_over_time[:,i-1], axis = 0))
                scen2_total_emissions_over_time.append(scen2_total_cons_both_hh_types[i].multiply(carbon_intensities_over_time[:,i-1], axis = 0))
                scen2_total_global_emissions_over_time.append(sum(scen2_total_cons_both_hh_types[i].multiply(carbon_intensities_over_time[:,i-1], axis = 0).sum()))
            
            
            for c in range(1,89): 
                 average_pc_cons_over_time[c-1,i] = sum(scen2_total_cons_both_hh_types[i][14*c-14:14*c].sum())/population_over_time4[c-1, i]
                 national_gini_coefficient_over_time[c-1,i] = gini_array_version(population_over_time3[c*5-5:c*5, i],np.array(scen2_total_emissions_over_time[i][14*c-14:14*c].sum()))

 
 
 
 
 
 
 
#### summarize emissions data to quintiles 
### this is not product granular but only countries, quintiles, yrs


labels2 = labels[0::14][:,0:3]    
population_over_time2 = population_over_time[0::14] ### population per country (country = rows) per quintile per year (yr = columns)
scen2_pc_emissions_over_time_quintiles = OrderedDict()
scen2_total_emissions_over_time_quintiles = OrderedDict()

#### https://stackoverflow.com/questions/6181935/how-do-you-create-different-variable-names-while-in-a-loop
for i in range(1,83):
        scen2_pc_emissions_over_time_quintiles["{0}".format(2019+i)] = pd.DataFrame(columns = [1,2,3,4,5], index = labels2);
        scen2_total_emissions_over_time_quintiles["{0}".format(2019+i)] = pd.DataFrame(columns = [1,2,3,4,5], index = labels2);
        

        for j in range(1,89):
            scen2_pc_emissions_over_time_quintiles[str(2019+i)].iloc[j-1,:] = scen2_consumption_over_time[i].multiply(carbon_intensities_over_time[:,i-1], axis = 0).iloc[j*14-14:j*14].sum();
        
        scen2_total_emissions_over_time_quintiles[str(2019+i)] = scen2_pc_emissions_over_time_quintiles[str(2019+i)].multiply(population_over_time2[:,i-1], axis = 0)
                                
        
        print("iteration is " + str(i))

##########################################################
##########################################################
##########################################################
##########################################################
##### PLOT ALL CATEGORIES EMISSION EVOLUTION scen2 #######
####### in combination with BAU for overview  ############
##########################################################
##########################################################
##########################################################
##########################################################

category_labels = copy.deepcopy(labels[0:14,3])
for i in range(0,14): category_labels[i] = category_labels[i][:-4]


rows = 4; cols = 4;
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 12), squeeze=0, sharex=True, sharey=True)
axes = np.array(axes)

category_emissions_BAU = np.zeros((82,1));
category_emissions_scen2 = np.zeros((82,1));

#residential_emissions_BAU = np.zeros((82,1));
#residential_emissions_scen2 = np.zeros((82,1));


for c, ax in enumerate(axes.reshape(-1)):
  if c < 14:
          ax.set_ylabel("GT/yr")
          ax.set_title(category_labels[c])
          for i in range(1,82):
              category_emissions_scen2[i-1] = sum(scen2_total_emissions_over_time[i][c::14].sum())/10**12 ### category emissions starts at 2020 then because i = 1 is 2020 in scen2 emissions over time and i = 0 is 2019
              category_emissions_BAU[i-1] = sum(ssp2_emissions_total_granular_list[i][c::14].sum())/10**12
          ax.plot(years, category_emissions_scen2[:-1], label = "with policy") 
          ax.plot(years, category_emissions_BAU[:-1], label = "without policy")
          if c == 0: ax.legend(frameon = False)### from 2020 to 2100
          if c == 4: 
              residential_emissions_BAU = copy.deepcopy(category_emissions_BAU[:-1])
              residential_emissions_scen2 = copy.deepcopy(category_emissions_scen2[:-1])
              
  else:
          pass 

plt.savefig('emissions_over_time_categories.png',bbox_inches = "tight", dpi = 300);    
plt.show()


##########################################################
##########################################################
##### write out all the necessary inequality data ########
##########################################################
##########################################################


##########################################################
##########################################################
##### inequality data #1   GINI COEFF. ########
##########################################################
##########################################################
### tuple list starts at t = 2020, not at t_0 = 2019 see code below 2019 + i, i element (1,82)
tuple_list2 = []
for i in range(1,82):        
        step1 = scen2_pc_emissions_over_time_quintiles[str(2019+i)].stack()
        step2 = scen2_total_emissions_over_time_quintiles[str(2019+i)].stack()
        step3 = population_over_time3[:,i-1]            
        step4 = pd.concat([step1, step2], axis = 1, ignore_index = True)
        step4.insert(2, "2", step3, True)       
        #step5 = step4.sort_values(0)
        resulttuple = gini_dynamic(step4.iloc[:,2], step4.iloc[:,1])
        tuple_list2.append(resulttuple)
        print("iteration is "+ str(i))

##### calculate Gini coefficient over the years

### concatenate all Gini measurements to one array
scen2_Gini_array = np.zeros((81,1))
for i in range(1,82):
   scen2_Gini_array[i-1] = tuple_list2[i-1][0]    
   
   
   
 
##########################################################
##########################################################
##### Global top 1% emission sharedata #1    ##
##########################################################
##########################################################  

### ALGORITHM TO SMOOTHLY INTERPOLATE CHUNKY LORENZ CURVE, WITH LINEAR INTERPOLATION SO THAT EXACT cumulative cut off for
### top 1% can be found.


sign_arr_BIG = np.zeros((441,81))

for j in range(0,len(tuple_list2)):
        dist_to_99 = tuple_list2[j][1]-0.99
        sign_arr = np.zeros((1,len(dist_to_99)))
        
        for i in range(1,len(dist_to_99)-1):
             if (i == 0) or (i == len(dist_to_99)-1):
                  pass 
             else:
                     if np.sign(dist_to_99[i-1]) == np.sign(dist_to_99[i+1]):
                         sign_arr[:,i] = 0
                     else:
                         sign_arr[:,i] = 1
        sign_arr_BIG[:,j] = np.squeeze(np.transpose(sign_arr))
                 
#### extract data on to be (linearly) interpolated lorenz curve sections
data_arr = np.zeros((162,2)) ### always 2x2 elements belong together
index = np.expand_dims(np.linspace(0,440, 441), axis = 1)
index_arr = np.multiply(sign_arr_BIG, index)
helparr1 = np.floor((np.sum(index_arr, axis = 0)/2))
helparr2 = np.ceil((np.sum(index_arr, axis = 0)/2))
index_arr_new  = index_arr[index_arr  > 0]

for j in range(1,len(tuple_list2)+1):
    data_arr[j*2-2,0] = tuple_list2[j-1][1][int(helparr1[j-1])]
    data_arr[j*2-1,0] = tuple_list2[j-1][1][int(helparr2[j-1])]
    data_arr[j*2-2,1] = tuple_list2[j-1][2][int(helparr1[j-1])]
    data_arr[j*2-1,1] = tuple_list2[j-1][2][int(helparr2[j-1])]
    
share_99_arr_cum = np.zeros((81,1))   
for j in range(1,82):
    result_fit = lin_fit_non_log(data_arr[j*2-2:j*2,0], data_arr[j*2-2:j*2,1])
    share_99_arr_cum[j-1,0] = result_fit[0][0]+ result_fit[0][1]*0.99

scen2_share_top1_arr = 1 - share_99_arr_cum
   

##########################################################
##########################################################
##### Luxury share data #1   LUXURY SHARE. ##
##########################################################
##########################################################  

luxury = np.where(income_elasticities  <= 1, 0, income_elasticities )
luxury = np.where(luxury > 0, 1, luxury)

basic = np.where(income_elasticities  <= 1, 1, income_elasticities )
basic = np.where(basic > 1 , 0, basic)


scen2_luxury_emissions_over_time = np.zeros((82,1))
scen2_basic_emissions_over_time = np.zeros((82,1))
for i in range(0,82):
     scen2_luxury_emissions_over_time[i,:] = sum((scen2_total_emissions_over_time[i]*luxury).sum())/scen2_total_global_emissions_over_time[i]
     scen2_basic_emissions_over_time[i,:] = sum((scen2_total_emissions_over_time[i]*basic).sum())/scen2_total_global_emissions_over_time[i]
     
   
#### collect data and write into csv. 

df_data_scen2 = pd.DataFrame(columns = ["Emissions","Gini","Top1","Luxury", "Residential"], index = years)

df_data_scen2.iloc[:,0] = np.array(scen2_total_global_emissions_over_time)[1:82]/10**12
df_data_scen2.iloc[:,1] = scen2_Gini_array
df_data_scen2.iloc[:,2] = scen2_share_top1_arr
df_data_scen2.iloc[:,3] = scen2_luxury_emissions_over_time[1:82]
df_data_scen2.iloc[:,4] = residential_emissions_scen2

df_data_scen2.to_csv("df_data_scen2.csv", index = True)


##### plot national consumption and gini trajectories





for i in range(0,88):
    plt.plot(years, average_pc_cons_over_time[i,:-2])
plt.show()


for i in range(0,88):
    plt.plot(years, national_gini_coefficient_over_time[i,:-2])
plt.show()






########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
##################################   Blanket price uniform - scenario END  ###################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################



#%%  


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
##################################   Carbon budget calculation associated with emissions in model #######################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

#### https://www.ipcc.ch/site/assets/uploads/sites/2/2019/02/SR15_Chapter2_Low_Res.pdf table 2.2 
###https://www.climatewatchdata.org/ghg-emissions?chartType=area&end_year=2018&start_year=1990

## 66% for 1.5 420 GT in 01.01 2018 ==> emissions - 2018,19,20 ~3*49 ~ 147 based on 2018 emissions co2 equivalent on climate watch
## == 420-147 = 273
## 66% for 2 1170 GT in 01.01.2018 = >
## == 1170-147 = 1023

### now allocation must be done to household emissions for the 88 countries that i include.
### we take population share of countries of total and global household emissions share in 2019
total_population_world_2019 = 7.67*10**9
total_population_share_2019 = sum((population_WB_2019[:,1]).astype(float))/total_population_world_2019

household_emissions_fraction_of_emissions = 0.64 ### calculated from GTAP 9 version (Joel's data)

##===> carbon budgets allocated to model 
#66% 1.6 
budget_allocated_for_1_point_5 = 273*0.83*0.64
#66% 2 
budget_allocated_for_2_point_0 = 1023*0.83*0.64


#### abatement pathway calculated in excel model
#### load excel data
carbon_budget_consistent_pathways_linear = np.genfromtxt('carbon_budget_consistent_pathways_linear.csv', dtype = float, delimiter=',')



#%%  

#################### PLOT GLOBAL HH CONSUMPION EXPENDITURE #############

test_cons_array = np.zeros((82,1))

for i in range(0,82):
   test_cons_array[i,0]  = sum(scen2_total_cons_both_hh_types[i].sum())


total_hh_cons_physical = np.sum(physical_cons_time_arr+physical_cons_time_arr_retrofits, axis = 0)[3:82]

### for now plot 2022 to 2100 later make sure it include 2020 and 2021
#plt.plot(years2[3:82], np.sum(physical_cons_time_arr+physical_cons_time_arr_retrofits, axis = 0)[3:82], label = "total global hh cons." )
plt.plot(years2, test_cons_array, label = "total global hh cons." )
plt.plot(years2[3:82], np.sum(tax_revenue_over_time_arr+tax_revenue_over_time_arr_retrofits, axis = 0)[3:82], label = " total global tax revenue" )
plt.ylabel("$")
plt.yscale('log')
plt.legend(frameon = False)
plt.show()

plt.plot(years2[3:82], np.sum(tax_revenue_over_time_arr+tax_revenue_over_time_arr_retrofits, axis = 0)[3:82]/(np.sum(tax_revenue_over_time_arr+tax_revenue_over_time_arr_retrofits, axis = 0)[3:82]+np.sum(physical_cons_time_arr+physical_cons_time_arr_retrofits, axis = 0)[3:82])*100, label = "tax revenue/total global cons.")
plt.ylabel("%")
plt.legend(frameon = False)
plt.show()


#%%
#####################   output currently running scen2

widths_grid = [2, 2]
heights_grid = [2, 2]
fig = plt.figure(constrained_layout=True, figsize=(8,5))


gs = GridSpec(2, 2, figure=fig ,width_ratios = widths_grid, height_ratios = heights_grid)
ax1 = fig.add_subplot(gs[0, :-1])
ax2 = fig.add_subplot(gs[0, -1])
ax3 = fig.add_subplot(gs[1, :-1])
ax4 = fig.add_subplot(gs[1, -1])
#ax5 = fig.add_subplot(gs[0, :-2])

###### ax1 = total emissions over time #######
ax1.plot(years2[1:82], np.transpose(ssp2_emissions_total)[1:82]/10**12, label = "BAU") 
ax1.plot(years2[1:82], (np.array(scen2_total_global_emissions_over_time)/10**12)[1:82], label = "test")
ax1.plot(years2[1:82], carbon_budget_consistent_pathways_linear[1,:], label = "1.5 °C consistent")
ax1.plot(years2[1:82], carbon_budget_consistent_pathways_linear[2,:], label = "2 °C consistent")
ax1.set_ylabel("GT CO2e/yr")
ax1.set_ylim((0,20))
ax1.legend(frameon = False, bbox_to_anchor=(-1.04,0.5), loc="center left", borderaxespad=0)
ax1.margins(x = 0, y= 0)


###### ax2 = total Gini of emissions over time #######  
ax2.plot(years, ssp2_Gini_array[:-1], label = "BAU")
ax2.plot(years, scen2_Gini_array, label = "test")
ax2.set_ylim((0,1))
ax2.set_ylabel("Gini coefficient CO2e")
ax2.margins(x = 0, y= 0)


###### ax3 = share of top 1% emitters in total emissions ####### 
ax3.plot(years, share_top1_arr[1:-1]*100)
ax3.plot(years, scen2_share_top1_arr*100)
ax3.set_ylabel("% share top 1%")
ax3.set_ylim((0,15))
ax3.margins(x = 0, y= 0)
ax3.annotate("luxury: " + str(per_product_yes_no), xy = (2030,5), xytext = (2030,5))
ax3.annotate(str(carbon_price_regime), xy = (2030,3), xytext = (2030,3))
ax3.annotate("retrofit: " + str(retrofit_policy), xy = (2030,1), xytext = (2030,1))

###### ax4 = share of luxury emissions total emissions ####### 
ax4.plot(years, luxury_emissions_over_time[1:82]*100)
ax4.plot(years, scen2_luxury_emissions_over_time[1:82]*100)
ax4.set_ylabel("% share luxury emissions")
ax4.margins(x = 0, y= 0)
ax4.set_ylim((30,70))
plt.savefig('fig4.png',bbox_inches = "tight", dpi = 300);

#%% #### compute potential rebounds by calculating differences in consumption expense in BAU and specific scenarios

#### we do this very simply. We just calculate the difference between consumption per country per time step between BAU and scen2 and then

###if the difference is a positive number mutiply it by the average carbon intensity of cons. (average estimate) and the max carbon intensity (upper boound)
### and add these emissions to the usual scen2 pathway. this then creates a rebound pathway and it will be interesting to see how much of emission reductions remains. 

total_tax_revenue_over_time = tax_revenue_over_time_arr_retrofits + tax_revenue_over_time_arr

###calculate baseline total consumption in BAU per country... starting year 2020 so i = 1 onwards
rebound_relevant_money = np.zeros((88,82))
for i in range(0,81):
    for j in range (1,89):
        sum1 = sum((consumption_over_time_differentiated_normed[i+1].iloc[14*j-14:14*j]*population_over_time2[j-1,i]).sum())
        sum2 = sum(scen2_total_cons_both_hh_types[i+1].iloc[14*j-14:14*j].sum()) 
        sum3 = total_tax_revenue_over_time[j-1,i+1]
        rebound_relevant_money[j-1,i+1] = sum1-sum2-sum3


#### rebound relevant money needs to be greater than 0 because 
#### negative amount means household have to spend more overall ("physical consumption" + tax) than before introducing the tax.


low_values_flags =  rebound_relevant_money < 0
rebound_relevant_money[low_values_flags] = 0



### make 3 rebound scenarios one with minimum carbon intensity per country, one with average carbon intensity per country, one with maximum carbon intensity

### calculate min and max carbon intensity over time

min_intensity_time = np.zeros((88,81));
max_intensity_time = np.zeros((88,81));
for i in range(0,81):
    for j in range (1,89):
        ### starts from 2020 at i = 0 
        min_intensity_time[j-1,i] = np.min(carbon_intensities_over_time[14*j-14:14*j, i])
        max_intensity_time[j-1,i] = np.max(carbon_intensities_over_time[14*j-14:14*j, i])


### calculate average caron intensity over time 

### from 2020 onwards i = 0
average_intensity_time = np.zeros((88,81));
for i in range(0,81):
     for j in range (1,89):
           average_intensity_time[j-1,i] = sum(scen2_total_emissions_over_time[i+1].iloc[14*j-14:14*j].sum())/sum(scen2_total_cons_both_hh_types[i+1].iloc[14*j-14:14*j].sum())            
     

#### calculate rebound effect by taking relevant money times average intensity (and min, max variations for control)

### where index = 3 is 2022 so index = 0 is 2019 in rebound_relevant_money
### for average intensity it is index = 0 is 2020

rebound_relevant_money_2 = np.delete(rebound_relevant_money,0,1)


rebound_emissions = np.multiply(rebound_relevant_money_2,average_intensity_time)

rebound_emissions_global = np.sum(np.multiply(rebound_relevant_money_2,average_intensity_time), axis = 0)



#%%
## calculate low policy adoption sensitivity 
## what if only use adopted the tax pathway ? no one else.. we do this by looking at 2050 and seeing what if only usa adopted till then
## 2050 index = 31, index usa in countries = 87, index china= 11, index india = 21

scen2_total_emissions_over_time[31].sum()
ssp2_emissions_total_granular_list[31].sum()
#### total emission reductions scen2



labels = np.genfromtxt('labels.csv', dtype = str, delimiter=',');

### prepare europe selection vector


selection_europe = np.where(meta_data_countries[:,6] == "Europe")

selection_Russia_Brazil_Japan_SA = [86, 44, 47, 7]

####  must compute cumulative savings till 2050 for each scenario. 
savings_per_year = np.zeros((8,32))
for i in range(0, 32):
    ### SCEN2 all countries
    savings_per_year[0,i] = (sum(ssp2_emissions_total_granular_list[i].sum()) - sum(scen2_total_emissions_over_time[i].sum()))/10**12
    
    #### scen2 USA only
    USA_only_2050 = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
    USA_only_2050.iloc[88*14-14:88*14,:] = scen2_total_emissions_over_time[i].iloc[88*14-14:88*14,:]
    USA_only_2050.iloc[:88*14-14,:]  = ssp2_emissions_total_granular_list[i].iloc[:88*14-14,:]
    savings_per_year[1,i] = (sum(ssp2_emissions_total_granular_list[i].sum()) - sum(USA_only_2050.sum()))/10**12
            
   
    #### scen2 China only
    CHINA_only_2050 = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
    CHINA_only_2050.iloc[:88*14,:]  = ssp2_emissions_total_granular_list[i].iloc[:88*14,:]
    CHINA_only_2050.iloc[12*14-14:12*14,:] = scen2_total_emissions_over_time[i].iloc[12*14-14:12*14,:]
    savings_per_year[2,i]  = (sum(ssp2_emissions_total_granular_list[i].sum()) - sum(CHINA_only_2050.sum()))/10**12
    
    
    ##2050 India only dataframe
    India_only_2050 = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
    India_only_2050.iloc[:88*14,:] = ssp2_emissions_total_granular_list[i].iloc[:88*14,:]
    India_only_2050.iloc[22*14-14:22*14,:] = scen2_total_emissions_over_time[i].iloc[22*14-14:22*14,:]
    savings_per_year[3,i] = (sum(ssp2_emissions_total_granular_list[i].sum()) - sum(India_only_2050.sum()))/10**12
    
        
    ##2050 Europe only dataframe
    Europe_only_2050 = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
    Europe_only_2050.iloc[:88*14,:] = ssp2_emissions_total_granular_list[i].iloc[:88*14,:]
    for j in selection_europe[0]:
        Europe_only_2050.iloc[(j+1)*14-14:(j+1)*14,:] = scen2_total_emissions_over_time[i].iloc[(j+1)*14-14:(j+1)*14,:]
    savings_per_year[4,i] = (sum(ssp2_emissions_total_granular_list[i].sum()) - sum(Europe_only_2050.sum()))/10**12
        
    ##2050 Russia only
    Russia_only_2050 = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
    Russia_only_2050.iloc[:88*14,:] = ssp2_emissions_total_granular_list[i].iloc[:88*14,:]
    Russia_only_2050.iloc[45*14-14:45*14,:] = scen2_total_emissions_over_time[i].iloc[45*14-14:45*14,:]
    savings_per_year[5,i] = (sum(ssp2_emissions_total_granular_list[i].sum()) - sum(Russia_only_2050.sum()))/10**12
    
    ##2050 Japan only
    
    Japan_only_2050 = pd.DataFrame(columns = [1,2,3,4,5], index = labels)
    Japan_only_2050.iloc[:88*14,:] = ssp2_emissions_total_granular_list[i].iloc[:88*14,:]
    Japan_only_2050.iloc[87*14-14:87*14,:] = scen2_total_emissions_over_time[i].iloc[87*14-14:87*14,:]
    savings_per_year[6,i] = (sum(ssp2_emissions_total_granular_list[i].sum()) - sum(Japan_only_2050.sum()))/10**12
    

#%%

### SET UP figure 7, output dynamic model for figure in paper

###  load data


data = np.genfromtxt('emissions_trajectories_output.csv', dtype = str, delimiter=',')
data2 = np.genfromtxt('carbon_prices_dynamic_scenarios.csv', dtype = str, delimiter=',')
data3 = np.genfromtxt('residential_emissions_trajectories_output.csv', dtype = str, delimiter=',')
data4 = np.genfromtxt('luxury_emissions_trajectories_output.csv', dtype = str, delimiter=',')
data5 = np.genfromtxt('carbon_price_trajectories_data.csv', dtype = str, delimiter=',')

## set up color map

cm1 = plt.cm.get_cmap('hot')
cm2 = plt.cm.get_cmap('winter')
cm3 = plt.cm.get_cmap('RdPu')
cm4 = plt.cm.get_cmap('Greens')

cmap_bar =  [  cm1(0.8), cm1(0.6), cm1(0.4), cm2(0.2), cm2(0.4), cm2(0.6),
            cm3(0.2), cm3(0.3), cm3(0.5), cm3(0.7), cm3(0.85), cm3(0.9),
            cm4(0.6), cm4(0.8)]

### MAKE figure 6, output dynamic model for figure in paper




widths_grid = [2, 2]
heights_grid = [2, 2, 2]
fig = plt.figure(constrained_layout=True, figsize=(8,8))

gs = GridSpec(3, 2, figure=fig ,width_ratios = widths_grid, height_ratios = heights_grid)
ax1 = fig.add_subplot(gs[0, :-1])
ax2 = fig.add_subplot(gs[0, -1])
ax3 = fig.add_subplot(gs[1, :-1])
ax4 = fig.add_subplot(gs[1, -1])
ax5 = fig.add_subplot(gs[2, :])




### PANEL A

barWidth = 0.25
# set heights of bars
bars1 = data2[1:4,2].astype(float)
bars2 = data2[1:4,3].astype(float)
bars3 = data2[1:4,4].astype(float)
bars4 = data2[1:4,5].astype(float)
# Set position of bar on X axis
r1 = np.arange(len(bars1))*1.2
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
# Make the plot
ax1.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label=data2[0,2])
ax1.bar(r2, bars2, color=cm1(0.6), width=barWidth, edgecolor='white', label=data2[0,3])
ax1.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label=data2[0,4])
ax1.bar(r4, bars4, color=cm2(0.2), width=barWidth, edgecolor='white', label=data2[0,5])
# details
ax1.set_xticks([0.4, 1.6, 2.8])
ax1.set_xticklabels(['Low price', 'Medium price', 'High price'])
ax1.set_ylabel('carbon price 2022 $/t')
ax1.legend(frameon = False, loc = "best")
ax1.annotate('a', xy=(ax1.get_xlim()[0],ax1.get_ylim()[1]+10),fontsize=14 ,annotation_clip=False)

### PANEL B


#ax2.plot(years2[1:82], data5[2:,5].astype(float), label ="low income/low price")
#ax2.plot(years2[1:82], data5[2:,1].astype(float), label ="low income/medium price")
#ax2.plot(years2[1:82], data5[2:,9].astype(float), label ="low income/high price")
ax2.plot(years2[1:82], data5[2:,8].astype(float), label ="high-income/low price")
ax2.plot(years2[1:82], data5[2:,4].astype(float), label ="high-income/medium price")
ax2.plot(years2[1:82], data5[2:,12].astype(float), label ="high-income/high price")
ax2.set_ylabel('carbon price $/t')
ax2.legend(frameon = False, loc = "best", fontsize=9)
ax2.annotate('b', xy=(ax2.get_xlim()[0]-1/3,ax2.get_ylim()[1]+100),fontsize=14 ,annotation_clip=False)

scenarios_plotted = np.array([1,2,3,4,5,6,10,11,12,13,14])

### PANEL C main emissions scenario across all countries
###### ax1 = total emissions over time #######
for i in scenarios_plotted:
    ax3.plot(years2[1:82], data[2:,i].astype(float), label = data[0,i], color = cmap_bar[i-1])     
ax3.set_ylabel("total GT CO2e/yr")
ax3.set_ylim((0,20))    
ax3.margins(x = 0)
ax3.annotate('c', xy=(ax3.get_xlim()[0],ax3.get_ylim()[1]+1),fontsize=14 ,annotation_clip=False)
handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0, 0.65), frameon = False)

scenarios_plotted2 = np.array([1,2,3,4,8,9,10,11,12])

### PANEL D
for i in scenarios_plotted2:
    ax4.plot(years2[1:82], data3[2:,i].astype(float), label = data[0,i], color = cmap_bar[i+1])  

ax4.margins(x = 0)
ax4.set_ylabel("residential GT CO2e/yr")
ax4.annotate('d', xy=(ax4.get_xlim()[0],ax4.get_ylim()[1]+0.25),fontsize=14 ,annotation_clip=False)


### interlude


savings_per_year[7,:] = savings_per_year[0,:] - np.sum(savings_per_year[1:7,:], axis = 0)
labels_waterfall = ["USA", "China", "India", "Europe", "Russia", "Japan", "RoW","All"]
cumulative_2050 = np.sum(savings_per_year, axis = 1)


### PANEL E

### only is correct like in paper in MP.RR scenario

x = [1,1.5,2,2.5,3,3.5,4,4.5]
y  = [cumulative_2050[1], 
      cumulative_2050[2]+cumulative_2050[1],
      cumulative_2050[3]+cumulative_2050[2]+cumulative_2050[1],
      cumulative_2050[4]+cumulative_2050[3]+cumulative_2050[2]+cumulative_2050[1],
      cumulative_2050[5]+cumulative_2050[4]+cumulative_2050[3]+cumulative_2050[2]+cumulative_2050[1],
      cumulative_2050[6]+cumulative_2050[5]+cumulative_2050[4]+cumulative_2050[3]+cumulative_2050[2]+cumulative_2050[1],
      cumulative_2050[0],
      cumulative_2050[0]
      ]

width = 0.35       # the width of the bars: can also be len(x) sequence

ax5.bar(1, cumulative_2050[1], width, label=labels_waterfall[1])

ax5.bar(1.5, cumulative_2050[2], width, bottom=cumulative_2050[1],
       label=labels_waterfall[2])

ax5.bar(2, cumulative_2050[3], width, bottom=cumulative_2050[2]+cumulative_2050[1],
       label=labels_waterfall[3])

ax5.bar(2.5, cumulative_2050[4], width, bottom=cumulative_2050[3]+cumulative_2050[2]+cumulative_2050[1],
       label=labels_waterfall[4])
ax5.bar(3, cumulative_2050[5], width, bottom=cumulative_2050[4]+cumulative_2050[3]+cumulative_2050[2]+cumulative_2050[1],
       label=labels_waterfall[5])
ax5.bar(3.5, cumulative_2050[6], width, bottom=cumulative_2050[5]+cumulative_2050[4]+cumulative_2050[3]+cumulative_2050[2]+cumulative_2050[1],
       label=labels_waterfall[6])
ax5.bar(4, cumulative_2050[7], width, bottom=cumulative_2050[6]+cumulative_2050[5]+cumulative_2050[4]+cumulative_2050[3]+cumulative_2050[2]+cumulative_2050[1],
       label=labels_waterfall[7])
ax5.bar(4.5, cumulative_2050[0], width,
       label=labels_waterfall[0])

for i in range(0,8):
        ax5.text(x[i], y[i]+2, labels_waterfall[i], ha = 'center')
        
ax5.set_ylim((0,110)) 
ax5.xaxis.set_visible(False)
ax5.set_ylabel('GT CO2e cumulative')
ax5.annotate('e', xy=(ax5.get_xlim()[0],ax5.get_ylim()[1]+6),fontsize=14 ,annotation_clip=False)

ax5.annotate('medium price scenario (MP.RR) until 2050', xy=(2.0,10),fontsize=12 ,annotation_clip=False)







plt.savefig('fig7.png',bbox_inches = "tight", dpi = 300);





#%%
### calculate the difference per country per between with tax and without tax consumption in a scenario without retrofit for simplicity

in_detail_cons_difference_between_tax_and_notax = []

country_pc_difference_tax_notax_consumption = np.zeros((88*5,83))

for i in range(0,83):
     for j in range (1,89):
       ### with tax divided by without tax  so if > 1 then with tax expenses are higher, if <1 then with tax expenses are lower
       country_pc_difference_tax_notax_consumption[j*5-5:j*5, i] = (scen2_revenue_over_time[i].iloc[j*14-14:j*14] +scen2_consumption_over_time[i].iloc[j*14-14:j*14]).sum()/(consumption_over_time_differentiated_normed[i].iloc[j*14-14:j*14].sum())
          
     in_detail_cons_difference_between_tax_and_notax.append((scen2_revenue_over_time[i]+scen2_consumption_over_time[i])/consumption_over_time_differentiated_normed[i])
         
         

    
    

#%%
##################################################################################################
# figure 7 in paper with tech bounce back equation instead of standard dynamics equation#########
##################################################################################################

data_bounce = np.genfromtxt('emissions_trajectories_output - with tax bounce back.csv', dtype = str, delimiter=',')
waterfall_bounce = np.genfromtxt('data_water_fall_diagram_tech_bounce.csv', dtype = str, delimiter=',')
#data3 = np.genfromtxt('residential_emissions_trajectories_output.csv', dtype = str, delimiter=',')
#data4 = np.genfromtxt('luxury_emissions_trajectories_output.csv', dtype = str, delimiter=',')
#data5 = np.genfromtxt('carbon_price_trajectories_data.csv', dtype = str, delimiter=',')

## set up color map

cm1 = plt.cm.get_cmap('hot')
cm2 = plt.cm.get_cmap('winter')
cm3 = plt.cm.get_cmap('RdPu')
cm4 = plt.cm.get_cmap('Greens')

cmap_bar =  [  cm1(0.8), cm1(0.6), cm1(0.4), cm2(0.2), cm2(0.4), cm2(0.6),
            cm3(0.2), cm3(0.3), cm3(0.5), cm3(0.7), cm3(0.85), cm3(0.9),
            cm4(0.6), cm4(0.8)]

### MAKE figure 6, output dynamic model for figure in paper




widths_grid = [2, 2]
heights_grid = [2, 2, 2]
fig = plt.figure(constrained_layout=True, figsize=(8,8))

gs = GridSpec(3, 2, figure=fig ,width_ratios = widths_grid, height_ratios = heights_grid)
ax1 = fig.add_subplot(gs[0, :-1])
ax2 = fig.add_subplot(gs[0, -1])
ax3 = fig.add_subplot(gs[1, :-1])
ax4 = fig.add_subplot(gs[1, -1])
ax5 = fig.add_subplot(gs[2, :])




### PANEL A

barWidth = 0.25
# set heights of bars
bars1 = data2[1:4,2].astype(float)
bars2 = data2[1:4,3].astype(float)
bars3 = data2[1:4,4].astype(float)
bars4 = data2[1:4,5].astype(float)
# Set position of bar on X axis
r1 = np.arange(len(bars1))*1.2
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
# Make the plot
ax1.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label=data2[0,2])
ax1.bar(r2, bars2, color=cm1(0.6), width=barWidth, edgecolor='white', label=data2[0,3])
ax1.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label=data2[0,4])
ax1.bar(r4, bars4, color=cm2(0.2), width=barWidth, edgecolor='white', label=data2[0,5])
# details
ax1.set_xticks([0.4, 1.6, 2.8])
ax1.set_xticklabels(['Low price', 'Medium price', 'High price'])
ax1.set_ylabel('Carbon price 2022 $/t')
ax1.legend(frameon = False, loc = "best")
ax1.annotate('a', xy=(ax1.get_xlim()[0],ax1.get_ylim()[1]+10),fontsize=14 ,annotation_clip=False)

### PANEL B


#ax2.plot(years2[1:82], data5[2:,5].astype(float), label ="low income/low price")
#ax2.plot(years2[1:82], data5[2:,1].astype(float), label ="low income/medium price")
#ax2.plot(years2[1:82], data5[2:,9].astype(float), label ="low income/high price")
ax2.plot(years2[1:82], data5[2:,8].astype(float), label ="high income/low price")
ax2.plot(years2[1:82], data5[2:,4].astype(float), label ="high income/medium price")
ax2.plot(years2[1:82], data5[2:,12].astype(float), label ="high income/high price")
ax2.set_ylabel('Carbon price $/t')
ax2.legend(frameon = False, loc = "best", fontsize=9)
ax2.annotate('b', xy=(ax2.get_xlim()[0]-1/3,ax2.get_ylim()[1]+100),fontsize=14 ,annotation_clip=False)

scenarios_plotted = np.array([1,2,3,4,5,6,10,11,12,13,14])

### PANEL C main emissions scenario across all countries
###### ax1 = total emissions over time #######
for i in scenarios_plotted:
    ax3.plot(years2[1:82], data_bounce[2:,i].astype(float), label = data_bounce[0,i], color = cmap_bar[i-1])     
ax3.set_ylabel("Total GT CO2e/yr")
ax3.set_ylim((0,20))    
ax3.margins(x = 0)
ax3.annotate('c', xy=(ax3.get_xlim()[0],ax3.get_ylim()[1]+1),fontsize=14 ,annotation_clip=False)
handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0, 0.65), frameon = False)

scenarios_plotted2 = np.array([1,2,3,4,8,9,10,11,12])

### PANEL D
for i in scenarios_plotted2:
    ax4.plot(years2[1:82], data3[2:,i].astype(float), label = data[0,i], color = cmap_bar[i+1])  

ax4.margins(x = 0)
ax4.set_ylabel("Residential GT CO2e/yr")
ax4.annotate('d', xy=(ax4.get_xlim()[0],ax4.get_ylim()[1]+0.25),fontsize=14 ,annotation_clip=False)


### interlude


savings_per_year[7,:] = savings_per_year[0,:] - np.sum(savings_per_year[1:7,:], axis = 0)
labels_waterfall = ["USA", "China", "India", "Europe", "Russia", "Japan", "RoW","All"]
cumulative_2050 = np.sum(savings_per_year, axis = 1)


### PANEL E

### only is correct like in paper in MP.RR scenario


width = 0.35       # the width of the bars: can also be len(x) sequence

ax5.bar(1, float(waterfall_bounce[0,2]), width, label=labels_waterfall[0])

ax5.bar(1.5, float(waterfall_bounce[1,2]), width, bottom = float(waterfall_bounce[0,2]),label=labels_waterfall[1])

ax5.bar(2, float(waterfall_bounce[2,2]), width, bottom = float(waterfall_bounce[1,2])+float(waterfall_bounce[0,2]),
       label=labels_waterfall[2])

ax5.bar(2.5, float(waterfall_bounce[3,2]), width, bottom =  float(waterfall_bounce[2,2])+float(waterfall_bounce[1,2])+float(waterfall_bounce[0,2]),
       label=labels_waterfall[3])

ax5.bar(3, float(waterfall_bounce[4,2]), width, bottom =  float(waterfall_bounce[3,2])+float(waterfall_bounce[2,2])+float(waterfall_bounce[1,2])+float(waterfall_bounce[0,2]),label=labels_waterfall[4])


ax5.bar(3.5, float(waterfall_bounce[5,2]), width,  bottom =  float(waterfall_bounce[4,2])+float(waterfall_bounce[3,2])+float(waterfall_bounce[2,2])+float(waterfall_bounce[1,2])+float(waterfall_bounce[0,2]),
       label=labels_waterfall[5])

ax5.bar(4, float(waterfall_bounce[6,2]), width,  bottom = float(waterfall_bounce[5,2]) + float(waterfall_bounce[4,2])+float(waterfall_bounce[3,2])+float(waterfall_bounce[2,2])+float(waterfall_bounce[1,2])+float(waterfall_bounce[0,2]),
       label=labels_waterfall[6])


ax5.bar(4.5,float(waterfall_bounce[7,2]), width,
       label=labels_waterfall[7])

for i in range(0,8):
        ax5.text(x[i], float(waterfall_bounce[i,1])+2, labels_waterfall[i], ha = 'center')
        
ax5.set_ylim((0,110)) 
ax5.xaxis.set_visible(False)
ax5.set_ylabel('GT CO2e cumulative')
ax5.annotate('e', xy=(ax5.get_xlim()[0],ax5.get_ylim()[1]+6),fontsize=14 ,annotation_clip=False)

ax5.annotate('Medium price scenario (MP.RR) until 2050', xy=(2.0,10),fontsize=12 ,annotation_clip=False)

