
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from scipy import optimize

# disaggregation function takes as arguement a dataframe containing the on/off states of all the appliances
#at all the instances alongwith the respective aggregate electricity consumption

def disagg_chunk(data, weight_for_stationary=False):
    # the last column of the data Dataframe should contain the aggregate meter data
    num_appliances = data.shape[1]-1
    df=data.groupby(data.columns.tolist()[:-1]).mean()["Mains"].reset_index()
    X=np.array(df.iloc[:,:num_appliances])
    # print np.linalg.matrix_rank(X)
    df['n']=data.groupby(data.columns.tolist()[:-1]).count().reset_index()['Mains']
    df['w']=df['n']/df.iloc[:,:num_appliances].sum(axis=1)
    W=np.diag(df['w'])
    Y=np.array(df.loc[:,"Mains"])
    # W=np.diag(np.ones_like(Y))
    num=X.T.dot(W.dot(Y))
    den=X.T.dot(W.dot(X))
    # solving for the estimate of the average power consumption
    obj=lambda p: np.linalg.norm(den.dot(p)-num) # probably this objective function is wrong
    const={'type': 'ineq', 'fun': lambda p: p }
    po=np.ones(num_appliances)
    optim=optimize.minimize(obj,po, constraints=const)
    # solving for the estimate of the variance
    # please note that here i am assuming that all the rows of the weighted mean table have satisfy the
    #contraint:n>sum(x)
    if weight_for_stationary:
        var= data.groupby(['x1', 'x2','x3']).var()["y"].reset_index()["y"]
        term1=df['n']*var
        term1=term1[var.dropna().index]
        X_notnull=X[var.dropna().index]
        term2 =lambda a: ((a.T**2)*X_notnull).sum(axis=1)
        obj_var=lambda a: np.linalg.norm((term1-term2(a))/(X_notnull.sum(axis=1)))
        const_var={'type': 'ineq', 'fun': lambda a: power**2-a**2>=0}
        ao=np.zeros_like(num_appliances)
        optim_var=optimize.minimize(obj_var,ao, constraints=const_var)
        return optim, optim_var
    else:
        return optim,None

def disagg_chunk_temp(data, weight_for_stationary=False):
    # the last column of the data Dataframe should contain the aggregate meter data
    num_appliances = data.shape[1]-1
    df=data.groupby(data.columns.tolist()[:-1]).mean()["Mains"].reset_index()
    X=np.array(df.iloc[:,:num_appliances])
    df['n']=data.groupby(data.columns.tolist()[:-1]).count().reset_index()['Mains']
    df['w']=df['n']/df.iloc[:,:num_appliances].sum(axis=1)
    W=np.diag(df['w'])
    Y=np.array(df.loc[:,"Mains"])
    # W=np.diag(np.ones_like(Y))
    num=X.T.dot(W.dot(Y))
    den=X.T.dot(W.dot(X))
    # solving for the estimate of the average power consumption
    obj=lambda p: np.linalg.norm(den.dot(p)-num) # probably this objective function is wrong
    const={'type': 'ineq', 'fun': lambda p: p }
    po=np.ones(num_appliances)
    optim=optimize.minimize(obj,po, constraints=const)
    power=optim.x
    # solving for the estimate of the variance
    # please note that here i am assuming that all the rows of the weighted mean table have satisfy the
    #contraint:n>sum(x)
    if weight_for_stationary:
        var= data.groupby(['x1', 'x2','x3']).var()["y"].reset_index()["y"]
        term1=df['n']*var
        term1=term1[var.dropna().index]
        X_notnull=X[var.dropna().index]
        term2 =lambda a: ((a.T**2)*X_notnull).sum(axis=1)
        obj_var=lambda a: np.linalg.norm((term1-term2(a))/(X_notnull.sum(axis=1)))
        const_var={'type': 'ineq', 'fun': lambda a: power**2-a**2>=0}
        ao=np.zeros_like(num_appliances)
        optim_var=optimize.minimize(obj_var,ao, constraints=const_var)
        return optim, optim_var
    else:
        return optim,None, W,df.loc[:,"Mains"], df.iloc[:,:num_appliances]

def state_detection(app_power, min_threshold):
    # takes as input a dataframe containing timestamp as index and corresponding power values, and a minimum on power
    states=app_power>=min_threshold
    return pd.DataFrame(states.astype(np.int8))

def state_detection_multi_level(app_power,threshold):
    # states1_temp1=app_power>=threshold[0]
    # states2_temp2 =app_power<threshold[1]
    # states1= (states1_temp1==states2_temp2)
    # states2= app_power>=threshold[1]
    states1= (app_power>=threshold[0]) & (app_power<threshold[1])
    states2= (app_power>=threshold[1])
    states1= pd.DataFrame(states1.astype(np.int8))
    states2= pd.DataFrame(states2.astype(np.int8))
    states1.rename(columns={states1.columns.tolist()[0]:"{}1".format(states1.columns.tolist()[0])}, inplace=True)
    states2.rename(columns={states2.columns.tolist()[0]:"{}2".format(states2.columns.tolist()[0])}, inplace=True)
    states= states1.join(states2, how="inner")
    return states

def relative_error(measured, disagg, states_table):
    # this function calculates the different error metrics
    measured= measured[states_table.index]
    measured=measured*states_table # we are only considering the error in the on state
    measured_grouped=measured.groupby(measured.index.hour).aggregate(np.mean)
    # disagg_energy= disagg * states_table
    disagg_energy=disagg*states_table # we are only considering the error in the on state, the error in the off-state is supposed to be negligible
    disagg_grouped=disagg_energy.groupby(disagg_energy.index.hour).aggregate(np.mean)
    measured_grouped=measured_grouped[measured_grouped>0] # avodiing cases when zero consumption is recorded

    #### grouping by hour#####
    # nep= (np.abs(measured_grouped-disagg_grouped)).sum()/(measured_grouped.sum())
    # re=(np.abs(measured_grouped-disagg_grouped)/(measured_grouped)).mean()
    # if np.sum(measured)<>0:
    #     nep=np.sum(np.abs(measured-disagg_energy))/(np.sum(measured))
    #     print np.sum(np.abs(measured-disagg_energy))
    #     print np.sum(measured)
    #     re=(np.abs(measured_grouped-disagg_grouped)/(measured_grouped)).mean()
    #     return nep, re, disagg_energy, measured
    # else:
    #     return 0, 0, disagg_energy, measured

    #### resampling by hour###
    disagg_resample=disagg_energy.resample('h', how="mean")
    measured_resample=measured.resample('h', how="mean")
    if measured_resample.sum()<>0:
        nep=(np.abs(measured_resample-disagg_resample)).sum()/(measured_resample.sum())
        return nep,0,0,0 # correct this !!!!
    else:
        return np.nan,0,0,0

def error_metrics(measured, disagg, states_table):

    """This calculates NEP and NEP on the hourly aggregate basis
    """
    measured= measured[states_table.index] # aligning the measured readings with the states table
    measured=measured*states_table # we are only considering the error in the on state
    disagg_power=disagg*states_table # we are only considering the error in the on state, the error in the off-state is supposed to be negligible

    if np.sum(measured)<>0:
        nep=np.sum(np.abs(measured-disagg_power))/(np.sum(measured))
    else:
        print "NaN"
        nep=np.nan

    #### resampling by hour ###
    disagg_agg=disagg_power.resample('h', how="mean")
    measured_agg=measured.resample('h', how="mean") # taking an average of the power during this duration
    if measured_agg.sum()<>0: # since it may happen that the appliance is off for certain hour
        nep_hour=np.sum(np.abs(measured_agg-disagg_agg))/np.sum(measured_agg)
    else:
        print "NaN"
        nep_hour=np.nan

    return nep, nep_hour

def fraction_accuracy_metrics(meas_app, disagg_app, states_table):
    # note in case of vampire power => the mains values will be higher than in its absence, as a result, a lower error will be lower
    # will be reported than if there was no vampire power
    mains=states_table["Mains"]
    # print mains.head()
    # print
    # print meas_app.head()
    # print
    # print states_table.head()
    # print
    meas_app=meas_app*states_table.iloc[:,:-1]
    # print meas_app.isnull().sum()
    disagg_app=disagg_app*states_table.iloc[:,:-1]
    disagg_app_agg=disagg_app.resample('h', how="mean")
    meas_app_agg=meas_app.resample('h', how="mean")
    mains_agg= mains.resample('h', how="mean")
    num= np.sum(np.sum(np.abs(meas_app-disagg_app)))
    num_agg=np.sum(np.sum(np.abs(meas_app_agg-disagg_app_agg)))

    fraction= 1-0.5*(num/np.sum(mains))
    fraction_agg=1-0.5*(num_agg/np.sum(mains_agg))
    return fraction, fraction_agg

def relative_error_daily(meas_app, disagg_app, states_table):
    meas_app=meas_app*states_table.iloc[:,:-1]
    # print meas_app.isnull().sum()
    disagg_app=disagg_app*states_table.iloc[:,:-1]
    meas_daily=meas_app.groupby(meas_app.index.day).aggregate(np.sum)
    disagg_daily= disagg_app.groupby(disagg_app.index.day).aggregate(np.sum)
    re= np.abs(meas_daily- disagg_daily)/meas_daily
    avg_re=re.mean(axis=1)
    return re, avg_re

# Computing MSE
def mse(y,X,W,p, sigma2):
    # mean squared error for the estimator
    # y: aggregated measured power signal
    # X: states table for all the appliances
    # W: weights table
    # p: estimated power consumption
    n=y.shape[0]
    num_appliances=X.shape[1]
#     sigma2=np.square(np.linalg.norm(y-X.dot(p)))/(n-(num_appliances+1))
    try:
        mse= sigma2*np.linalg.inv(X.T.dot(W).dot(X))
        return mse
    except:
        pass

def sigma2(y,X,p):
    # to test you can use the following
    #    p=[ 111.87,	309.381,	57.732	]
    # X= [0	0	1
    #    0	1	1
    #    1	0	0
    #    1	0	1
    #    1	1	1]
    # Y= 59.3
    #    369.3
    #    120
    #    160
    #    469
    # sigma2=265.197

    n=y.shape[0]
    num_appliances=X.shape[1]
    sigma2=np.square(np.linalg.norm(y-X.dot(p)))/(n-(num_appliances+1))
    print (n-(num_appliances+1))
    return sigma2


def optimal_state(state_list, state_labels, appliance_name):
    num_states=len(state_list)
    states_table=pd.DataFrame(index=state_labels.index)
    for i in xrange(num_states):
        states_table["{}{}".format(appliance_name, i+1)]=(state_labels==i)

    return states_table.astype(np.int8)
