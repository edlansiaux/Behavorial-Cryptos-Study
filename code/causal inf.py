# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 13:49:35 2022

@author: Edouard LANSIAUX
"""
from causallib.estimation import IPW
from sklearn.linear_model import LogisticRegression
from causallib.evaluation import PropensityEvaluator
import pandas
import seaborn as sns

#Datasets
data1 =  pandas.read_csv("C:/Users/PATRICE LANSIAUX/Desktop/crypto/papers/doge VS LTC/data/DOGE_final.csv")
data2 =  pandas.read_csv("C:/Users/PATRICE LANSIAUX/Desktop/crypto/papers/doge VS LTC/data/LTC_final.csv") 

 
 #Causal model
learner = LogisticRegression(penalty='none',  # No regularization, new in scikit-learn 0.21.*
                             solver='lbfgs',
                             max_iter=500)    # Increased to achieve convergence with 'lbfgs' solver
ipw = IPW(learner)

#Applying causal analysis
##DOGE
ipw.fit(data1.transactions, data1.tweets)
potential_outcomes1 = ipw.estimate_population_outcome(data1.transactions, data1.tweets, data1.price)
causal_effect1 = potential_outcomes1[1] - potential_outcomes1[0]
causal_effect1

##LTC
ipw.fit(data2.transactions, data2.tweets)
potential_outcomes2 = ipw.estimate_population_outcome(data2.transactions, data2.tweets, data2.price)
causal_effect2 = potential_outcomes2[1] - potential_outcomes2[0]
causal_effect2

#Estimating Observed Effect
##DOGE
observed_outcomes1 = data1.price.groupby(data1.tweets).mean()
observed_effect1 = observed_outcomes1[1] - observed_outcomes1[0]
observed_effect1


##LTC
observed_outcomes2 = data2.price.groupby(data2.tweets).mean()
observed_effect2 = observed_outcomes2[1] - observed_outcomes2[0]
observed_effect2

#Analysis of the difference between causal and observed effect
##DOGE
evaluator1 = PropensityEvaluator(ipw)
evaluations1 = evaluator1.evaluate_simple(data1.transactions, data1.tweets, data1.price, 
                                        plots=["covariate_balance_love"],
                                        metrics_to_evaluate={})

fig1 = evaluations1.plots["covariate_balance_love"].get_figure()
fig1.set_size_inches(6, 6);  # set a more compact size than default

ax1 = sns.scatterplot(data1.transactions["age"], data1.price, hue=data1.tweets, alpha=0.7)
ax1.get_figure().set_size_inches(9, 5);

##LTC
evaluator2 = PropensityEvaluator(ipw)
evaluations2 = evaluator2.evaluate_simple(data2.transactions, data2.tweets, data2.price, 
                                        plots=["covariate_balance_love"],
                                        metrics_to_evaluate={})

fig2 = evaluations2.plots["covariate_balance_love"].get_figure()
fig2.set_size_inches(6, 6);  # set a more compact size than default

