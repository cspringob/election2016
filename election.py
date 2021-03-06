#!/Users/christopherspringob/anaconda/bin/python

import matplotlib as mpl
import matplotlib.pyplot as pp
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats

f1 = open("correlations.txt","w")
f2 = open("bestfitparams.txt","w")
f3 = open("electionpredictions.txt","w")


#This program reads in 2016 primary results and demographic information at the county level, and constructs a simple model of the Republican primary, based on the correlation between demographic data and primary results.  The correlations are done on the primaries that were held before Cruz and Kasich dropped out of the race.  The model then predicts what the vote shares for Cruz, Kasich and Trump would have been in the remaining states had Cruz and Kasich not dropped out.

#Read in the data:
elec = pd.read_csv('primary_results.csv')

demos = pd.read_csv('county_facts.csv')

#Make tables for each of the three main Republican candidates:
cruztableb = elec.loc[elec['candidate'] == 'Ted Cruz']
kasichtableb = elec.loc[elec['candidate'] == 'John Kasich']
trumptableb = elec.loc[elec['candidate'] == 'Donald Trump']

#Add a flag that tells you whether the primary was held before or after Cruz and Kasich dropped out of the race:
cruztableb['dropout']=0
cruztableb['dropout'][cruztableb['state'] == 'Nebraska'] = 1
cruztableb['dropout'][cruztableb['state'] == 'Oregon'] = 1
cruztableb['dropout'][cruztableb['state'] == 'West Virginia'] = 1
cruztableb['dropout'][cruztableb['state'] == 'Washington'] = 1
cruztableb['dropout'][cruztableb['state'] == 'California'] = 1
cruztableb['dropout'][cruztableb['state'] == 'Montana'] = 1
cruztableb['dropout'][cruztableb['state'] == 'New Jersey'] = 1
cruztableb['dropout'][cruztableb['state'] == 'New Mexico'] = 1
cruztableb['dropout'][cruztableb['state'] == 'South Dakota'] = 1

#Also, there was no primary contest in CO, ND, or WY on the Republican side, so drop those from consideration as well by flagging with a "2":
cruztableb['dropout'][cruztableb['state'] == 'Colorado'] = 2
cruztableb['dropout'][cruztableb['state'] == 'North Dakota'] = 2
cruztableb['dropout'][cruztableb['state'] == 'Wyoming'] = 2


#Now merge the election results tables together:
repubtable = pd.merge(cruztableb,kasichtableb,left_on='fips',right_on='fips')
repubtable2 = pd.merge(repubtable,trumptableb,left_on='fips',right_on='fips')

#And the demographics too:
biggertable = pd.merge(repubtable2,demos,left_on='fips',right_on='fips')

#Now select only the counties in states that voted before Cruz and Kasich dropped out.  Also, we'll only consider counties with more than 2000 votes, so that we can cut down on statistical noise:

biggertable['totvotes']=biggertable['votes']/biggertable['fraction_votes']
bigtable = biggertable.loc[biggertable['totvotes'] > 2000]


droptable = bigtable.loc[bigtable['dropout'] < 1]


print(droptable.describe())


#droptable.plot(kind='scatter',x='HSG495213',y='fraction_votes_x',xlim=[0,500000])



#Let's calculate the correlation matrix for each candidates' support, then rank the correlations so that we can see which parameters are most important.  Put the ranked correlations in a file, so we can browse through the list:
cruzfrac = droptable.corr()['fraction_votes_x']
cruzorder = cruzfrac.order(kind="quicksort")
kasichfrac = droptable.corr()['fraction_votes_y']
kasichorder = kasichfrac.order(kind="quicksort")
trumpfrac = droptable.corr()['fraction_votes']
trumporder = trumpfrac.order(kind="quicksort")


print(cruzorder,file=f1)
print(kasichorder,file=f1)
print(trumporder,file=f1)


#Looks like the strongest trends for each GOP candidate are:
#AGE295214 for Cruz (positive correlation)
#EDU635213 for Kasich (positive correlation)
#PST120214 for Trump (negative correlation)


#So let's make a simple model just based on these three parameters.  First, do a regression line for each candidate on these three parameters, in order to get a rough guess as to what the trend is:

cruzageline = sc.stats.linregress(droptable['AGE295214'],droptable['fraction_votes_x'])

kasicheduline = sc.stats.linregress(droptable['EDU635213'],droptable['fraction_votes_y'])

trumppopline = sc.stats.linregress(droptable['PST120214'],droptable['fraction_votes'])

print(cruzageline)
print(kasicheduline)
print(trumppopline)



#So we could just have our model be these three trendlines, with each of the three county demographics determining the vote share for one of the candidates, according to the trendlines from our regression fits.  The problem with that is that you could have counties where the vote total for the three candidates is well over 100%.....or a very low number.  We don't expect candidates other than the three remaining in the race to get many votes, so for simplicity, let's just set the  Cruz+Kasich+Trump combined voteshare to 90%.

#So we evaluate the vote share as a linear function of the three parameters, but then scale up or down the totals so that Cruz+Kasich+Trump is always at 90%.  We'll use the best fit line trends as our initial guess, and evaluate a range of values around that:

oldtot=10000.0

for im in range(0,5):
    cruzm=(cruzageline[0]-0.02)+(0.01*im)
    for ib in range(0,5):
        cruzb=(cruzageline[1]-0.02)+(0.01*ib)
        for jm in range(0,5):
            kasichm=(kasicheduline[0]-0.004)+(0.002*jm)
            for jb in range(0,5):
                kasichb=(kasicheduline[1]-0.04)+(0.02*jb)
                for km in range(0,5):
                    trumpm=(trumppopline[0]-0.02)+(0.01*km)
                    for kb in range(0,5):
                        trumpb=(trumppopline[1]-0.2)+(0.1*kb)
                        cruzest=cruzm*droptable['AGE295214'].values+cruzb
                        kasichest=kasichm*droptable['EDU635213'].values+kasichb
                        trumpest=trumpm*droptable['PST120214'].values+trumpb
                        totest=cruzest+kasichest+trumpest
                        cruzest = 0.9*cruzest/totest
                        kasichest = 0.9*kasichest/totest
                        trumpest = 0.9*trumpest/totest
                        cruzdev2=(droptable['fraction_votes_x'].values-cruzest)*(droptable['fraction_votes_x'].values-cruzest)
                        kasichdev2=(droptable['fraction_votes_y'].values-kasichest)*(droptable['fraction_votes_y'].values-kasichest)
                        trumpdev2=(droptable['fraction_votes_x'].values-trumpest)*(droptable['fraction_votes_x'].values-trumpest)
                        totdev=np.sum(cruzdev2)+np.sum(kasichdev2)+np.sum(trumpdev2)
                        if(totdev<oldtot):
                            print(im,ib,jm,jb,km,kb,totdev,file=f2)
                            oldtot=totdev
                            bestim=im
                            bestib=ib
                            bestjm=jm
                            bestjb=jb
                            bestkm=km
                            bestkb=kb
#                            print(totest.max(),file=f4)


#OK, so these are the best fit parameters for the model:
bestimx=(cruzageline[0]-0.01)+(0.005*bestim)
bestibx=(cruzageline[1]-0.01)+(0.005*bestib)
bestjmx=(kasicheduline[0]-0.002)+(0.001*bestjm)
bestjbx=(kasicheduline[1]-0.02)+(0.01*bestjb)
bestkmx=(trumppopline[0]-0.01)+(0.005*bestkm)
bestkbx=(trumppopline[1]-0.1)+(0.05*bestkb)

print(bestimx,bestibx,bestjmx,bestjbx,bestkmx,bestkbx,oldtot)


#Now let's go back to the election dataframe, to select the states that voted after Cruz and Kasich dropped out.  Since we're interested in forecasting the vote totals for states rather than individual counties, we sum up the vote totals for all counties in a state:
elec['dropout']=0
elec['dropout'][elec['state'] == 'Nebraska'] = 1
elec['dropout'][elec['state'] == 'Oregon'] = 1
elec['dropout'][elec['state'] == 'West Virginia'] = 1
elec['dropout'][elec['state'] == 'Washington'] = 1
elec['dropout'][elec['state'] == 'California'] = 1
elec['dropout'][elec['state'] == 'Montana'] = 1
elec['dropout'][elec['state'] == 'New Jersey'] = 1
elec['dropout'][elec['state'] == 'New Mexico'] = 1
elec['dropout'][elec['state'] == 'South Dakota'] = 1

latestates = elec[(elec['party'] == 'Republican') & (elec['dropout'] == 1)].groupby('state',as_index=False).sum()

#print(demos)
print(latestates)


#Now merge the election dataframe for late states with the demographic info:
latetable = pd.merge(latestates,demos,left_on='state',right_on='area_name')

#And now we calculate the model's prediction for the Cruz, Kasich, and Trump vote share, setting their combined vote total to 90%:
cruzest2p = bestimx*latetable['AGE295214'].values+bestibx
kasichest2p = bestjmx*latetable['EDU635213'].values+bestjbx
trumpest2p = bestkmx*latetable['PST120214'].values+bestkbx
totest2 = cruzest2p+kasichest2p+trumpest2p
cruzest2p = 0.9*cruzest2p/totest2
kasichest2p = 0.9*kasichest2p/totest2
trumpest2p = 0.9*trumpest2p/totest2

#Now multiply that by the total vote count in each state.  We're assuming that the total number of votes wouldn't have been different with Cruz and Kasich in the race, which is probably wrong, but this at least gives us a lower limit:
cruzest2t = cruzest2p*latetable['votes'].values
kasichest2t = kasichest2p*latetable['votes'].values
trumpest2t = trumpest2p*latetable['votes'].values


#Now we print the results into a file.  This file gives us, for each of the nine states that voted after Cruz and Kasich dropped out, the model's prediction for the Trump, Cruz, and Kasich vote share and vote totals given the model predictions of how they would have done if Cruz and Kasich stayed in the race:
for i in range(0,9):
    print('{0:s} {1:6.4f} {2:6.4f} {3:6.4f} {4:7.0f} {5:7.0f} {6:7.0f}'.format(latetable['state'][i],trumpest2p[i],cruzest2p[i],kasichest2p[i],trumpest2t[i],cruzest2t[i],kasichest2t[i]),file=f3)



f1.close()
f2.close()
f3.close()
#pp.show()
