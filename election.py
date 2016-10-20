#!/Users/christopherspringob/anaconda/bin/python

import matplotlib as mpl
import matplotlib.pyplot as pp
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats

f1 = open("correlations2.txt","w")
f2 = open("test.txt","w")
f3 = open("test2.txt","w")
f4 = open("test3.txt","w")

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


#Now merge the tables together:
repubtable = pd.merge(cruztableb,kasichtableb,left_on='fips',right_on='fips')
repubtable2 = pd.merge(repubtable,trumptableb,left_on='fips',right_on='fips')

#elecun=elec.pivot_table(values=["votes","fraction_votes"],index=["state","fips"],columns="candidate")

#And the demographics too:
biggertable = pd.merge(repubtable2,demos,left_on='fips',right_on='fips')

#Make a table for each candidate, only counting counties with more than 2000 votes:

biggertable['totvotes']=biggertable['votes']/biggertable['fraction_votes']
bigtable = biggertable.loc[biggertable['totvotes'] > 2000]


droptable = bigtable.loc[bigtable['dropout'] < 1]

#droptable = bigtable.loc[(bigtable['dropout'] == 0) & (bigtable['votes'] > 0)]


#elecun.to_csv('elecout.csv')








#print(biggertable.describe())
#print(bigtable.describe())

"""cruztable = bigtable.loc[(bigtable['candidate'] == 'Ted Cruz') & (bigtable['dropout'] == 0)]
kasichtable = bigtable.loc[(bigtable['candidate'] == 'John Kasich') & (bigtable['dropout'] == 0)]
trumptable = bigtable.loc[(bigtable['candidate'] == 'Donald Trump') & (bigtable['dropout'] == 0)]
clintontable = bigtable.loc[bigtable['candidate'] == 'Hillary Clinton']
sanderstable = bigtable.loc[bigtable['candidate'] == 'Bernie Sanders']"""

print(scipy.stats.pearsonr(droptable['fraction_votes_x'],droptable['AGE295214']))
print(scipy.stats.pearsonr(droptable['fraction_votes_y'],droptable['EDU635213']))

print(droptable.describe())


print(bigtable[bigtable['dropout'] == 1],file=f2)

droptable.plot(kind='scatter',x='HSG495213',y='fraction_votes_x',xlim=[0,500000])

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
#cruzeduline = sc.stats.linregress(cruztable['fraction_votes'], cruztable['EDU635213'])
#cruzpopline = sc.stats.linregress(cruztable['fraction_votes'], cruztable['PST120214'])
#kasichageline = sc.stats.linregress(kasichtable['fraction_votes'], kasichtable['AGE295214'])
kasicheduline = sc.stats.linregress(droptable['EDU635213'],droptable['fraction_votes_y'])
#kasichpopline = sc.stats.linregress(kasichtable['fraction_votes'], kasichtable['PST120214'])
#trumpageline = sc.stats.linregress(trumptable['fraction_votes'], trumptable['AGE295214'])
#trumpeduline = sc.stats.linregress(trumptable['fraction_votes'], trumptable['EDU635213'])
trumppopline = sc.stats.linregress(droptable['PST120214'],droptable['fraction_votes'])

print(cruzageline)
#print(cruzeduline)
#print(cruzpopline)
#print(kasichageline)
print(kasicheduline)
#print(kasichpopline)
#print(trumpageline)
#print(trumpeduline)
print(trumppopline)

print(cruzageline[0])
print(kasicheduline[1])















#So we could just have our model be these three trendlines, with each of the three county demographics determining the vote share for one of the candidates, according to the trendlines from our regression fits.  The problem with that is that you could have counties where the vote total for the three candidates is well over 100%.....or a very low number.  We don't expect candidates other than the three remaining in the race to get many votes, so for simplicity, let's just set the  Cruz+Kasich+Trump combined voteshare to 90%.

#So we evaluate the vote share as a linear function of the three parameters, but then scale up or down the totals so that Cruz+Kasich+Trump is always at 90%.  We'll use the best fit line trends as our initial guess, and evaluate a range of values around that:

oldtot=10000.0

for im in range(0,5):
    cruzm=(cruzageline[0]-0.01)+(0.005*im)
    for ib in range(0,5):
        cruzb=(cruzageline[1]-0.01)+(0.005*ib)
        for jm in range(0,5):
            kasichm=(kasicheduline[0]-0.002)+(0.001*jm)
            for jb in range(0,5):
                kasichb=(kasicheduline[1]-0.02)+(0.01*jb)
                for km in range(0,5):
                    trumpm=(trumppopline[0]-0.01)+(0.005*km)
                    for kb in range(0,5):
                        trumpb=(trumppopline[1]-0.1)+(0.05*kb)
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
                            print(im,ib,jm,jb,km,kb,totdev,file=f4)
                            oldtot=totdev
#                            print(totest.max(),file=f4)








#cruztable.hist(column='PST120214')

print(droptable,file=f3)
print(bigtable.describe(),file=f3)
print(bigtable['state'],file=f3)
print(bigtable['state_abbreviation_y'],file=f3)

#print(cruztable['votes'])

f1.close()
f2.close()
f3.close()
f4.close()
#pp.show()
