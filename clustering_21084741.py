
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:39:56 2023

@author: ALIENWARE-CERDAS
"""
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct
import numpy as np
import sklearn.cluster as cluster
from scipy.optimize import curve_fit
import pandas as pd
import errors as err
import scipy.optimize as opt

def logistics(t, a, k, t0):

  f = a / (1.0 + np.exp(-k * (t - t0)))
  return f
def poly(t, c0, c1, c2, c3):

  t = t - 1950
  f = c0 + c1*t + c2*t**2 + c3*t**3
  return f

dataset = 'changeclimate.csv'
data = pd.read_csv(dataset)
popuag_df = data[data['Indicator Name'].str.contains('Population in urban agglomerations')]
popuag_df = popuag_df.reset_index()
df_popuag = popuag_df[["1960", "1965", "1970","1980", "1990","1995", "2010", "2015","2017","2020"]]
df_popuag.describe()
corr = df_popuag.corr()
print(corr)
ct.map_corr(df_popuag)
plt.show()
pd.plotting.scatter_matrix(df_popuag, figsize=(12, 12), s=5, alpha=0.8)
plt.show()
clusterdf = df_popuag[["1980", "2010"]] # extract the two columns for clustering
clusterdf = clusterdf.dropna() # entries with one nan are useless
clusterdf = clusterdf.reset_index()
clusterdf = clusterdf.drop("index", axis=1)
print(clusterdf.iloc[0:15])
# normalise, store minimum and maximum
df_normalized, df_mini, df_maxi = ct.scaler(clusterdf)
# from sklearn import cluster
# from sklearn import cluster
import sklearn.cluster as cluster
from scipy.optimize import curve_fit
import sklearn.metrics as skmet
for ncluster in range(2, 10):
  # set up the clusterer with the number of expected clusters
  kmeans = cluster.KMeans(n_clusters=ncluster)
  # Fit the data, results are stored in the kmeans object
  kmeans.fit(df_normalized) # fit done on x,y pairs
  labels = kmeans.labels_
  # extract the estimated cluster centres
  cen = kmeans.cluster_centers_
# calculate the silhoutte score
  print(ncluster, skmet.silhouette_score(clusterdf, labels))
  
  
import numpy as np
ncluster = 7 # best number of clusters
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_normalized) # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(5.0, 5.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_normalized["1980"], df_normalized["2010"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Population in urban agglomerations (1980)")
plt.ylabel("Population in urban agglomerations (2010)")
plt.show()
# Population in urban agglomerations in US during the era from 1961 till 2021
US_df = data[data['Country Name']== 'United States']
usdf = US_df[US_df['Indicator Name']=='Population in urban agglomerations of more than 1 million (% of total population)'].T
usdf = usdf.reset_index()
usdf =usdf.iloc[4:]
usdf = usdf.rename(columns={'index': 'date',19095: 'population_ammal'})
usdf['date'] = usdf['date'].astype(int)
usdf['population_ammal'] = usdf['population_ammal'].astype(int)
# Print the updated DataFrame
usdf.info()

plt.figure()
plt.plot(usdf["date"], usdf["population_ammal"], label="data")
x_values = usdf['date'].values
subset_dates = x_values[::5]  # Select every 10th date or adjust the subset as per your requirement
plt.xticks(subset_dates, rotation=45)
plt.title("Population in urban agglomerations in United States")
print(subset_dates)
plt.show()

# find a feasible start value the pedestrian way
popt, pcorr = opt.curve_fit(logistics, usdf["date"], usdf["population_ammal"], p0=(16e8, 0.004, 1985.0), maxfev=8000)
print("Fit parameter", popt)
usdf["pop_logistics"] = logistics(usdf["date"], *popt)
plt.figure()
plt.title("logistics function")
plt.plot(usdf["date"], usdf["population_ammal"], label="data")
plt.plot(usdf["date"], usdf["pop_logistics"], label="fit")
plt.legend()
plt.show()
print("Population in urban agglomerations in United States in United States")
print("2030:", logistics(2030, *popt) / 1.0e6, "Mill.")
print("2040:", logistics(2040, *popt) / 1.0e6, "Mill.")
print("2050:", logistics(2050, *popt) / 1.0e6, "Mill.")

# extract variances and calculate sigmas
popt, pcorr = opt.curve_fit(poly, usdf["date"], usdf["population_ammal"])
print("Fit parameter", popt)
# extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))
# call function to calculate upper and lower limits with extrapolation
# create extended year range
years = np.arange(1950, 2050)
lower, upper = err.err_ranges(years, poly, popt, sigmas)
usdf["poly"] = poly(usdf["date"], *popt)
plt.figure()
plt.title("polynomial")
plt.plot(usdf["date"], usdf["population_ammal"], label="data")
plt.plot(usdf["date"], usdf["poly"], label="fit")
# plot error ranges with transparency
plt.fill_between(years, lower, upper, alpha=0.5)
plt.legend(loc="upper left")
plt.show()
