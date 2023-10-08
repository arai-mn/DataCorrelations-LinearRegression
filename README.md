# Housing Prices

**Course:** MGOC15 - Introductory Business Data Analytics <br>
**Topics Tested:** Data Correlations, Linear Regressions <br>

**Given Data:** 
The data provides the sale prices of different homes in a small town in Iowa, USA. 

**Final objective** is to use linear regression to predict the house prices as accurately as possible, understanding what drives housing prices and how potential home owners value different features of a house.

**The data contains the following variables:** <br>

Id: Unique ID of property
    
Neighborhood: Physical locations within city
    
LotFrontage: Linear feet of street connected to property
    
Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved

Total_sqr_footage: Total size of the house in square feet	
    
BldgType: Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
	
OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
		
YearBuilt: Original construction date

BsmtFullBath: Basement full bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

GarageCars: Size of garage in car capacity

PoolArea: Pool area in square feet

GrLivArea: Above grade (ground) living area square feet


**Target/Outcome Variable**: *SalePrice* <br>

### Importing modules and Data processing

Before we start working with our data, we will do some standard procedures for data pre-processing. 


```python
#Import Modules Here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)
```

### Read the File and Inspect Columns


```python
df_housing = pd.read_csv('Housing_prices.csv')
df_housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Neighborhood</th>
      <th>Total_sqr_footage</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>GrLivArea</th>
      <th>GarageCars</th>
      <th>LotFrontage</th>
      <th>PoolArea</th>
      <th>SalePrice</th>
      <th>Street</th>
      <th>BldgType</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BsmtFullBath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>CollgCr</td>
      <td>2566</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>1710</td>
      <td>2</td>
      <td>65.0</td>
      <td>0</td>
      <td>208500</td>
      <td>Pave</td>
      <td>1Fam</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Veenker</td>
      <td>2524</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1262</td>
      <td>2</td>
      <td>80.0</td>
      <td>0</td>
      <td>181500</td>
      <td>Pave</td>
      <td>1Fam</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>CollgCr</td>
      <td>2706</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>1786</td>
      <td>2</td>
      <td>68.0</td>
      <td>0</td>
      <td>223500</td>
      <td>Pave</td>
      <td>1Fam</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Crawfor</td>
      <td>2473</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1717</td>
      <td>3</td>
      <td>60.0</td>
      <td>0</td>
      <td>140000</td>
      <td>Pave</td>
      <td>1Fam</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>NoRidge</td>
      <td>3343</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2198</td>
      <td>3</td>
      <td>84.0</td>
      <td>0</td>
      <td>250000</td>
      <td>Pave</td>
      <td>1Fam</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_housing.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Total_sqr_footage</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>GrLivArea</th>
      <th>GarageCars</th>
      <th>LotFrontage</th>
      <th>PoolArea</th>
      <th>SalePrice</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BsmtFullBath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>2567.048630</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1515.463699</td>
      <td>1.767123</td>
      <td>70.049958</td>
      <td>2.758904</td>
      <td>180921.195890</td>
      <td>1.565068</td>
      <td>0.382877</td>
      <td>0.425342</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>821.714421</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>525.480383</td>
      <td>0.747315</td>
      <td>24.284752</td>
      <td>40.177307</td>
      <td>79442.502883</td>
      <td>0.550916</td>
      <td>0.502885</td>
      <td>0.518911</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>334.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>34900.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>2009.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1129.500000</td>
      <td>1.000000</td>
      <td>59.000000</td>
      <td>0.000000</td>
      <td>129975.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>2474.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1464.000000</td>
      <td>2.000000</td>
      <td>69.000000</td>
      <td>0.000000</td>
      <td>163000.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>3004.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>1776.750000</td>
      <td>2.000000</td>
      <td>80.000000</td>
      <td>0.000000</td>
      <td>214000.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>11752.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>5642.000000</td>
      <td>4.000000</td>
      <td>313.000000</td>
      <td>738.000000</td>
      <td>755000.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Data Inconsistencies: dealing with outliers and missing data

We will deal with outliers and missing data in the following columns only:

1. Total_sqr_footage
2. OverallQual
3. LotFrontage
4. SalePrice

Naturally, all subsequent analysis is on the updated dataframe (without outliers or missing data in the above columns).


```python
df_housing.isnull().sum()
```




    Id                     0
    Neighborhood           0
    Total_sqr_footage      0
    OverallQual            0
    OverallCond            0
    YearBuilt              0
    GrLivArea              0
    GarageCars             0
    LotFrontage          259
    PoolArea               0
    SalePrice              0
    Street                 0
    BldgType               0
    FullBath               0
    HalfBath               0
    BsmtFullBath           0
    dtype: int64



We need to deal with the missing data in the LotFrontage column first. 
We will use the Kendall-Tau correlation coefficient to identify the column that is correlated with LotFrontage the strongest to use for categorical imputation. 


```python
df_housing.corr('kendall')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Total_sqr_footage</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>GrLivArea</th>
      <th>GarageCars</th>
      <th>LotFrontage</th>
      <th>PoolArea</th>
      <th>SalePrice</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BsmtFullBath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Id</th>
      <td>1.000000</td>
      <td>-0.006009</td>
      <td>-0.020898</td>
      <td>0.002698</td>
      <td>-0.004035</td>
      <td>0.002175</td>
      <td>0.009987</td>
      <td>-0.022489</td>
      <td>0.045303</td>
      <td>-0.012030</td>
      <td>0.005745</td>
      <td>0.002064</td>
      <td>0.003751</td>
    </tr>
    <tr>
      <th>Total_sqr_footage</th>
      <td>-0.006009</td>
      <td>1.000000</td>
      <td>0.522757</td>
      <td>-0.161402</td>
      <td>0.283426</td>
      <td>0.696934</td>
      <td>0.468802</td>
      <td>0.310275</td>
      <td>0.054162</td>
      <td>0.640582</td>
      <td>0.504823</td>
      <td>0.202828</td>
      <td>0.130470</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>-0.020898</td>
      <td>0.522757</td>
      <td>1.000000</td>
      <td>-0.152513</td>
      <td>0.505804</td>
      <td>0.464189</td>
      <td>0.543120</td>
      <td>0.193309</td>
      <td>0.050609</td>
      <td>0.669660</td>
      <td>0.513944</td>
      <td>0.265575</td>
      <td>0.087385</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>0.002698</td>
      <td>-0.161402</td>
      <td>-0.152513</td>
      <td>1.000000</td>
      <td>-0.329379</td>
      <td>-0.118681</td>
      <td>-0.226809</td>
      <td>-0.065029</td>
      <td>-0.005175</td>
      <td>-0.103492</td>
      <td>-0.241425</td>
      <td>-0.065978</td>
      <td>-0.048669</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>-0.004035</td>
      <td>0.283426</td>
      <td>0.505804</td>
      <td>-0.329379</td>
      <td>1.000000</td>
      <td>0.191389</td>
      <td>0.491814</td>
      <td>0.138978</td>
      <td>0.007335</td>
      <td>0.470960</td>
      <td>0.437111</td>
      <td>0.199885</td>
      <td>0.132515</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>0.002175</td>
      <td>0.696934</td>
      <td>0.464189</td>
      <td>-0.118681</td>
      <td>0.191389</td>
      <td>1.000000</td>
      <td>0.404789</td>
      <td>0.261243</td>
      <td>0.055744</td>
      <td>0.543942</td>
      <td>0.539408</td>
      <td>0.355922</td>
      <td>0.007298</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.009987</td>
      <td>0.468802</td>
      <td>0.543120</td>
      <td>-0.226809</td>
      <td>0.491814</td>
      <td>0.404789</td>
      <td>1.000000</td>
      <td>0.278278</td>
      <td>0.020564</td>
      <td>0.572168</td>
      <td>0.487084</td>
      <td>0.216737</td>
      <td>0.134372</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>-0.022489</td>
      <td>0.310275</td>
      <td>0.193309</td>
      <td>-0.065029</td>
      <td>0.138978</td>
      <td>0.261243</td>
      <td>0.278278</td>
      <td>1.000000</td>
      <td>0.069963</td>
      <td>0.290361</td>
      <td>0.180774</td>
      <td>0.079747</td>
      <td>0.070795</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>0.045303</td>
      <td>0.054162</td>
      <td>0.050609</td>
      <td>-0.005175</td>
      <td>0.007335</td>
      <td>0.055744</td>
      <td>0.020564</td>
      <td>0.069963</td>
      <td>1.000000</td>
      <td>0.047800</td>
      <td>0.041651</td>
      <td>0.027284</td>
      <td>0.068642</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>-0.012030</td>
      <td>0.640582</td>
      <td>0.669660</td>
      <td>-0.103492</td>
      <td>0.470960</td>
      <td>0.543942</td>
      <td>0.572168</td>
      <td>0.290361</td>
      <td>0.047800</td>
      <td>1.000000</td>
      <td>0.518693</td>
      <td>0.278698</td>
      <td>0.183182</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>0.005745</td>
      <td>0.504823</td>
      <td>0.513944</td>
      <td>-0.241425</td>
      <td>0.437111</td>
      <td>0.539408</td>
      <td>0.487084</td>
      <td>0.180774</td>
      <td>0.041651</td>
      <td>0.518693</td>
      <td>1.000000</td>
      <td>0.152882</td>
      <td>-0.055522</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>0.002064</td>
      <td>0.202828</td>
      <td>0.265575</td>
      <td>-0.065978</td>
      <td>0.199885</td>
      <td>0.355922</td>
      <td>0.216737</td>
      <td>0.079747</td>
      <td>0.027284</td>
      <td>0.278698</td>
      <td>0.152882</td>
      <td>1.000000</td>
      <td>-0.041701</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.003751</td>
      <td>0.130470</td>
      <td>0.087385</td>
      <td>-0.048669</td>
      <td>0.132515</td>
      <td>0.007298</td>
      <td>0.134372</td>
      <td>0.070795</td>
      <td>0.068642</td>
      <td>0.183182</td>
      <td>-0.055522</td>
      <td>-0.041701</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



As can be seen above, Total_sqr_footage has the strongest correlation with LotFrontage. 


```python
#we choose 2700 as our split as it lies between the 50th and 75th percentiles
split = 2700 

dfLowTSF = df_housing[df_housing['Total_sqr_footage'] <=split]
dfHighTSF = df_housing[df_housing['Total_sqr_footage'] >split]

LF_LowTSF = dfLowTSF['LotFrontage'].mean()
LF_HighTSF = dfHighTSF['LotFrontage'].mean()

print(LF_LowTSF, LF_HighTSF)

df_housing.loc[(df_housing['LotFrontage'].isnull()) & (df_housing['Total_sqr_footage']>split), 'LotFrontage'] = LF_HighTSF
df_housing.loc[(df_housing['LotFrontage'].isnull()) & (df_housing['Total_sqr_footage']<=split), 'LotFrontage'] = LF_LowTSF

```

    63.974530831099194 80.01098901098901



```python
df_housing['LotFrontage'].describe()
```




    count    1460.000000
    mean       70.169437
    std        22.276689
    min        21.000000
    25%        60.000000
    50%        68.000000
    75%        80.010989
    max       313.000000
    Name: LotFrontage, dtype: float64




```python
df_housing.isnull().sum()
```




    Id                   0
    Neighborhood         0
    Total_sqr_footage    0
    OverallQual          0
    OverallCond          0
    YearBuilt            0
    GrLivArea            0
    GarageCars           0
    LotFrontage          0
    PoolArea             0
    SalePrice            0
    Street               0
    BldgType             0
    FullBath             0
    HalfBath             0
    BsmtFullBath         0
    dtype: int64



Now we can deal with outliers. Let's start off with Total_sqr_footage. First, we will see if there are any outliers by drawing a box plot.  

Total_sqr_footage has outliers. We will remove the outliers. 


```python
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
df_housing['Total_sqr_footage'].plot.box()
plt.title('Total_sqr_footage with Outliers')
plt.subplot(1, 2, 2)
thr = df_housing['Total_sqr_footage'].mean() + 5*df_housing['Total_sqr_footage'].std()
df_noTSFout = df_housing[df_housing['Total_sqr_footage'] < thr]
df_noTSFout['Total_sqr_footage'].plot.box()
plt.title('Total_sqr_footage without Outliers')
```




    Text(0.5, 1.0, 'Total_sqr_footage without Outliers')




    
![png](/images/output_16_1.png)
    



```python
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
df_noTSFout['OverallQual'].plot.box()
plt.title('OverallQual with Outliers')
plt.subplot(1, 2, 2)
thrOQ = df_noTSFout['OverallQual'].mean() - 3*df_noTSFout['OverallQual'].std()
df_noOQout = df_noTSFout[df_noTSFout['OverallQual'] > thrOQ]
df_noOQout['OverallQual'].plot.box()
plt.title('OverallQual without Outliers')
```




    Text(0.5, 1.0, 'OverallQual without Outliers')




    
![png](/images/output_17_1.png)
    



```python
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
df_noOQout['LotFrontage'].plot.box()
plt.title('LotFrontage with outliers')
plt.subplot(1, 2, 2)
thrLF = df_noOQout['LotFrontage'].mean() + 5*df_noOQout['LotFrontage'].std()
df_noLFout = df_noOQout[df_noOQout['LotFrontage'] < thrLF]
df_noLFout['LotFrontage'].plot.box()
plt.title('LotFrontage without outliers')
```




    Text(0.5, 1.0, 'LotFrontage without outliers')




    
![png](/images/output_18_1.png)
    



```python
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
df_noLFout['SalePrice'].plot.box()
plt.title('SalePrice with outliers')
plt.subplot(1, 2, 2)
thrSP = df_noLFout['SalePrice'].mean() + 5*df_noLFout['SalePrice'].std()
df_final = df_noLFout[df_noLFout['SalePrice'] < thrSP]
df_final['SalePrice'].plot.box()
plt.title('SalePrice without outliers')
```




    Text(0.5, 1.0, 'SalePrice without outliers')




    
![png](/images/output_19_1.png)
    


### Correlations

Before we start predicting home prices, it's important that we understand how American homes have changed over the years and how these changes drive prices. With this in mind, we use both statistical methods (correlational coefficients) and visualizations to answer the following questions.

**Part a**: Although American families are shrinking [2], it is believed that homes have gotten bigger over the years. Support or refute this statement using the dataset.

**Part b**: Let's talk home prices. What are home owners willing to pay more for - more living area (in square feet) or a better quality home? 


```python
#Solution to part a
yearbyfootage = df_final.groupby('YearBuilt')['Total_sqr_footage'].mean()
yearbyfootage.plot.line()
plt.title('Mean Total_sqr_footage by Year')
```




    Text(0.5, 1.0, 'Mean Total_sqr_footage by Year')




    
![png](/images/output_21_1.png)
    


The statement is not correct. As can be seen in the line graph, the average size of the houses has been going up and down over the years. There is no clear trend. 


```python
df_final['OverallQual'].describe()
```




    count    1449.000000
    mean        6.087647
    std         1.351250
    min         2.000000
    25%         5.000000
    50%         6.000000
    75%         7.000000
    max        10.000000
    Name: OverallQual, dtype: float64




```python
df_final['GrLivArea'].describe()
```




    count    1449.000000
    mean     1504.359558
    std       491.718296
    min       438.000000
    25%      1128.000000
    50%      1458.000000
    75%      1774.000000
    max      3608.000000
    Name: GrLivArea, dtype: float64



To make a comparison, we need to create two dataframes - one for high quality homes and another for homes with large living area. To consider a home high quality, it needs to have an overall quality of higher than what the 75th percentile of homes have. In this case, a home needs to have a quality ranking of higher than 7 to be considered high quality. As for the homes with large living area, we will use the 75th percentile value as the cutoff. Therefore, a home would be considered to have large living area if it had more than 1774 square feet living area.   


```python
#Solution to part b
df_highQuality = df_final[df_final['OverallQual'] > 7]
df_largeGrLivArea = df_final[df_final['GrLivArea'] > 1774]
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
df_highQuality['SalePrice'].plot.box()
plt.subplot(1, 2, 2)
df_largeGrLivArea['SalePrice'].plot.box()
print(df_highQuality['SalePrice'].mean(), df_largeGrLivArea['SalePrice'].mean())
```

    298148.1126126126 251598.68698060943



    
![png](/images/output_26_1.png)
    


#Solution to part b: as it can be seen in the two box plots above, on average, people are willing to pay 298148 dollars for high quality homes, whereas they, on average, pay 251598 dollars for homes with large living area. Therefore, it can be concluded that people are willing to pay more for better quality. 

### Feature Engineering: modifying/transforming existing data

Feature engineering is the process of creating new columns in a dataset by modifying or transforming existing data. These new columns are more amenable to linear regression. Based on this, let's answer the following question.

There are three categorical columns in our dataset - *Neighborhood, BldgType, Street*. We will ignore Neighborhood for the time being as there are too many categories (consider this for the bonus question) and we will also ignore *Street* (you can see why once you do a value counts).

Let us consider the building type column. 

**Part a**: Do you think including the building type will allow us to predict home prices more accurately? Why/Why not?

**Part b**: There are multiple categories here, so we cannot simply convert this into a 0/1 column. Instead we will attempt something simpler. What do you think is the single building type that is most helpful in making a prediction? Let's call this T. Create a column called **IsT** that equals one if the building type is T and is zero otherwise.


#Solution to part a: Yes, including the building type would allow to increase the accuracy of predicting home prices. By including the Building type, we can examine how and to what extend the building type affects the home prices. 

Solution to part a: Yes. The building type can be a heplful feature to add to predict home prices more accurately. That's because building types are usually correlated with the sale price, meaning it is a relevant data to add. 


```python
#Solution to part b
df_final['BldgType'].value_counts()
```




    1Fam      1209
    TwnhsE     114
    Duplex      52
    Twnhs       43
    2fmCon      31
    Name: BldgType, dtype: int64




```python
df_final['Is1Fam'] = 0
df_final.loc[df_final['BldgType']=='1Fam', 'Is1Fam'] = 1
```


```python
df_final['Is1Fam'].value_counts()
```




    1    1209
    0     240
    Name: Is1Fam, dtype: int64



### Q3: Brute Force Linear Regression

It is natural to think that our linear regression must take into account every single feature in our dataset, right? After all, more data never hurt anybody.

Run a linear regression using all possible features in the dataset (you can exclude feature "Id" and all the categorical features). Report the RMSE and R^2 error. 


```python
#When you use train_test_split, do not forget to include random_state = 0
```

**Important**: The random_state=0 ensures that even though the training-test split is done randomly, when you run the code multiple times, the final answer does not change. In other words, you don't want the grader to get a different R^2 error than you.


```python
x_cols = ['LotFrontage', 'Total_sqr_footage','Is1Fam','OverallQual', 
          'OverallCond', 'YearBuilt', 'BsmtFullBath', 'FullBath', 
         'HalfBath', 'GarageCars', 'PoolArea', 'GrLivArea']
y_col = ['SalePrice']

# Choose features involved in the prediction
dfX = df_final[x_cols]

# Choose column to predict
dfY = df_final[y_col]

# Break the data
X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfY, test_size=0.3,random_state=0)
print('Traning and test data split: ', len(X_train)/len(dfX),':', len(X_test)/len(dfX))

# Create linear regression object
linearRegression = LinearRegression()

# Fit data
linearRegression.fit(X_train, Y_train)

pd.DataFrame(np.transpose(linearRegression.coef_), x_cols, ['Regression Coeffs'])

Y_predicted = linearRegression.predict(X_test)

# Check error metrics
# Mean squared error
meanSquaredError = metrics.mean_squared_error(Y_test, Y_predicted)
print("Mean Squared Error: ", meanSquaredError)

# Mean root squared error
meanRootError = np.sqrt(meanSquaredError)
print("MeanRootError: ", meanRootError)

#Calculate R^2 score, aka goodness of fit
print('r2_score: ', r2_score(Y_test, Y_predicted))
```

    Traning and test data split:  0.6997929606625258 : 0.3002070393374741
    Mean Squared Error:  815541017.9977646
    MeanRootError:  28557.678792187657
    r2_score:  0.8497175477958767


### Improving the brute-force regression 

You must have noticed by now that the brute-force regression has a reasonable R^2 error. How can we improve upon this? Is that even possible since we've already used all of the features up our sleeve?

We will use feature selection. Use the Kendall-Tau correlation coefficient to identify 4 features that are strongly correlated with the outcome variable *SalePrice* but not necessarily correlated with each other (at least not strongly). Think of these as "diverse features".

Run a linear regression with just these four features. In other words, the dfX dataframe should only contain 4 columns. What is the R-squared in this case? 

Answer (very) briefly on why you think selecting just 4 features gives us an R^2 error that is so close to the actual R^2 error in Q2. In other words, economists always claim that more data is useful but here we see that whether we use 4 features or several, the error is almost identical. Why is this?


```python
df_final.corr('kendall')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Total_sqr_footage</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>GrLivArea</th>
      <th>GarageCars</th>
      <th>LotFrontage</th>
      <th>PoolArea</th>
      <th>SalePrice</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BsmtFullBath</th>
      <th>Is1Fam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Id</th>
      <td>1.000000</td>
      <td>-0.010322</td>
      <td>-0.026038</td>
      <td>0.002563</td>
      <td>-0.007005</td>
      <td>-0.001851</td>
      <td>0.005719</td>
      <td>-0.025947</td>
      <td>0.030589</td>
      <td>-0.015919</td>
      <td>0.001254</td>
      <td>0.000634</td>
      <td>0.000409</td>
      <td>-0.015457</td>
    </tr>
    <tr>
      <th>Total_sqr_footage</th>
      <td>-0.010322</td>
      <td>1.000000</td>
      <td>0.516230</td>
      <td>-0.164125</td>
      <td>0.278189</td>
      <td>0.694093</td>
      <td>0.462561</td>
      <td>0.341973</td>
      <td>0.031686</td>
      <td>0.638799</td>
      <td>0.498980</td>
      <td>0.198112</td>
      <td>0.126777</td>
      <td>0.081254</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>-0.026038</td>
      <td>0.516230</td>
      <td>1.000000</td>
      <td>-0.156039</td>
      <td>0.503391</td>
      <td>0.457193</td>
      <td>0.537146</td>
      <td>0.203064</td>
      <td>0.023834</td>
      <td>0.667681</td>
      <td>0.506974</td>
      <td>0.261731</td>
      <td>0.082438</td>
      <td>0.023974</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>0.002563</td>
      <td>-0.164125</td>
      <td>-0.156039</td>
      <td>1.000000</td>
      <td>-0.331857</td>
      <td>-0.121809</td>
      <td>-0.231307</td>
      <td>-0.070241</td>
      <td>0.006840</td>
      <td>-0.106866</td>
      <td>-0.246665</td>
      <td>-0.063268</td>
      <td>-0.046623</td>
      <td>0.167692</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>-0.007005</td>
      <td>0.278189</td>
      <td>0.503391</td>
      <td>-0.331857</td>
      <td>1.000000</td>
      <td>0.185937</td>
      <td>0.489170</td>
      <td>0.141747</td>
      <td>-0.013244</td>
      <td>0.470646</td>
      <td>0.434699</td>
      <td>0.196919</td>
      <td>0.130299</td>
      <td>-0.106705</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>-0.001851</td>
      <td>0.694093</td>
      <td>0.457193</td>
      <td>-0.121809</td>
      <td>0.185937</td>
      <td>1.000000</td>
      <td>0.398457</td>
      <td>0.283694</td>
      <td>0.033482</td>
      <td>0.541137</td>
      <td>0.534302</td>
      <td>0.353478</td>
      <td>0.001683</td>
      <td>0.075491</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.005719</td>
      <td>0.462561</td>
      <td>0.537146</td>
      <td>-0.231307</td>
      <td>0.489170</td>
      <td>0.398457</td>
      <td>1.000000</td>
      <td>0.286107</td>
      <td>0.002290</td>
      <td>0.567890</td>
      <td>0.479772</td>
      <td>0.213123</td>
      <td>0.131624</td>
      <td>0.016188</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>-0.025947</td>
      <td>0.341973</td>
      <td>0.203064</td>
      <td>-0.070241</td>
      <td>0.141747</td>
      <td>0.283694</td>
      <td>0.286107</td>
      <td>1.000000</td>
      <td>0.044042</td>
      <td>0.313255</td>
      <td>0.202592</td>
      <td>0.089943</td>
      <td>0.069210</td>
      <td>0.295824</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>0.030589</td>
      <td>0.031686</td>
      <td>0.023834</td>
      <td>0.006840</td>
      <td>-0.013244</td>
      <td>0.033482</td>
      <td>0.002290</td>
      <td>0.044042</td>
      <td>1.000000</td>
      <td>0.041009</td>
      <td>0.017636</td>
      <td>0.002987</td>
      <td>0.045007</td>
      <td>0.026200</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>-0.015919</td>
      <td>0.638799</td>
      <td>0.667681</td>
      <td>-0.106866</td>
      <td>0.470646</td>
      <td>0.541137</td>
      <td>0.567890</td>
      <td>0.313255</td>
      <td>0.041009</td>
      <td>1.000000</td>
      <td>0.514277</td>
      <td>0.277556</td>
      <td>0.182433</td>
      <td>0.110582</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>0.001254</td>
      <td>0.498980</td>
      <td>0.506974</td>
      <td>-0.246665</td>
      <td>0.434699</td>
      <td>0.534302</td>
      <td>0.479772</td>
      <td>0.202592</td>
      <td>0.017636</td>
      <td>0.514277</td>
      <td>1.000000</td>
      <td>0.147492</td>
      <td>-0.061127</td>
      <td>-0.096115</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>0.000634</td>
      <td>0.198112</td>
      <td>0.261731</td>
      <td>-0.063268</td>
      <td>0.196919</td>
      <td>0.353478</td>
      <td>0.213123</td>
      <td>0.089943</td>
      <td>0.002987</td>
      <td>0.277556</td>
      <td>0.147492</td>
      <td>1.000000</td>
      <td>-0.046796</td>
      <td>0.045494</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.000409</td>
      <td>0.126777</td>
      <td>0.082438</td>
      <td>-0.046623</td>
      <td>0.130299</td>
      <td>0.001683</td>
      <td>0.131624</td>
      <td>0.069210</td>
      <td>0.045007</td>
      <td>0.182433</td>
      <td>-0.061127</td>
      <td>-0.046796</td>
      <td>1.000000</td>
      <td>-0.042243</td>
    </tr>
    <tr>
      <th>Is1Fam</th>
      <td>-0.015457</td>
      <td>0.081254</td>
      <td>0.023974</td>
      <td>0.167692</td>
      <td>-0.106705</td>
      <td>0.075491</td>
      <td>0.016188</td>
      <td>0.295824</td>
      <td>0.026200</td>
      <td>0.110582</td>
      <td>-0.096115</td>
      <td>0.045494</td>
      <td>-0.042243</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Choose features involved in the prediction
dfX = df_final[['Total_sqr_footage', 'OverallQual', 'GarageCars','GrLivArea']]

# Choose column to predict
dfY = df_final['SalePrice']

# Break the data
X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfY, test_size=0.3,random_state=0)
print('Traning and test data split: ', len(X_train)/len(dfX),':', len(X_test)/len(dfX))

# Create linear regression object
linearRegression = LinearRegression()

# Fit data
linearRegression.fit(X_train, Y_train)

pd.DataFrame(linearRegression.coef_, dfX.columns, ['Regression Coeffs'])

Y_predicted = linearRegression.predict(X_test)

# Check error metrics
# Mean squared error
meanSquaredError = metrics.mean_squared_error(Y_test, Y_predicted)
print("Mean Squared Error: ", meanSquaredError)

# Mean root squared error
meanRootError = np.sqrt(meanSquaredError)
print("MeanRootError: ", meanRootError)

#Calculate R^2 score, aka goodness of fit
print('r2_score: ', r2_score(Y_test, Y_predicted))
```

    Traning and test data split:  0.6997929606625258 : 0.3002070393374741
    Mean Squared Error:  954286764.3922529
    MeanRootError:  30891.532244164468
    r2_score:  0.8241504082640767


What is important when it comes to building an accurate predictive model is not the number of features selected, but rather whether those features that are highly correlated with the target feature have been selected. This is because some features can be irrelevant and decrease the model's predictive accuracy.  

### Data analysis Interpretation

Based on your analysis, what do you think are the most informative features (columns), i.e., what are potential buyers most influenced by when they buy a house? Name at least two features (and at most 4) for this answer.

**Note:** You may have run multiple regressions by this point. Use any of them to answer this question.


```python
# Choose features involved in the prediction
dfX = df_final[['Total_sqr_footage', 'OverallQual']]

# Choose column to predict
dfY = df_final['SalePrice']

# Break the data
X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfY, test_size=0.3,random_state=0)
print('Traning and test data split: ', len(X_train)/len(dfX),':', len(X_test)/len(dfX))

# Create linear regression object
linearRegression = LinearRegression()

# Fit data
linearRegression.fit(X_train, Y_train)

pd.DataFrame(linearRegression.coef_, dfX.columns, ['Regression Coeffs'])

Y_predicted = linearRegression.predict(X_test)

# Check error metrics
# Mean squared error
meanSquaredError = metrics.mean_squared_error(Y_test, Y_predicted)
print("Mean Squared Error: ", meanSquaredError)

# Mean root squared error
meanRootError = np.sqrt(meanSquaredError)
print("MeanRootError: ", meanRootError)

#Calculate R^2 score, aka goodness of fit
print('r2_score: ', r2_score(Y_test, Y_predicted))
```

    Traning and test data split:  0.6997929606625258 : 0.3002070393374741
    Mean Squared Error:  1000962061.8565565
    MeanRootError:  31637.984478417024
    r2_score:  0.8155493961684328


Total_sqr_footage and OverallQual are the two most informative features. 

### Training Test Split 

In all our experiments so far, we have used a 70:30 split for the training and test data. We need to verify if this 70:30 split is sacred or if other ratios are also okay. To answer this question, run the linear regression in Q2 again with a a) 50:50 split and b) 90:10 split. 

What are the RMSE and R^2 error in both cases. Based on your experiments, can you conclude which of the three (50:50, 70:30, 90:10) is the best ratio for a training-test split. 



```python
x_cols = ['LotFrontage', 'Total_sqr_footage','Is1Fam','OverallQual', 
          'OverallCond', 'YearBuilt', 'BsmtFullBath', 'FullBath', 
         'HalfBath', 'GarageCars', 'PoolArea', 'GrLivArea']
y_col = ['SalePrice']

# Choose features involved in the prediction
dfX = df_final[x_cols]

# Choose column to predict
dfY = df_final[y_col]

# Break the data
X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfY, test_size=0.5,random_state=0)
print('Traning and test data split: ', len(X_train)/len(dfX),':', len(X_test)/len(dfX))

# Create linear regression object
linearRegression = LinearRegression()

# Fit data
linearRegression.fit(X_train, Y_train)

pd.DataFrame(np.transpose(linearRegression.coef_), x_cols, ['Regression Coeffs'])

Y_predicted = linearRegression.predict(X_test)

# Check error metrics
# Mean squared error
meanSquaredError = metrics.mean_squared_error(Y_test, Y_predicted)
print("Mean Squared Error: ", meanSquaredError)

# Mean root squared error
meanRootError = np.sqrt(meanSquaredError)
print("MeanRootError: ", meanRootError)
#Calculate R^2 score, aka goodness of fit
print('r2_score: ', r2_score(Y_test, Y_predicted))
```

    Traning and test data split:  0.4996549344375431 : 0.5003450655624568
    Mean Squared Error:  787958597.3313371
    MeanRootError:  28070.600231048447
    r2_score:  0.8475411139959701



```python
x_cols = ['LotFrontage', 'Total_sqr_footage','Is1Fam','OverallQual', 
          'OverallCond', 'YearBuilt', 'BsmtFullBath', 'FullBath', 
         'HalfBath', 'GarageCars', 'PoolArea', 'GrLivArea']
y_col = ['SalePrice']

# Choose features involved in the prediction
dfX = df_final[x_cols]

# Choose column to predict
dfY = df_final[y_col]

# Break the data
X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfY, test_size=0.1,random_state=0)
print('Traning and test data split: ', len(X_train)/len(dfX),':', len(X_test)/len(dfX))

# Create linear regression object
linearRegression = LinearRegression()

# Fit data
linearRegression.fit(X_train, Y_train)

pd.DataFrame(np.transpose(linearRegression.coef_), x_cols, ['Regression Coeffs'])

Y_predicted = linearRegression.predict(X_test)

# Check error metrics
# Mean squared error
meanSquaredError = metrics.mean_squared_error(Y_test, Y_predicted)
print("Mean Squared Error: ", meanSquaredError)

# Mean root squared error
meanRootError = np.sqrt(meanSquaredError)
print("MeanRootError: ", meanRootError)
#Calculate R^2 score, aka goodness of fit
print('r2_score: ', r2_score(Y_test, Y_predicted))
```

    Traning and test data split:  0.8999309868875086 : 0.10006901311249138
    Mean Squared Error:  802695104.7805327
    MeanRootError:  28331.874360524274
    r2_score:  0.8527449288758877


Based on this experiment, it looks like the 90:10 split produces the best r2_score. 

### Improve upon the R-squared

The two teams with the best R-squared errors (highest) will receive one bonus point each if they can improve upon the error in Q2/Q3. If more than two teams can beat the R^2 error, then they will all receive extra 0.5 points.

Hint 1: How can we incorporate Neighborhood as a feature?
    
Hint 2: Try to identify features that are correlated with SalePrice but not so much with the other features, i.e., our input features need to be complementary to each other.

Hint 3: Start with the best R^2 error from Q2 or Q3 and try adding or removing a single feature.

Please don't waste too much time on this question. Your final answer must use a 70:30 split.


```python
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_final['Neighborhood'] = labelencoder.fit_transform(df_final['Neighborhood'])
df_final['BldgType'] = labelencoder.fit_transform(df_final['BldgType'])
df_final.loc[df_final['Street'] == 'Pave', 'Street'] = 1
df_final.loc[df_final['Street'] == 'Grvl', 'Street'] = 0
df_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Neighborhood</th>
      <th>Total_sqr_footage</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>GrLivArea</th>
      <th>GarageCars</th>
      <th>LotFrontage</th>
      <th>PoolArea</th>
      <th>SalePrice</th>
      <th>Street</th>
      <th>BldgType</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BsmtFullBath</th>
      <th>Is1Fam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2566</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>1710</td>
      <td>2</td>
      <td>65.0</td>
      <td>0</td>
      <td>208500</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>24</td>
      <td>2524</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1262</td>
      <td>2</td>
      <td>80.0</td>
      <td>0</td>
      <td>181500</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>2706</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>1786</td>
      <td>2</td>
      <td>68.0</td>
      <td>0</td>
      <td>223500</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>6</td>
      <td>2473</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1717</td>
      <td>3</td>
      <td>60.0</td>
      <td>0</td>
      <td>140000</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15</td>
      <td>3343</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2198</td>
      <td>3</td>
      <td>84.0</td>
      <td>0</td>
      <td>250000</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_cols = ['LotFrontage', 'Total_sqr_footage','OverallQual', 'OverallCond', 'BsmtFullBath',
          'YearBuilt', 'GarageCars', 'FullBath', 
           'GrLivArea', 'Neighborhood', 'Street', 'BldgType']
y_col = ['SalePrice']

# Choose features involved in the prediction
dfX = df_final[x_cols]

# Choose column to predict
dfY = df_final[y_col]

# Break the data
X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfY, test_size=0.3,random_state=0)
print('Traning and test data split: ', len(X_train)/len(dfX),':', len(X_test)/len(dfX))

# Create linear regression object
linearRegression = LinearRegression()

# Fit data
linearRegression.fit(X_train, Y_train)

pd.DataFrame(np.transpose(linearRegression.coef_), x_cols, ['Regression Coeffs'])

Y_predicted = linearRegression.predict(X_test)

# Check error metrics
# Mean squared error
meanSquaredError = metrics.mean_squared_error(Y_test, Y_predicted)
print("Mean Squared Error: ", meanSquaredError)

# Mean root squared error
meanRootError = np.sqrt(meanSquaredError)
print("MeanRootError: ", meanRootError)
#Calculate R^2 score, aka goodness of fit
print('r2_score: ', r2_score(Y_test, Y_predicted))
```

    Traning and test data split:  0.6997929606625258 : 0.3002070393374741
    Mean Squared Error:  757280619.9116006
    MeanRootError:  27518.732163956982
    r2_score:  0.8604533848629967



```python

```
