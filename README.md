## Objective 
### To understand the influence of the parents background, test preparation etc on students performance in the <br> Given Data

<b><h3>Intriduction</h3></b>
<p>Let's be real we have all wondered at some point in our lives,  how some people score more marks than us? is it due to they study more? or they come from a better background? or is it due to some other reason? more importantly, how can we improve our performace? lets try to find answers to these questions</p>

<b><p>Variable Description<p></b>
<ul>
<li><b>gender</b>:- male or female</li>
<li><b>race/ethnicity</b>:- Group A,Group B,Group C, Group D and Group E</li>
<li><b>parental level of education</b>:- bachelor's degree, some college, master's degree,associate's degree, high school and some high school</li>
<li><b>lunch</b>:- free/reduced or standard</li>
<li><b>test preparation course</b>:- completed or none</li>
<li><b>math score</b>:- integer</li>
<li><b>reading score</b>:- integer</li>
<li><b>writing score</b>:- integer</li>
</ul>


```python
df = pd.read_csv("../dataset/data.csv")
```


```python
df.head()
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
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



<p>By Looking at this Data we find we have to segregate the Students who perform well and not
For this we have cluster the data in Groups <br>
So <b>WSS Plot</b> had been used the to find Out Optimal No of Clusters for given Data. <br>
For this data <b>2</b> was the Optimal Cluster.</p>
<p>By Using the <b>K-Means++ Unsupervised Machine Learning Model</b> Data is Segregate</p>


```python
train_data.head()
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
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data['cluster'].value_counts().plot.pie(autopct = '%1.2f%%')
```




    <AxesSubplot:ylabel='cluster'>




![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_7_1.png?raw=true) 


<b>cluster</b> is the new Variable thet had been added after processing the Model to identify which cluster/Group does the Student Belong to.

<table>
    <tr>
        <th></th>
        <th>math score</th>
        <th>reading score</th>
        <th>writing score</th>
    </tr>
    <tr>
        <th>Cluster  1 Students Subjects  AVG</th>
        <th>55</th>
        <th>58</th>
        <th>56</th>
    </tr>
    <tr>
        <th>Cluster  2 Students Subjects AVG</th>
        <th>75</th>
        <th>78</th>
        <th>77</th>    
    </tr>
</table>

<p> # <b>Cluster 2</b> Students are scoring better that <b>Cluster 1</b> Students</p>
      


```python
for feature in non_numerical_feature:
    if feature != "cluster":
        print(f"--------------------------------{feature}--------------------------------------")
        print(train_data.groupby(feature)[feature].count())
        print("\n")
        print(pd.crosstab(train_data[feature], train_data["cluster"]))
        sns.countplot(feature, data = train_data, hue = 'cluster')
        plt.show()
        g = sns.factorplot(x=feature, y="cluster", data=train_data, size=6)
        g.set_ylabels("Survived Probability")
        plt.show()
        print(f"--------------------------------------------------------------------------------\n\n")
```

    --------------------------------gender--------------------------------------
    gender
    female    518
    male      482
    Name: gender, dtype: int64
    
    
    cluster    1    2
    gender           
    female   198  320
    male     242  240
    
 

![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_1.png?raw=true)



![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_2.png?raw=true)


    --------------------------------------------------------------------------------
    
    
    --------------------------------race/ethnicity--------------------------------------
    race/ethnicity
    group A     89
    group B    190
    group C    319
    group D    262
    group E    140
    Name: race/ethnicity, dtype: int64
    
    
    cluster           1    2
    race/ethnicity          
    group A          54   35
    group B         102   88
    group C         139  180
    group D         103  159
    group E          42   98
    


![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_4.png?raw=true)



![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_5.png?raw=true)


    --------------------------------------------------------------------------------
    
    
    --------------------------------parental level of education--------------------------------------
    parental level of education
    associate's degree    222
    bachelor's degree     118
    high school           196
    master's degree        59
    some college          226
    some high school      179
    Name: parental level of education, dtype: int64
    
    
    cluster                        1    2
    parental level of education          
    associate's degree            88  134
    bachelor's degree             37   81
    high school                  109   87
    master's degree               20   39
    some college                  97  129
    some high school              89   90
    


![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_7.png?raw=true)



![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_8.png?raw=true)


    --------------------------------------------------------------------------------
    
    
    --------------------------------lunch--------------------------------------
    lunch
    free/reduced    355
    standard        645
    Name: lunch, dtype: int64
    
    
    cluster         1    2
    lunch                 
    free/reduced  212  143
    standard      228  417
    


![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_10.png?raw=true)



![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_11.png?raw=true)


    --------------------------------------------------------------------------------
    
    
    --------------------------------test preparation course--------------------------------------
    test preparation course
    completed    358
    none         642
    Name: test preparation course, dtype: int64
    
    
    cluster                    1    2
    test preparation course          
    completed                110  248
    none                     330  312
    


![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_13.png?raw=true)



![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_10_14.png?raw=true)


    --------------------------------------------------------------------------------
    
    
    

<ul>
<li>    
<p>    
     <b>Gender</b> <br>
        Performance Sequence Best to Least <br>
        female > male <br>
        -> Students of "female"  Gender  is more likely to score best compared to male.
</p>
</li>
<li>    
<p>
     <b>race/ethnicity</b> <br>
        Performance Sequence Best to Least <br>
        Group E > Group D > Group C  > Group B > Group A <br>
        -> Students of "Group E"  race/ethnicity  is more likely to score best compared to all the other. <br>
        -> Students of "Group A"  race/ethnicity  is more likely score least compared to all the other. 
</p>
</li>
<li>    
<p>
     <b>parental level of education</b> <br>
        Performance Sequence Best to Least <br>
        bachelor's degree > master's degree > associate's degree > some college > some high school > high school     <br>
        -> Parent's Education Level who have "bachelor's degree" is more likely to score best compared to all the other education level. <br>
        -> Parent's Education Level who have "high school" is more likely score least compared to all the other education level.
</p>
</li>
<li>    
<p>
     <b>lunch</b> <br>
        Performance Sequence Best to Least <br>
        standard > free/reduced <br>
</p>
</li>
<li>    
<p>
     <b>test preparation course</b> <br>
        Performance Sequence Best to Least <br>
        completed > none <br>
        -> The Students who have Completed the have more likely to score better  
</p>
</li>
</ul>    


```python
from sklearn.preprocessing import KBinsDiscretizer
from numpy import mean
features = [feature for feature in train_data.columns if feature != "cluster" and train_data[feature].dtypes != 'O' ]
discrete=KBinsDiscretizer(n_bins=10,encode='ordinal', strategy='quantile')
num_binned=pd.DataFrame(discrete.fit_transform(train_data[features]),index=train_data[features].index, columns=train_data[features].columns)
Y=train_data[['cluster']]
X_bin_combined=pd.concat([Y,num_binned],axis=1,join='inner')

for feature in features:
    print(f"--------------------------------{feature}--------------------------------------------")
    data=train_data.copy()
    data[feature].hist(bins=10)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()
    sns.barplot(x=feature, y="cluster",data=X_bin_combined, estimator=mean )
    plt.show()
    print(f"-------------------------------{feature} END--------------------------------------------\n\n")
```

    --------------------------------math score--------------------------------------------
    


![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_12_1.png?raw=true)



![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_12_2.png?raw=true)


    -------------------------------math score END--------------------------------------------
    
    
    --------------------------------reading score--------------------------------------------
    


![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_12_4.png?raw=true)



![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_12_5.png?raw=true)


    -------------------------------reading score END--------------------------------------------
    
    
    --------------------------------writing score--------------------------------------------
    


![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_12_7.png?raw=true)



![png](https://github.com/melwinmpk/Students_Performance_in_Exams_ML/blob/main/img/output_12_8.png?raw=true)


    -------------------------------writing score END--------------------------------------------
    
    
    


```python

```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)
```


```python
train_data = pd.read_csv("../dataset/train_model_processed_data.csv")
print(train_data.shape)
```

    (1000, 9)
    


```python
%config InlineBackend.print_figure_kwargs = {'bbox_inches':None}
%matplotlib notebook
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sns.set(style = "darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x = train_data['math score']
y = train_data['reading score']
z = train_data['writing score']

ax.set_xlabel("math score")
ax.set_ylabel("reading score")
ax.set_zlabel("writing score")

ax.scatter(x, y, z)

plt.show()
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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" width="432">



```python
train_data.describe()
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
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>66.08900</td>
      <td>69.169000</td>
      <td>68.054000</td>
      <td>1.560000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.16308</td>
      <td>14.600192</td>
      <td>15.195657</td>
      <td>0.496635</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
      <td>17.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>57.00000</td>
      <td>59.000000</td>
      <td>57.750000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>66.00000</td>
      <td>70.000000</td>
      <td>69.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77.00000</td>
      <td>79.000000</td>
      <td>79.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
numerical_features = [feature for feature in train_data.columns if train_data[feature].dtypes != 'O']
non_numerical_feature = [feature for feature in train_data.columns if train_data[feature].dtypes == 'O']
```


```python
'''2) In the Data 
        population of the Cluster 2 is more than the Cluster 1
''' 
```


```python
plt.rcParams["figure.figsize"]=20,20
```


```python
'''
    2) Gender 
        Performance Sequence Best to Least
        female > male
        -> Students of "female"  Gender  is more likely to score best compared to male.
        
    3) race/ethnicity
        Performance Sequence Best to Least
        Group E > Group D > Group C  > Group B > Group A
        -> Students of "Group E"  race/ethnicity  is more likely to score best compared to all the other.
        -> Students of "Group A"  race/ethnicity  is more likely score least compared to all the other. 
    
    4) parental level of education
        Performance Sequence Best to Least
        bachelor's degree > master's degree > associate's degree > some college > some high school > high school    
        -> Parent's Education Level who have "bachelor's degree" is more likely to score best compared to all the other education level.
        -> Parent's Education Level who have "high school" is more likely score least compared to all the other education level.
    
    5) lunch
        Performance Sequence Best to Least
        standard > free/reduced
    
    6) ** test preparation course
        Performance Sequence Best to Least
        completed > none
        -> The Students who have Completed the have more likely to score better 
        
        
'''
```


```python

```
