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




![png](output_7_1.png)


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
    


![png](output_10_1.png)



![png](output_10_2.png)


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
    


![png](output_10_4.png)



![png](output_10_5.png)


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
    


![png](output_10_7.png)



![png](output_10_8.png)


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
    


![png](output_10_10.png)



![png](output_10_11.png)


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
    


![png](output_10_13.png)



![png](output_10_14.png)


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
    


![png](output_12_1.png)



![png](output_12_2.png)


    -------------------------------math score END--------------------------------------------
    
    
    --------------------------------reading score--------------------------------------------
    


![png](output_12_4.png)



![png](output_12_5.png)


    -------------------------------reading score END--------------------------------------------
    
    
    --------------------------------writing score--------------------------------------------
    


![png](output_12_7.png)



![png](output_12_8.png)


    -------------------------------writing score END--------------------------------------------
    
    
    


```python

```
