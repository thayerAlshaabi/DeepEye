
# DashCam Statistical Analytics


### Compare the two dataframes and identify each feature where the `GROUND TRUTH` and the `PREDICTION` by DeepEye were different, then calculate a score for each frame based on the number of false predictions


```python
scores = comparison_table.groupby(['FRAME_ID']).size().reset_index(name='SCORE')
scores =  len(tester_log.columns) - scores['SCORE'] - 1

results['ACCURACY', 'SCORE'] = scores
results['ACCURACY', 'SCORE'].fillna(len(tester_log.columns) - 1, inplace=True) 
results['ACCURACY', 'SCORE'] = results['ACCURACY', 'SCORE'] / 7
results.style.format("{:.2%}", subset=pd.IndexSlice[:, pd.IndexSlice[:, 'SCORE']])

results
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">PEDESTRIAN</th>
      <th colspan="2" halign="left">VEHICLES</th>
      <th colspan="2" halign="left">BIKES</th>
      <th colspan="2" halign="left">STOP_SIGN</th>
      <th colspan="2" halign="left">TRAFFIC_LIGHT</th>
      <th colspan="2" halign="left">OFF_LANE</th>
      <th colspan="2" halign="left">COLLISION</th>
      <th>ACCURACY</th>
    </tr>
    <tr>
      <th></th>
      <th>GROUND TRUTH</th>
      <th>PREDICTION</th>
      <th>GROUND TRUTH</th>
      <th>PREDICTION</th>
      <th>GROUND TRUTH</th>
      <th>PREDICTION</th>
      <th>GROUND TRUTH</th>
      <th>PREDICTION</th>
      <th>GROUND TRUTH</th>
      <th>PREDICTION</th>
      <th>GROUND TRUTH</th>
      <th>PREDICTION</th>
      <th>GROUND TRUTH</th>
      <th>PREDICTION</th>
      <th>SCORE</th>
    </tr>
    <tr>
      <th>FRAME_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>274 rows × 15 columns</p>
</div>



# Basic Statistical Analytics

## Overall Performance

GTA-V Performance Results         |  Dash-Cam Performance Results 
:-------------------------:|:-------------------------:
![User Interface](gtav_score.png)  |  ![Dash-Cam](dashcam_score.png)


## Dash-Cam Analysis

### Predictions results for Pedstrians

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>(PEDESTRIAN, PREDICTION)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>230.0</td>
      <td>0.937267</td>
      <td>0.097490</td>
      <td>0.428571</td>
      <td>0.857143</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.0</td>
      <td>0.801948</td>
      <td>0.131495</td>
      <td>0.428571</td>
      <td>0.714286</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

### Predictions results for Vehicles


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>(VEHICLES, PREDICTION)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>0.821429</td>
      <td>0.214286</td>
      <td>0.571429</td>
      <td>0.678571</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>270.0</td>
      <td>0.916931</td>
      <td>0.112798</td>
      <td>0.428571</td>
      <td>0.857143</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

### Predictions results for Bikes



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>(BIKES, PREDICTION)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>258.0</td>
      <td>0.923034</td>
      <td>0.110625</td>
      <td>0.428571</td>
      <td>0.857143</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.0</td>
      <td>0.794643</td>
      <td>0.116277</td>
      <td>0.571429</td>
      <td>0.714286</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


### Predictions results for Stop Signs


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>(STOP_SIGN, PREDICTION)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>274.0</td>
      <td>0.915537</td>
      <td>0.114775</td>
      <td>0.428571</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


### Predictions results for Traffic Lights


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>(TRAFFIC_LIGHT, PREDICTION)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>225.0</td>
      <td>0.935238</td>
      <td>0.093405</td>
      <td>0.571429</td>
      <td>0.857143</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.0</td>
      <td>0.825073</td>
      <td>0.155033</td>
      <td>0.428571</td>
      <td>0.714286</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

### Predictions results for Off-Lane Warnings

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>(OFF_LANE, PREDICTION)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>262.0</td>
      <td>0.918212</td>
      <td>0.113283</td>
      <td>0.428571</td>
      <td>0.857143</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.0</td>
      <td>0.857143</td>
      <td>0.136209</td>
      <td>0.571429</td>
      <td>0.821429</td>
      <td>0.857143</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


### Predictions results for Collision Warnings

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>(COLLISION, PREDICTION)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>266.0</td>
      <td>0.920516</td>
      <td>0.109486</td>
      <td>0.428571</td>
      <td>0.857143</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>0.750000</td>
      <td>0.166424</td>
      <td>0.428571</td>
      <td>0.678571</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.857143</td>
    </tr>
  </tbody>
</table>
</div>
