#The app takes real-time data from usgs website and apply's the machine learning technique(XGBoost) through each loop it makes and can predict to a range of five days
from flask import Flask

app = Flask(__name__)

from flask import render_template, flash, request
import logging, io, base64, os, datetime
from datetime import datetime
from datetime import timedelta
import numpy as np
import xgboost as xgb
import pandas as pd

# Variables(Global)
earthquake_live = None
days_out_to_predict = 5




def prepare_earthquake_data_and_model(days_out_to_predict = 5, max_depth=3, eta=0.1):

#Desccription : From extraction to model preparation. This function takes in how many days to predict or rolling window
# period, max_depth for XGboost and learning rate. I extract data directly from https://earthquake.usgs.gov/
# instead of loading from existing database since we want real time data that is updated every minute.
    
#Arguments : int (days_to_predict rolling window), int (maximum depth hyperparameter for xgboost), float (learning rate of alogrithm)

#Return : Pandas Dataframe (Prediction dataframe with live/ future NaN values in outcome magnitutde of quake that has to be predicted)
    
    # get latest data from USGS servers

    data = pd.read_csv('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv')
    data = data.sort_values('time', ascending=True)
    # truncate time from datetime
    data['date'] = data['time'].str[0:10]

    # For better performance of the app and model we will keep  only columns needed

    data = data[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]
    temp_data = data['place'].str.split(', ', expand=True) 
    data['place'] = temp_data[1]
    data = data[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]

    
    #Get the mean latitude and longitude 
	
    data_coords = data[['place', 'latitude', 'longitude']]
    data_coords = data_coords.groupby(['place'], as_index=False).mean()
    data_coords = data_coords[['place', 'latitude', 'longitude']]
	
    data = data[['date', 'depth', 'mag', 'place']]
    data = pd.merge(left=data, right=data_coords, how='inner', on=['place'])

    # Apply the machine learning algorithm(XGBoost) to each loop
    
    eq_data = []
    data_live = []
    for symbol in list(set(data['place'])):
        temp_data = data[data['place'] == symbol].copy()
        temp_data['depth_avg_22'] = temp_data['depth'].rolling(window=22,center=False).mean() 
        temp_data['depth_avg_15'] = temp_data['depth'].rolling(window=15,center=False).mean()
        temp_data['depth_avg_7'] = temp_data['depth'].rolling(window=7,center=False).mean()
        temp_data['mag_avg_22'] = temp_data['mag'].rolling(window=22,center=False).mean() 
        temp_data['mag_avg_15'] = temp_data['mag'].rolling(window=15,center=False).mean()
        temp_data['mag_avg_7'] = temp_data['mag'].rolling(window=7,center=False).mean()
        temp_data.loc[:, 'mag_outcome'] = temp_data.loc[:, 'mag_avg_7'].shift(days_out_to_predict * -1)

        data_live.append(temp_data.tail(days_out_to_predict))

        eq_data.append(temp_data)

    # Add together all location-based dataframes into master dataframe
    
    data = pd.concat(eq_data)

    # Remove all null fields
    
    data = data[np.isfinite(data['depth_avg_22'])]
    data = data[np.isfinite(data['mag_avg_22'])]
    data = data[np.isfinite(data['mag_outcome'])]

    # prepare outcome variable
    
    data['mag_outcome'] = np.where(data['mag_outcome'] > 2.5, 1,0)

    data = data[['date',
             'latitude',
             'longitude',
             'depth_avg_22',
             'depth_avg_15',
             'depth_avg_7',
             'mag_avg_22', 
             'mag_avg_15',
             'mag_avg_7',
             'mag_outcome']]

    # keep only data where we can make predictions
    
    data_live = pd.concat(data_live)
    data_live = data_live[np.isfinite(data_live['mag_avg_22'])]

    # let's train the model whenever the webserver is restarted
    
    from sklearn.model_selection import train_test_split
    features = [f for f in list(data) if f not in ['date', 'mag_outcome', 'latitude',
     'longitude']]

    X_train, X_test, y_train, y_test = train_test_split(data[features],
                         data['mag_outcome'], test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train[features], label=y_train)
    dtest = xgb.DMatrix(X_test[features], label=y_test)

    param = {
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'eval_metric': 'auc',
            'max_depth': max_depth,  # the maximum depth of each tree
            'eta': eta,  # the training step for each iteration
            }  # logging mode - quiet}  # the number of classes that exist in this datset

    # Determine the number of training iterations   
    
    num_round = 1000   
    early_stopping_rounds=30
    xgb_model = xgb.train(param, dtrain, num_round) 


    # train on live data
    
    dlive = xgb.DMatrix(data_live[features])  
    preds = xgb_model.predict(dlive)

    # add predictions to live data
    
    data_live = data_live[['date', 'place', 'latitude', 'longitude']]
    
    # add predictions back to dataset 
    
    data_live = data_live.assign(preds=pd.Series(preds).values)

    # aggregate down dups
    
    data_live = data_live.groupby(['date', 'place'], as_index=False).mean()

    # increment date to include DAYS_OUT_TO_PREDICT
    
    data_live['date']= pd.to_datetime(data_live['date'],format='%Y-%m-%d') 
    data_live['date'] = data_live['date'] + pd.to_timedelta(days_out_to_predict,unit='d')

    return(data_live)
    
def get_earth_quake_estimates(desired_date, data_live):
	from datetime import datetime
	live_set_tmp = data_live[data_live['date'] == desired_date]
    # Convert to strings format lat/lon for Google Maps 
	LatLngString = ''
	if (len(live_set_tmp) > 0):
		for lat, lon, pred in zip(live_set_tmp['latitude'], live_set_tmp['longitude'], live_set_tmp['preds']):
			if (pred > 0.3):
				LatLngString += "new google.maps.LatLng(" + str(lat) + "," + str(lon) + "),"
		return(LatLngString)		 
            


@app.before_first_request
def startup():
    global earthquake_live

    # prepare data, model and get live data set with earthquake forecasts
    
    earthquake_live = prepare_earthquake_data_and_model()


@app.route("/", methods=['POST', 'GET'])
def build_page():
        if request.method == 'POST':

            horizon_int = int(request.form.get('slider_date_horizon'))
            horizon_date = datetime.today() + timedelta(days=horizon_int)

            return render_template('index.html',
                date_horizon = horizon_date.strftime('%d/%m/%Y'),
                earthquake_horizon = get_earth_quake_estimates(str(horizon_date)[:10], earthquake_live),
                current_value=horizon_int, 
                days_out_to_predict=days_out_to_predict)

        else:
            # set blank map
            
            return render_template('index.html',
                date_horizon = datetime.today().strftime('%d/%m/%Y'),
                earthquake_horizon = '',
                current_value=0,
                days_out_to_predict=days_out_to_predict)
#set debug mode true and run on port 88, for production purposes set host='0.0.0.0'
app.run(debug=True, port=88)
