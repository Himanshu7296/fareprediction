import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import datetime

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def h_distance(lat1, long1, lat2, long2):
    R = 6371  #radius of earth in kilometers
        #R = 3959 #radius of earth in miles
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    
    delta_phi = np.radians(lat2-lat1)
    delta_lambda = np.radians(long2-long1)
    
    #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
    #c = 2 * atan2( √a, √(1−a) )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    #d = R*c
    d = (R * c) #in kilometers
 
    return d

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        pickup_longitude = float(request.form['plongitude'])
        pickup_latitude = float(request.form['platitude'])
        dropoff_longitude = float(request.form['dlongitude'])
        dropoff_latitude = float(request.form['dlatitude'])
        passenger_count = int(request.form['passengers'])
        date = (request.form['pdate'])
        time = (request.form['ptime'])
        dt = list(map(int, date.split('-')))
        dayofweek = datetime.date(dt[0],dt[1],dt[2]).weekday()
        t = time.split(':')
        h_dist = h_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude) 
        Year = dt[0]
        Month = dt[1]
        Date = dt[2]
        Hour = t[0]


    final_features = np.array([[pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count, h_dist, Year, Month, Date, dayofweek, Hour]])
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('homepage.html', prediction_text='Estimated Fare is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)