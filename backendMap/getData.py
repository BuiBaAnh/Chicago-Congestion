import flask
from flask import jsonify
from flask_cors import CORS, cross_origin
# from flask_ngrok import run_with_ngrok
import requests
import json
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from coordinate import haversine
import collections
# from pyngrok import ngrok
import random

app = flask.Flask(__name__)
# app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# run_with_ngrok(app) 
def updateData():
    response = requests.get('https://data.cityofchicago.org/resource/n4j6-wkkf.json').text

    data = json.loads(response)
    df = pd.json_normalize(data)
    df.drop(['_fromst', '_tost', '_length', '_strheading', '_comments'], axis=1, inplace=True)
    # df = df[df['_traffic'].astype(int) != -1]
    df = df[df['_traffic'].astype(int) < 30]

    latLngData = df.filter(['_lit_lat', '_lit_lon'], axis=1).astype(float)
    pointFeat = df.filter(['_traffic', 'street', '_direction', '_last_updt'], axis=1)

    distance_matrix = squareform(pdist(latLngData, (lambda u,v: haversine(u,v))))
    # neigh = NearestNeighbors(n_neighbors=5)
    # nbrs = neigh.fit(distance_matrix)
    # distances, indices = nbrs.kneighbors(distance_matrix)

    # distances = np.sort(distances, axis=0)
    # distances = distances[:,1]
    # plt.plot(distances)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()


    # db = DBSCAN(eps=0.02, min_samples=5, metric='haversine')
    # y_db = db.fit_predict(latLngData) 
    db = DBSCAN(eps=0.6, min_samples=5, metric='precomputed') 
    y_db = db.fit_predict(distance_matrix)
    print(collections.Counter(y_db))


    latLngData['cluster'] = y_db
    # print(latLngData)

    plt.scatter(latLngData['_lit_lat'], latLngData['_lit_lon'], c=latLngData['cluster'])
    # plt.show()


    dataReturn = {
        "type": "FeatureCollection",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": [],
    }

    for i in range(0, len(latLngData.values)):
        featureData = {}
        featureData['type'] = "Feature"
        featureData['properties'] = {}
        featureData['properties']['id'] = df.iloc[i].segmentid
        featureData['properties']['clusterid'] = latLngData.values[i][2]
        featureData['geometry'] = {}
        featureData['geometry']['type'] = "Point"
        featureData['geometry']['coordinates'] = []
        featureData['geometry']['coordinates'].append(latLngData.values[i][1])
        featureData['geometry']['coordinates'].append(latLngData.values[i][0])
        featureData['properties']['_traffic'] = pointFeat.values[i][0]
        featureData['properties']['street'] = pointFeat.values[i][1]
        featureData['properties']['_direction'] = pointFeat.values[i][2]
        featureData['properties']['_last_updt'] = pointFeat.values[i][3]
        dataReturn['features'].append(featureData)
    return dataReturn

# latLngData['json'] = latLngData.apply(lambda x: x.to_json(), axis=1)
# returnData = latLngData['json'].values
# print(dataReturn)

@app.route('/')
@cross_origin()
def index():
    return "<h1>Welcome to our server !!</h1>" 


@app.route('/window', methods=['GET'])
@cross_origin()
def getData():
    dataReturn = updateData()
    return jsonify(dataReturn) 

if __name__ == "__main__":
    app.run()