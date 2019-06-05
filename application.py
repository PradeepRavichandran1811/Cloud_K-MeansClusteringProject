#Author:PRADEEP RAVICHANDRAN
#CSE 6331-Cloud Computing

#References
#google charts
#https://developers.google.com/chart/interactive/docs/gallery/scatterchart
#https://developers.google.com/chart/interactive/docs/gallery/barchart
#https://developers.google.com/chart/interactive/docs/gallery/piechart
#for non-numerical data function
#https://pythonprogramming.net/k-means-titanic-dataset-machine-learning-tutorial/

import os
from flask import Flask, render_template, request
import sqlite3 as sql
from math import sin, cos, sqrt, atan2, radians
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
#from pandas import read_csv
application = Flask(__name__)
csvdata = pd.read_csv('minnow.csv')
csvdata.fillna(0, inplace=True)

import sqlite3, csv, base64
con=sqlite3.connect('titanic.db')
#creating cursor to perform database operations
cursor = con.cursor()
#cursor.execute("CREATE TABLE titanic (CabinNum, Fname, Lname, Age, Survived,Lat,Long,PictureCap,PicturePas,Fare,Decklevel);")
con.commit()

#@application.route are decorators in Flask
@application.route('/')
def index():
    return render_template('home.html')

@application.route('/kmeanscluster')
def kmeanscluster():
    return render_template('nclusters.html')

@application.route('/cabpic')
def cabpic():
    return render_template('cabpic.html')

@application.route('/ban')
def ban():
    return render_template('banner.html')

#Function to print Pie Chart
@application.route('/magrange')
def magrange():
    con = sql.connect("eq.db")
    con.row_factory = sql.Row
    start="2018-06-01T17:20:32.340Z"
    end="2018-06-07T20:25:07.390Z"
    startdate = start.split('T')[0]
    enddate = end.split('T')[0]
    print(startdate)
    print(enddate)
    cursor = con.cursor()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 2.0 AND 2.5) AND (time BETWEEN ? AND ?)",(startdate,enddate))
    mag = cursor.fetchone()
    print(mag[0])
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 2.5 AND 3.0) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag1 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 3.0 AND 3.5) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag2 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 3.5 AND 4.0) AND (time BETWEEN ? AND ?)", (startdate, enddate))
    mag3 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 4.0 AND 4.5) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag4 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 4.5 AND 5.0) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag5 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 5.0 AND 5.5) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag6 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 5.5 AND 6.0) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag7 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 6.0 AND 6.5) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag8 = cursor.fetchone()
    count = [mag[0],mag1[0],mag2[0],mag3[0],mag4[0],mag5[0],mag6[0],mag7[0],mag8[0]]
    print(count)

    return render_template('magrange.html',count=count)

#Function to print Horizontal Bar Chart
@application.route('/magpie')
def magpie():
    con = sql.connect("eq.db")
    con.row_factory = sql.Row
    start="2018-06-01T17:20:32.340Z"
    end="2018-06-07T20:25:07.390Z"
    startdate = start.split('T')[0]
    enddate = end.split('T')[0]
    print(startdate)
    print(enddate)
    cursor = con.cursor()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 2.0 AND 2.5) AND (time BETWEEN ? AND ?)",(startdate,enddate))
    mag = cursor.fetchone()
    print(mag[0])
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 2.5 AND 3.0) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag1 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 3.0 AND 3.5) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag2 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 3.5 AND 4.0) AND (time BETWEEN ? AND ?)", (startdate, enddate))
    mag3 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 4.0 AND 4.5) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag4 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 4.5 AND 5.0) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag5 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 5.0 AND 5.5) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag6 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 5.5 AND 6.0) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag7 = cursor.fetchone()
    cursor.execute("select Count(mag) from earthquake where (mag BETWEEN 6.0 AND 6.5) AND (time BETWEEN ? AND ?)",(startdate, enddate))
    mag8 = cursor.fetchone()
    count = [mag[0],mag1[0],mag2[0],mag3[0],mag4[0],mag5[0],mag6[0],mag7[0],mag8[0]]
    print(count)

    return render_template('magpie.html',count=count)

@application.route('/far')
def far():
    return render_template('farerange.html')


@application.route('/fsurvivor')
def fsurvivor():
    con = sql.connect("titanic.db")
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select Count(survived) from titanic1 where survived = '1' and sex='female'")
    rows = cursor.fetchone()
    cursor.execute("select Count(survived) from titanic1 where survived = '0' and sex='female'")
    row = cursor.fetchone()
    return render_template('fsurvivor.html',rows=rows,row=row)

#function to get male survivors
@application.route('/msurvivor')
def msurvivor():
    con = sql.connect("titanic.db")
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select Count(survived) from titanic1 where survived = '1' AND sex='male'")
    rows = cursor.fetchone()
    cursor.execute("select Count(survived) from titanic1 where survived = '0' AND sex='male'")
    row = cursor.fetchone()
    return render_template('msurvivor.html',rows=rows,row=row)

@application.route('/picupload',methods=['get','post'])
def picupload():
    lname = request.form['lname']
    con = sqlite3.connect('titanic.db')
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute('select PictureCap from titanic2 where Lname =?', (lname,))
    row = cursor.fetchall()
    cursor.execute('select PicturePas from titanic2 where Lname =?', (lname,))
    rows = cursor.fetchall()
    return render_template('picdisplay.html',rows=rows,row=row)

#to get the size of the current working directory
@application.route('/getsize')
def getsize():
    cwd = os.getcwd()
    total_size = os.path.getsize(cwd)
    return render_template('size.html')

#function to handle non-numerical data
def non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


nd = non_numerical_data(csvdata)


#K-Means Clustering
@application.route('/kmeansclustering', methods=['GET','POST'])
def kmeansclustering():
    attr = request.form['attr1']
    attr1 = request.form['attr2']
    f = nd[attr]
    a = nd[attr1]
    #do for three dimensional also
    X = list(zip(f, a))
    #print(X)
    # KMeans Clustering with cluster size 100
    n = int(request.form['nclusters'])
    #print(n)
    kmean = KMeans(n_clusters=n,random_state=10)
    kmean.fit(X)
    kmeans = kmean.predict(X)
    label = kmean.labels_
   # print(label)
    totpts = len(label)
    cluster_labels = kmean.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n,
          "The average silhouette_score is :", silhouette_avg)
    # to get cluster centers
    C = kmean.cluster_centers_
    #print(len(C))
    #print(C)
    dist = []
    #print(C[0][0])
    #print(C[1])
    for i in range(0, len(C)):
        for j in range(i+1, len(C)):
                distance = np.linalg.norm(C[i]-C[j])
                dist.append(distance)
    #print(dist)
    #print(len(dist))
    unique, count = np.unique(label, return_counts=True)
    points = dict(zip(unique, count))
    return render_template('clustering.html', C=C, dist=dist,n=n,points=points,totpts=totpts,silhouette_avg=silhouette_avg)



@application.route('/updatedeck',methods=['GET','POST'])
def updatedeck():
    att = request.form['attr1']
    att1 = request.form['attr2']
    f = nd[att]
    a = nd[att1]
    d = []
    for row in f:
        decklevel = int(row/100)
        d.append(decklevel)
    print(d)
    #do for three dimensional also
    X = list(zip(d, a))
    #print(X)
    # KMeans Clustering
    n = int(request.form['nclusters'])
    #print(n)
    kmean = KMeans(n_clusters=n,random_state=10)
    kmean.fit(X)
    kmeans = kmean.predict(X)
    label = kmean.labels_
   # print(label)
    totpts = len(label)
    cluster_labels = kmean.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n,
          "The average silhouette_score is :", silhouette_avg)
    # to get cluster centers
    C = kmean.cluster_centers_
    print(C)
    #print(len(C))
    dist = []
    #print(C[0])
    #print(C[1])
    for i in range(0, len(C)):
        for j in range(i+1, len(C)):
                distance = np.linalg.norm(C[i]-C[j])
                dist.append(distance)
    #print(dist)
    #print(len(dist))
    unique, count = np.unique(label, return_counts=True)
    points = dict(zip(unique, count))
    return render_template('clustering.html', C=C, dist=dist,n=n,points=points,totpts=totpts,silhouette_avg=silhouette_avg)

@application.route('/csvearthquake', methods=['GET','POST'])
def csvearthquake():
    if request.method == 'POST':
        try:
            #converting csv file into table with values
            if request.method == 'POST':
                 file = request.files['myfile']
                 con = sqlite3.connect('eq.db')
                 con.row_factory = sql.Row

            csv = pd.read_csv(file)
            csv.to_sql(name="earthquake", con=con, if_exists="replace", index=False)
            cursor=con.cursor()
            cursor.execute("select * from earthquake")
            row=cursor.fetchall()
            con.close()
        except:
                con.rollback()
        finally:
            return render_template("home.html")
            con.close()
            print(msg)


#to read CSV and to create a Table
@application.route('/csvtitanic', methods=['GET','POST'])
def csvtitanic():
    if request.method == 'POST':
        try:
            #converting csv file into table with values
            if request.method == 'POST':
                 file = request.files['myfile']
                 con = sqlite3.connect('titanic.db')
                 con.row_factory = sql.Row
            csv = pd.read_csv(file)
            csv.fillna(0,inplace=True)
            csv.to_sql(name="titanic1", con=con, if_exists="replace", index=False)
            cursor = con.cursor()
            cursor.execute("select * from titanic1")
            row = cursor.fetchall()
            con.close()
           
        except:
                con.rollback()
        finally:
            return render_template("home.html")
            con.close()
            print(msg)


#function to list the table
@application.route('/lists')
def lists():
    if request.method == 'POST':
        file = request.files['myfile']
  
    con = sqlite3.connect('titanic.db')
    con.row_factory = sql.Row
    cursor = con.cursor()

    cursor.execute("select * from titanic1")
    row = cursor.fetchall()
    return render_template("list.html", row=row)


@application.route('/listsearthquake')
def listsearthquake():
    if request.method == 'POST':
        file = request.files['myfile']

    con = sqlite3.connect('eq.db')
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select * from earthquake")
    row = cursor.fetchall()
    return render_template("list1.html", row=row)

@application.route('/liststitanic')
def liststitanic():
    if request.method == 'POST':
        file = request.files['myfile']

    con = sqlite3.connect('titanic1.db')
    con.row_factory = sql.Row
    cursor = con.cursor()
    cursor.execute("select * from titanic2")
    row = cursor.fetchall()
    return render_template("list1.html", row=row)



if __name__ == '__main__':
    application.run(debug=True)


