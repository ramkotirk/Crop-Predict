from django.http import Http404
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create your views here.

def get_referer(request):
    referer = request.META.get('HTTP_REFERER')
    if not referer:
        return None
    return referer

def index(request):
    return render(request, 'index.html')

def result(request):
    if not get_referer(request):
        raise Http404
        
    result=''
    if request.method == 'POST':
        nitrogen = request.POST['nitrogen']
        phosphorus = request.POST['phosphorus']
        potassium = request.POST['potassium']
        temperature = request.POST['temperature']
        humidity = request.POST['humidity']
        ph = request.POST['ph']
        rainfall = request.POST['rainfall']

        df = pd.read_csv("Mechine Learning\Crop_recommendation.csv")

        features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
        target = df['label']
        #features = df[['temperature', 'humidity', 'ph', 'rainfall']]
        labels = df['label']

        # Initialzing empty lists to append all model's name and corresponding name
        acc = []
        model = []

        # Splitting into train and test data
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

        RF = RandomForestClassifier(n_estimators=20, random_state=0)
        RF.fit(Xtrain.values,Ytrain.values)

        predicted_values = RF.predict(Xtest.values)

        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('RF')

        data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = RF.predict(data)
        result = str(prediction).replace("'","")[1:-1]

    return render(request, 'result.html',{'result':result})