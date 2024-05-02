from flask import Flask,request,jsonify
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return "hello world"

@app.route('/predict',methods=['POST'])
def predict():
    age=int(request.form.get('age'))
    sex=int(request.form.get('sex'))
    cp=int(request.form.get('cp'))
    trestbps=int(request.form.get('trestbps'))
    chol=int(request.form.get('chol'))
    fbs=int(request.form.get('fbs'))
    restecg=int(request.form.get('restecg'))
    thalach=int(request.form.get('thalach'))
    exang=int(request.form.get('exang'))
    oldpeak=float(request.form.get('oldpeak'))
    slope=int(request.form.get('slope'))
    ca=int(request.form.get('ca'))
    thal=int(request.form.get('thal'))
    
    input_query=np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    result=model.predict(input_query)[0]
    result = int(result)
    
    # result={'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal}

    return jsonify({'heart disease':result})



if __name__ == '__main__':
    app.run(debug=True)