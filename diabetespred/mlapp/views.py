from django.shortcuts import render,redirect
import pandas as pd  # Importing Pandas to process Dataset as a dataframe
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from django.http import HttpResponse
from django.templatetags.static import static
from mlapp.main import train  # train() which trains the data model in main.py
from mlapp.main1 import startup  # startup() which predicts the outcome
from datetime import datetime

# Importing reportlab requirements for PDF Generation

from django.http import FileResponse
import io
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

import warnings
warnings.filterwarnings("ignore")


def home(request):
    return render(request,'home.html')
def view(request):
    return render(request,'index.html')

# Predict function - predict the diabetes

def prediction(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        gender = request.POST.get('gender')
        number = int(request.POST.get('number'))
        email = request.POST.get('email')
        pregnancy = request.POST.get('Pregnancy')
        glucose = request.POST.get('Glucose')
        bp = request.POST.get('Bp')
        skin = request.POST.get('Skin')
        insulin = request.POST.get('Insulin')
        bmi = float(request.POST.get('Bmi'))
        dpf = float(request.POST.get('Dpf'))
        age = request.POST.get('Age')
        request.session['name'] = name
        request.session['gender'] = gender
        request.session['number'] = number
        request.session['email'] = email
        request.session['pregnancy'] = pregnancy
        request.session['glucose'] = glucose
        request.session['bp'] = bp
        request.session['skin'] = skin
        request.session['insulin'] = insulin
        request.session['bmi'] = bmi
        request.session['dpf'] = dpf
        request.session['age'] = age

        result = int(startup(pregnancy, glucose, bp, skin, insulin, bmi, dpf, age))
        if result == 1:
            output = "Sorry! You have diabetes😐"
        if result == 0:
            output = "Hurray! You do not have diabetes😀"
        request.session['output'] = output
        message = ''
        message = "Diabetes Final Report"+"\n"+"Name: "+name+"\n"+"Age: "+age+" Years"+"\n"+"Gender: "+gender+"\n"+"Date and Time: "+str(datetime.now())+"\n"+"-----------------------------"+"\n"+"Metrics of "+name+"\n"+"-----------------------------"+"\n"+"Your Pregnancy: "+str(pregnancy)+"\n"+"Your Glucose level: "+str(glucose)+"\n"+"Your Blood Pressure: "+str(bp)+"\n"+"Your Skin Thickeness: "+str(skin)+"\n"+"Your Insulin level: "+str(insulin)+"\n"+"Your BMI: "+str(bmi)+"\n"+"Your Diabetes Pedigree Function: "+str(dpf)+"\n"+"Your Age: "+str(age)+"\n"+"------------------------------"+"\n"+"FINAL RESULT: "+output
        return render(request,'output.html',{"outcome":output,"email":email,"message":message})
    else:
        return HttpResponse("Sorry! no response")

# PDF Report Generation

def venuepdf(request):
    name = request.session['name']
    gender = request.session['gender']
    number = request.session['number']
    pregnancy = request.session['pregnancy']
    glucose = request.session['glucose']
    bp = request.session['bp']
    skin = request.session['skin']
    insulin = request.session['insulin']
    bmi = request.session['bmi']
    dpf = request.session['dpf']
    age = request.session['age']
    output = request.session['output']

    def pdfgen(pregnancy, glucose, bp, skin, insulin, bmi, dpf, age):
        buf = io.BytesIO()
        c = canvas.Canvas(buf, bottomup=0)
        textob = c.beginText()
        textob.setTextOrigin(inch, inch)
        textob.setFont('Helvetica',14)
        lines = [
        "Diabetes Final Report",
        "--------------------------",   # PDF Report Contents list
        "Name: "+name,
        "Age: "+age+" Years",
        "Gender: "+gender,
        "Date and Time: "+str(datetime.now()),
        "-----------------------------",
        "Metrics of "+name,
        "-----------------------------",
        "Your Pregnancy: "+str(pregnancy),
        "Your Glucose level: "+str(glucose),
        "Your Blood Pressure: "+str(bp),
        "Your Skin Thickeness: "+str(skin),
        "Your Insulin level: "+str(insulin),
        "Your BMI: "+str(bmi),
        "Your Diabetes Pedigree Function: "+str(dpf),
        "Your Age: "+str(age),
        "------------------------------",
        "FINAL RESULT: "+output,
        ]
        for line in lines:
            textob.textLine(line)

        c.drawText(textob)
        c.showPage()
        c.save()
        buf.seek(0)
        return FileResponse(buf, as_attachment=True, filename=''+name[:]+'_DiabetesTestReport.pdf')
    x = pdfgen(pregnancy, glucose, bp, skin, insulin, bmi, dpf, age)
    return x

# Validating the Login credentials

def login(request):
    auth = {'name':'jvs','password':'jvs'} # Authenticated User
    if request.method == 'POST':
        name = str(request.POST.get('username'))
        pwd = str(request.POST.get('password'))
        print(auth["name"])
        if name == auth["name"] and pwd == auth["password"]:
            return redirect('/adminpage/')
        if not name == auth["name"] and pwd == auth["password"]:
            return HttpResponse('<center>Invalid Username. Try Again:)<br><br><a href="/login/">Back</a></center>')
        if name == auth["name"] and not pwd == auth["password"]:
            return HttpResponse('<center>Invalid Password. Try Again:)<br><br><a href="/login/">Back</a></center>')
        else:
            return HttpResponse('<center>Invalid Credentials. Try Again:)<br><br><a href="/login/">Back</a></center>')
    return render(request,'login.html')

def adminPage(request):
    return render(request,'adminpage.html')


# Displays accuracy and metrics of algorithms

def training(request):
    list, metrics, accuracy, algos = train()
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.bar(algos, accuracy)
    fig.savefig('mlapp/static/accuracyplot.jpg')
    return render(request,'train.html',{'list':list,'metrics':metrics})

def developer(request):
    return redirect('https://jaividyasagar.pythonanywhere.com/')

def graph(request):
    return render(request,'graph.html')

# Importing twilio rest API for SMS Generation

from twilio.rest import Client
def sms(request):
    name = request.session['name']
    gender = request.session['gender']
    number = request.session['number']
    pregnancy = request.session['pregnancy']
    glucose = request.session['glucose']
    bp = request.session['bp']
    skin = request.session['skin']
    insulin = request.session['insulin']
    bmi = request.session['bmi']
    dpf = request.session['dpf']
    age = request.session['age']
    output = request.session['output']
    account_sid = 'AC67751ea2cf9b14d40736147dfbce1a04'
    auth_token = '18bb438017b9053a6546d15896047b26'


    smsmessage = 'DIABETES PREDICTION REPORT'+'\n'+'Name: '+name+'\n'+'Age: '+str(age)+'\n'+'Gender: '+gender+'\n'+'Date and Time: '+str(datetime.now())+'\n'+'-------------------------'+'\n'+'Pregnancy: '+str(pregnancy)+'\n'+'Glucose: '+str(glucose)+'\n'+'Blood Pressure: '+str(bp)+'\n'+'Skin Thickness: '+str(skin)+'\n'+'Insulin: '+str(insulin)+'\n'+'Body Mass Index: '+str(bmi)+'\n'+'Diabetes Pedigree Function: '+str(dpf)+'\n'+'---------------------------'+'\n'+'FINAL RESULT: '+output
    print(smsmessage)
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        from_ = '+12054486473',
        body = smsmessage,
        to = '+919500442237'
    )
    print(message.sid)
    return render(request,'sms.html')
