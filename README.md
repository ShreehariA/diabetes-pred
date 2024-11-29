# diabetes-pred

## Project Description

This project is a Django web application for diabetes prediction using machine learning techniques. The application allows users to input their health metrics and receive a prediction on whether they have diabetes. The project uses various machine learning algorithms to provide accurate predictions and includes features such as generating PDF reports and sending SMS notifications.

## Features

- Predict diabetes based on user input metrics
- Use multiple machine learning algorithms for prediction
- Generate PDF reports of the prediction results
- Send SMS notifications with the prediction results
- Web interface for easy user interaction

## Project Setup

### Installation Instructions

1. Clone the repository:
   ```
   git clone https://github.com/ShreehariA/diabetes-pred.git
   ```
2. Navigate to the project directory:
   ```
   cd diabetes-pred
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```
5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Apply the database migrations:
   ```
   python manage.py migrate
   ```
7. Start the development server:
   ```
   python manage.py runserver
   ```

## Usage Instructions

### Using the Web Interface

1. Open your web browser and navigate to `http://127.0.0.1:8000/`.
2. You will see the home page with options for Admin and User.
3. Click on "User" to access the user interface.
4. Fill in the required health metrics such as Pregnancy, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.
5. Click on the "Result" button to get the diabetes prediction.

### Generating PDF Reports

1. After receiving the prediction result, you will see an option to generate a PDF report.
2. Click on the "Generate Report PDF" button to download the PDF report of the prediction results.

### Sending SMS Notifications

1. After receiving the prediction result, you will see an option to send an SMS notification.
2. Click on the "Send SMS" button to send the prediction results to the provided mobile number.

## Machine Learning Algorithms

The project uses the following machine learning algorithms for diabetes prediction:

- Logistic Regression
- Adaboost
- Random Forest
- Decision Tree
- Gaussian Naive Bayes
- k-Nearest Neighbour

### Accuracy Scores

The accuracy scores of the machine learning algorithms used in the project are as follows:

- Logistic Regression: 0.78
- Adaboost: 0.80
- Random Forest: 0.82
- Decision Tree: 0.75
- Gaussian Naive Bayes: 0.76
- k-Nearest Neighbour: 0.77
