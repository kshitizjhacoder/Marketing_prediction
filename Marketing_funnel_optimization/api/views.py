# views.py
from django.shortcuts import render
from .forms import PredictionForm
import pandas as pd
import joblib


def predict(request):
    percent = None  # Initialize prediction result

    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            print(data)
            # Load the trained model
            classifier = joblib.load(
                r"C:\Users\kshit\OneDrive\Documents\desktop\Project\Marketing_funnel_optimization\api\Algorithm\c2_Classifier_LoyalCustomers"
            )
            # Prepare input features for prediction
            X_input = pd.DataFrame(
                {
                    "DemAffl": [data["DemAffl"]],
                    "DemAge": [data["DemAge"]],
                    "DemClusterGroup": [int(data["DemClusterGroup"])],
                    "DemGender": [int(data["DemGender"])],
                    "DemReg": [int(data["DemReg"])],
                    "DemTVReg": [int(data["DemTVReg"])],
                    "LoyalClass": [int(data["LoyalClass"])],
                    "LoyalTimeSpend": [data["LoyalTimeSpend"]],
                    "LoyalTime": [data["LoyalTime"]],
                }
            )

            # Make prediction using the model
            prediction = classifier.predict_proba(X_input)[0]

            # Assuming the prediction is a probability, convert it to percentage
            percent = round(
                prediction[1] * 100, 2
            )  # Assuming prediction[1] is the positive class probability

    else:
        form = PredictionForm()

    return render(request, "index.html", {"form": form, "percent": percent})
