# forms.py
from django import forms

CLUSTER_GROUP_CHOICES = [
    (0, "A"),
    (1, "B"),
    (2, "C"),
    (3, "D"),
    (4, "E"),
    (5, "F"),
    (6, "U"),
]

GENDER_CHOICES = [
    (2, "U"),
    (1, "M"),
    (0, "F"),
]

REGION_CHOICES = [
    (0, "Midlands"),
    (1, "North"),
    (2, "Scottish"),
    (3, "South East"),
    (4, "South West"),
]

TV_REGION_CHOICES = [
    (0, "Border"),
    (1, "C Scotland"),
    (2, "East"),
    (3, "London"),
    (4, "Midlands"),
    (5, "N East"),
    (6, "N Scot"),
    (7, "N West"),
    (8, "S & S East"),
    (9, "S West"),
    (10, "Ulster"),
    (11, "Wales & West"),
    (12, "Yorkshire"),
]

LOYAL_CLASS_CHOICES = [
    (0, "Gold"),
    (1, "Platinum"),
    (2, "Silver"),
    (3, "Tin"),
]


class PredictionForm(forms.Form):
    DemAffl = forms.IntegerField(label="DemAffl")
    DemAge = forms.IntegerField(label="DemAge")
    DemClusterGroup = forms.ChoiceField(
        choices=CLUSTER_GROUP_CHOICES, label="DemClusterGroup"
    )
    DemGender = forms.ChoiceField(choices=GENDER_CHOICES, label="DemGender")
    DemReg = forms.ChoiceField(choices=REGION_CHOICES, label="DemReg")
    DemTVReg = forms.ChoiceField(choices=TV_REGION_CHOICES, label="DemTVReg")
    LoyalClass = forms.ChoiceField(choices=LOYAL_CLASS_CHOICES, label="LoyalClass")
    LoyalTimeSpend = forms.FloatField(label="LoyalTimeSpend")
    LoyalTime = forms.IntegerField(label="LoyalTime")
