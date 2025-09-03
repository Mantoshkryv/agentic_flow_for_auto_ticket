from django import forms

class UploadFilesForm(forms.Form):
    session_file = forms.FileField(label="Session Data (xlsx)", required=True)
    kpi_file = forms.FileField(label="KPI Data (xlsx)", required=True)
    meta_file = forms.FileField(label="Metadata / advancetags (xlsx)", required=True)
