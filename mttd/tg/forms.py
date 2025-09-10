from django import forms

class UploadFilesForm(forms.Form):
    session_file = forms.FileField(
        label="Upload Session Data (CSV or Excel)",
        widget=forms.FileInput(attrs={"accept": ".csv,.xlsx,.xls"})
    )
    kpi_file = forms.FileField(
        label="Upload KPI Data (CSV or Excel)",
        widget=forms.FileInput(attrs={"accept": ".csv,.xlsx,.xls"})
    )
    meta_file = forms.FileField(
        label="Upload Metadata (CSV or Excel)",
        widget=forms.FileInput(attrs={"accept": ".csv,.xlsx,.xls"})
    )

