
from django.urls import path
from .views import ticket_dashboard

urlpatterns = [
    path("", ticket_dashboard, name="ticket_dashboard"),
]
