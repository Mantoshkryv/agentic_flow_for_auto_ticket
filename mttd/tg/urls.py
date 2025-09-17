from django.urls import path
from . import views

app_name = "tg"

urlpatterns = [
    # Home redirect to dashboard
    path("", views.tg_home_redirect, name="home_redirect"),

    # Dashboard overview
    path("dashboard/", views.dashboard_view, name="dashboard"),

    # File upload functionality
    path("upload/", views.upload_files, name="upload_files"),

    # Ticket management
    path("tickets/", views.ticket_list, name="ticket_list"),
    path("tickets/<str:ticket_id>/", views.ticket_detail, name="ticket_detail"),
    path("tickets/<str:ticket_id>/update/", views.update_ticket_status, name="update_ticket_status"),

    # Analytics view (FIXED - was missing)
    path("analytics/", views.analytics_view, name="analytics"),

    # MongoDB ingestion
    path("mongodb-ingestion/", views.mongodb_ingestion_view, name="mongodb_ingestion"),

    # API endpoints
    path("api/analytics/data/", views.analytics_data_api, name="analytics_data_api"),
    path("api/data-status/", views.api_data_ingestion_status, name="api_data_status"),
]
