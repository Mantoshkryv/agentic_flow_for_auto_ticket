# tg/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Original dashboard view
    path('', views.dashboard_view, name='dashboard'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    
    # New page views
    path('management/', views.management_view, name='management'),
    path('dashboard/management/', views.management_view, name='management'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('dashboard/analytics/', views.analytics_view, name='analytics'),
    # API endpoints for the UI
    path('api/tickets/', views.tickets_api_view, name='tickets_api'),
    path('api/tickets/update/', views.ticket_update_api_view, name='ticket_update_api'),
    path('api/tickets/bulk-update/', views.ticket_bulk_update_api_view, name='ticket_bulk_update_api'),
    path('api/tickets/stats/', views.ticket_stats_api_view, name='ticket_stats_api'),
]
