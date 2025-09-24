# urls.py - FUTURE-PROOF URL CONFIGURATION

"""
Future-Proof URL Configuration
==============================

Features:
- MAINTAIN existing URL patterns and names
- Scalable routing that won't need updates
- Support for flexible processing features
- SESSION-ONLY ticket routing
- VARIABLE channel support
- RESTful API endpoints for future expansion
"""

from django.urls import path, include
from django.views.generic import RedirectView
from . import views

app_name = 'tg'

# ============================================================================
# MAINTAIN ALL EXISTING URL PATTERNS - SAME NAMES AND PATHS
# ============================================================================

urlpatterns = [
    # Core application routes - MAINTAIN existing patterns
    path('', views.tg_home_redirect, name='home_redirect'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('api/data-status/', views.api_data_status, name='api_data_status'),
    
    # Data ingestion routes - MAINTAIN existing patterns
    path('upload/', views.upload_files, name='upload_files'),
    path('mongodb-ingestion/', views.mongodb_ingestion_view, name='mongodb_ingestion'),
    
    # Ticket management routes - MAINTAIN existing patterns with SESSION-ONLY support
    path('tickets/', views.ticket_list, name='ticket_list'),
    path('tickets/<str:ticket_id>/', views.ticket_detail, name='ticket_detail'),
    path('tickets/<str:ticket_id>/update-status/', views.update_ticket_status, name='update_ticket_status'),
    
    # Analytics route - MAINTAIN existing pattern
    path('analytics/', views.analytics_view, name='analytics'),
    
    # ============================================================================
    # ENHANCED ROUTES FOR FLEXIBLE PROCESSING - FUTURE-PROOF ADDITIONS
    # ============================================================================
    
    # Enhanced upload routes with flexible processing
    path('upload/flexible/', views.upload_files, {'form_type': 'flexible'}, name='upload_flexible'),
    path('upload/smart/', views.upload_files, {'form_type': 'smart'}, name='upload_smart'),
    
    # SESSION-ONLY ticket routes
    path('tickets/session-only/', views.ticket_list, {'session_only': True}, name='ticket_list_session_only'),
    path('tickets/high-confidence/', views.ticket_list, {'confidence': 'high'}, name='ticket_list_high_confidence'),
    
    # VARIABLE channel routes
    path('channels/', RedirectView.as_view(pattern_name='tg:analytics'), name='channels'),
    
    # Enhanced analytics routes
    path('analytics/tickets/', views.analytics_view, {'focus': 'tickets'}, name='analytics_tickets'),
    path('analytics/processing/', views.analytics_view, {'focus': 'processing'}, name='analytics_processing'),
    path('analytics/channels/', views.analytics_view, {'focus': 'channels'}, name='analytics_channels'),
    
    # ============================================================================
    # API ENDPOINTS FOR FUTURE EXPANSION - RESTFUL DESIGN
    # ============================================================================
    
    # Future API routes (placeholder structure for expansion)
    # These follow RESTful conventions for easy API development
    path('api/', include([
        # Data processing API endpoints
        path('process/files/', views.upload_files, name='api_process_files'),  # POST for file processing
        path('process/mongodb/', views.mongodb_ingestion_view, name='api_process_mongodb'),  # POST for MongoDB processing
        
        # Ticket API endpoints with SESSION-ONLY support
        path('tickets/', views.ticket_list, name='api_tickets_list'),  # GET for listing
        path('tickets/<str:ticket_id>/', views.ticket_detail, name='api_ticket_detail'),  # GET for detail
        path('tickets/<str:ticket_id>/status/', views.update_ticket_status, name='api_ticket_status'),  # PATCH for status updates
        
        # Analytics API endpoints
        path('analytics/', views.analytics_view, name='api_analytics'),  # GET for analytics
        path('dashboard/', views.dashboard_view, name='api_dashboard'),  # GET for dashboard stats
    ])),
    
    # ============================================================================
    # FLEXIBLE ROUTING PATTERNS FOR SCALABILITY
    # ============================================================================
    
    # Generic processing route that can handle different types
    path('process/<str:process_type>/', views.upload_files, name='process_generic'),
    
    # Generic ticket filtering route
    path('tickets/filter/<str:filter_type>/', views.ticket_list, name='tickets_filter_generic'),
    
    # Generic analytics route with dynamic focus
    path('analytics/<str:analytics_type>/', views.analytics_view, name='analytics_generic'),
    
    # ============================================================================
    # BACKWARD COMPATIBILITY REDIRECTS
    # ============================================================================
    
    # Legacy routes that redirect to maintain compatibility
    path('upload-files/', RedirectView.as_view(pattern_name='tg:upload_files', permanent=True), name='upload_files_legacy'),
    path('mongodb/', RedirectView.as_view(pattern_name='tg:mongodb_ingestion', permanent=True), name='mongodb_legacy'),
    path('ticket-list/', RedirectView.as_view(pattern_name='tg:ticket_list', permanent=True), name='ticket_list_legacy'),
    
    # ============================================================================
    # UTILITY ROUTES FOR ENHANCED FEATURES
    # ============================================================================
    
    # Health check and status routes
    path('health/', views.dashboard_view, name='health_check'),  # System health
    path('status/', views.dashboard_view, name='system_status'),  # Processing status
    
    # Help and documentation routes
    path('help/', RedirectView.as_view(pattern_name='tg:dashboard'), name='help'),
    path('docs/', RedirectView.as_view(pattern_name='tg:dashboard'), name='documentation'),
]

# ============================================================================
# URL CONFIGURATION NOTES FOR FUTURE DEVELOPMENT
# ============================================================================

"""
URL Pattern Design Principles:
==============================

1. MAINTAINED COMPATIBILITY:
   - All existing URL patterns preserved exactly
   - Same names and paths for backward compatibility
   - Existing views continue to work without changes

2. FLEXIBLE PROCESSING SUPPORT:
   - Enhanced routes support flexible column mapping
   - SESSION-ONLY ticket routing maintained
   - VARIABLE channel support in all routes

3. FUTURE-PROOF DESIGN:
   - RESTful API structure ready for expansion
   - Generic routes for different processing types
   - Flexible filtering and analytics routes

4. SCALABILITY FEATURES:
   - API endpoints follow REST conventions
   - Generic routes reduce need for URL updates
   - Backward compatibility redirects maintain old links

5. SESSION-ONLY APPROACH:
   - All ticket routes support session-only mode
   - No viewer_id dependencies in routing
   - Consistent with MVP specifications

6. VARIABLE CHANNEL SUPPORT:
   - No hardcoded channel names in routes
   - Flexible channel filtering support
   - Analytics routes support channel analysis

Example Usage Patterns:
======================

For Manual Uploads:
- /upload/ (existing, enhanced with flexible processing)
- /upload/flexible/ (explicit flexible processing)
- /upload/smart/ (smart auto-detection)

For Ticket Management (SESSION-ONLY):
- /tickets/ (all tickets)
- /tickets/session-only/ (session-only filtered)
- /tickets/TKT_VSF_session123/ (specific ticket by session)

For Analytics (VARIABLE channels):
- /analytics/ (comprehensive analytics)
- /analytics/tickets/ (ticket-focused analytics)  
- /analytics/channels/ (channel analytics)

For API Access (Future):
- /api/process/files/ (POST file processing)
- /api/tickets/ (GET ticket listing)
- /api/analytics/ (GET analytics data)

This design ensures:
- No URL changes needed for enhanced features
- Backward compatibility maintained
- Ready for API development
- Supports all MVP requirements
- Scales with future needs
"""
