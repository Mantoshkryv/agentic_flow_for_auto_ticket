# views.py - ENHANCED WITH UNIFIED PIPELINE INTEGRATION

"""
Enhanced Views with Unified Pipeline Integration
===============================================

Features:
- MAINTAIN all existing function names and signatures
- SESSION-ID ONLY ticket generation (no viewer_id)
- VARIABLE channel support (no hardcoded names)
- Flexible column mapping and data processing
- Import existing functions to prevent duplicacy
- Decoupled from data processing for future-proofing
"""

import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.utils import timezone
from django.db.models import Q, Count
from datetime import datetime, timedelta
import time
from .mongo_service import get_mongodb_service, test_mongodb_connection, get_mongodb_stats

def api_data_status(request):
    """API endpoint to check data processing status"""
    try:
        from .models import Session, KPI, Advancetags
        
        status = {
            'sessions': Session.objects.count(),
            'kpi_data': KPI.objects.count(),
            'advancetags': Advancetags.objects.count(),
            'last_update': timezone.now().isoformat(),
            'status': 'success'
        }
        return JsonResponse(status)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
 
# Import existing functions to prevent duplicacy
try:
    from .data_processing import create_flexible_processor, process_files_flexible, process_mongodb_flexible
    from .data_validation import create_data_validator
    from .models import Session, KPI, Advancetags, Ticket, DataIngestionLog, UnifiedDataManager
    from .operation.ticket_engine import create_mvp_ticket_engine
except ImportError as e:
    logging.getLogger(__name__).warning(f"Some imports failed: {e}")

# Import existing forms to prevent duplicacy
from .forms import FlexibleUploadForm, SmartUploadForm, get_form_by_preference

logger = logging.getLogger(__name__)

# ============================================================================
# MAINTAIN EXISTING VIEW FUNCTIONS - SAME NAMES AND SIGNATURES
# ============================================================================

def tg_home_redirect(request):
    """MAINTAIN existing function - Redirect to dashboard"""
    return redirect("tg:dashboard")

# Update dashboard view to show Django MongoDB status:
# Replace dashboard_view function in views.py
def dashboard_view(request):
    try:
        # MongoDB status
        mongodb_status = test_mongodb_connection()
        mongodb_stats = get_mongodb_stats()
        
        # REAL ticket statistics
        total_tickets = Ticket.objects.count()
        open_tickets = Ticket.objects.exclude(status__in=['resolved', 'closed']).count()
        high_priority_tickets = Ticket.objects.filter(priority__in=['high', 'critical']).count()
        
        # REAL recent data (24 hours)
        yesterday = timezone.now() - timedelta(days=1)
        recent_sessions = Session.objects.filter(created_at__gte=yesterday).count() if Session.objects.filter(created_at__isnull=False).exists() else Session.objects.count()
        recent_kpis = KPI.objects.count()  # KPI table may not have created_at
        recent_metadata = Advancetags.objects.count()  # Advancetags table may not have created_at
        
        # REAL chart data for frontend
        import json
        from django.db.models import Count
        
        status_distribution = list(Ticket.objects.values('status').annotate(count=Count('id')))
        priority_distribution = list(Ticket.objects.values('priority').annotate(count=Count('id')))
        
        # Weekly trend data
        weekly_tickets = []
        for i in range(7):
            date = (timezone.now() - timedelta(days=6-i)).date()
            count = Ticket.objects.filter(created_at__date=date).count()
            weekly_tickets.append({'date': date.strftime('%m/%d'), 'count': count})
        
        # Recent tickets
        recent_tickets = Ticket.objects.order_by('-created_at')[:10]
        
        context = {
            # REAL DATA matching template variable names
            'total_tickets': total_tickets,
            'open_tickets': open_tickets,
            'high_priority_tickets': high_priority_tickets,
            'recent_sessions': recent_sessions,
            'recent_kpis': recent_kpis,  
            'recent_metadata': recent_metadata,
            
            # REAL chart data (JSON serialized for JavaScript)
            'status_distribution': json.dumps(status_distribution),
            'priority_distribution': json.dumps(priority_distribution),
            'weekly_tickets': json.dumps(weekly_tickets),
            
            # REAL tickets list
            'recent_tickets': recent_tickets,
            
            # MongoDB integration
            'mongodb_connected': mongodb_status["connected"],
            'mongodb_status': mongodb_status,
            'mongodb_stats': mongodb_stats,
            
            # System capabilities
            'session_only_mode': True,
            'flexible_processing': True,
        }
        
        return render(request, "tg/dashboard.html", context)
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        # Return minimal context on error
        return render(request, "tg/dashboard.html", {
            'total_tickets': 0,
            'open_tickets': 0, 
            'high_priority_tickets': 0,
            'recent_sessions': 0,
            'recent_kpis': 0,
            'recent_metadata': 0,
            'recent_tickets': [],
            'mongodb_connected': False,
            'error': str(e)
        })
    
#def dashboard_view(request):
#    """FIXED: Dashboard with Django MongoDB integration"""
#    try:
#        # Test Django MongoDB connection  
#        mongodb_status = test_mongodb_connection()
#        mongodb_stats = get_mongodb_stats()
#        
#        # Get Django model stats
#        stats = UnifiedDataManager.get_data_statistics()
#        
#        # Merge with MongoDB stats
#        if mongodb_status["connected"]:
#            stats["mongodb_collections"] = mongodb_status["collections"]
#            stats["mongodb_detailed"] = mongodb_stats
#        
#        # Get recent processing activity
#        recent_ingestions = DataIngestionLog.objects.order_by('-started_at')[:10]
#        
#        # Enhanced ticket statistics
#        ticket_stats = Ticket.objects.aggregate(
#            total=Count('id'),
#            new=Count('id', filter=Q(status='new')),
#            in_progress=Count('id', filter=Q(status='in_progress')),
#            resolved=Count('id', filter=Q(status='resolved')),
#            high_priority=Count('id', filter=Q(priority='high')),
#            critical=Count('id', filter=Q(priority='critical')),
#            session_based=Count('id', filter=Q(session_id__isnull=False)),
#            high_confidence=Count('id', filter=Q(confidence_score__gte=0.8))
#        )
#        
#        # Processing metrics
#        processing_metrics = {
#            "total_batches": DataIngestionLog.objects.count(),
#            "successful_batches": DataIngestionLog.objects.filter(status='success').count(),
#            "failed_batches": DataIngestionLog.objects.filter(status='failed').count(),
#            "avg_processing_time": _calculate_avg_processing_time(),
#            "recent_success_rate": _calculate_recent_success_rate()
#        }
#        
#        context = {
#            "stats": stats,
#            "ticket_stats": ticket_stats,
#            "processing_metrics": processing_metrics,
#            "recent_ingestions": recent_ingestions,
#            "can_ingest_mongodb": mongodb_status["connected"],
#            "mongodb_connected": mongodb_status["connected"],
#            "mongodb_status": mongodb_status,
#            "mongodb_stats": mongodb_stats,
#            "can_upload_files": True,
#            "session_only_mode": True,
#            "flexible_processing": True,
#            "django_mongodb_backend": True  # Indicate we're using Django MongoDB backend
#        }
#        
#        return render(request, "tg/dashboard.html", context)
#        
#    except Exception as e:
#        logger.error(f"❌ Dashboard error: {e}", exc_info=True)
#        messages.error(request, f"Dashboard loading error: {str(e)}")
#        
#        # Fallback context
#        mongodb_status = {"connected": False, "error": str(e)}
#        context = {
#            "stats": {"sessions": {"total": 0}, "tickets": {"total": 0}, "kpi": {"total": 0}, "advancetags": {"total": 0}},
#            "recent_ingestions": [],
#            "mongodb_connected": False,
#            "mongodb_status": mongodb_status,
#            "error": str(e)
#        }
#        return render(request, "tg/dashboard.html", context)
#
## ============================================================================
## ENHANCED FILE UPLOAD PROCESSING - MAINTAIN EXISTING FUNCTION NAME
# ============================================================================

@csrf_exempt
def upload_files(request):
    """
    MAINTAIN existing function name and structure
    Enhanced with FLEXIBLE processing, SESSION-ONLY tickets, VARIABLE channels
    """
    if request.method == "POST":
        # Form selection - MAINTAIN existing logic but enhanced
        form_type = request.POST.get('form_type', 'flexible')
        
        # Get appropriate form using existing pattern
        FormClass = get_form_by_preference(form_type)
        form = FormClass(request.POST, request.FILES)
        
        if form.is_valid():
            start_time = time.time()
            
            try:
                # Get file mapping and processing config - USE EXISTING METHODS
                files_mapping = form.get_file_mapping()
                processing_config = form.get_processing_config()
                
                if not files_mapping:
                    messages.error(request, "No valid files found to process")
                    return redirect("tg:upload_files")
                
                # Extract VARIABLE channels (no hardcoded names)
                target_channels = processing_config.get('target_channels', [])
                
                logger.info(f"Starting flexible upload processing: {len(files_mapping)} files")
                logger.info(f"Target channels: {target_channels or 'All channels'}")
                logger.info(f"Session-only tickets: {processing_config.get('session_only_tickets', True)}")
                
                # Process files using ENHANCED unified pipeline with FLEXIBLE processing
                result = process_files_flexible_enhanced(files_mapping, target_channels, processing_config)
                
                # Handle result - MAINTAIN existing structure
                if result.get('success', False):
                    success_msg = build_success_message(result)
                    messages.success(request, success_msg)
                    
                    # Show warnings if any
                    warnings = result.get('warnings', [])
                    for warning in warnings[:3]:  # Limit to 3 warnings
                        messages.warning(request, f"Warning: {warning}")
                    
                    logger.info("Flexible upload successful")
                    return redirect("tg:dashboard")
                    
                else:
                    error_msg = f"File upload failed: {'; '.join(result.get('errors', ['Unknown error']))}"
                    messages.error(request, error_msg)
                    logger.error("Flexible upload failed")
                    
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = f"Upload processing failed: {str(e)} (after {processing_time:.1f}s)"
                messages.error(request, error_msg)
                logger.error(f"File upload exception: {e}", exc_info=True)
        
        else:
            # Form validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    
    else:
        # Form selection for GET request - MAINTAIN existing pattern
        form_preference = request.GET.get('type', 'flexible')
        FormClass = get_form_by_preference(form_preference)
        form = FormClass()
    
    # Render upload form with enhanced context
    context = {
        'form': form,
        'form_type': form_preference if 'form_preference' in locals() else 'flexible',
        'recent_uploads': DataIngestionLog.objects.filter(
            source_type='manual'
        ).order_by('-started_at')[:8],
        'processing_tips': [
            "Supports CSV and Excel files (.csv, .xlsx, .xls)",
            "Flexible column mapping handles data from any position", 
            "Automatic removal of instruction headers and blank data",
            "SESSION-ID ONLY ticket generation for Video Start Failures",
            "Variable channel support - no hardcoded channel names",
            "Advanced data cleaning and validation"
        ],
        'supported_features': [
            'Flexible Column Mapping',
            'SESSION-Only Tickets',
            'Variable Channels',
            'Smart Data Cleaning',
            'MVP Diagnosis Rules'
        ]
    }
    
    return render(request, "tg/upload.html", context)

def process_files_flexible_enhanced(files_mapping, target_channels=None, processing_config=None):
    """
    Enhanced file processing with flexible pipeline integration
    Uses imported functions to prevent duplicacy
    """
    try:
        # Use existing flexible processor
        processor = create_flexible_processor()
        
        # Configure processor based on form settings
        if processing_config:
            processor.config.session_only_tickets = processing_config.get('session_only_tickets', True)
            processor.config.flexible_column_mapping = processing_config.get('flexible_column_mapping', True)
            processor.config.remove_blank_columns_threshold = 0.95 if processing_config.get('remove_blank_data') else 0.0
            processor.config.remove_blank_rows_threshold = 0.90 if processing_config.get('remove_blank_data') else 0.0
        
        try:
            # Process with flexible pipeline - SESSION-ONLY approach
            result = processor.process_manual_upload_flexible(files_mapping, target_channels)
            
            # Add processing configuration info to result
            if result.get('success'):
                result['processing_config'] = processing_config
                result['warnings'] = result.get('warnings', [])
                
                # Add validation warnings if available
                stats = result.get('stats', {})
                if stats.get('errors'):
                    result['warnings'].extend(stats['errors'][:3])
            
            return result
            
        finally:
            # Always cleanup processor resources
            processor.cleanup()
            
    except Exception as e:
        logger.error(f"Enhanced file processing failed: {e}", exc_info=True)
        return {
            'success': False,
            'errors': [str(e)]
        }

# ============================================================================
# MONGODB INGESTION - MAINTAIN EXISTING FUNCTION NAME
# ============================================================================

@csrf_exempt  
def mongodb_ingestion_view(request):
    """FIXED: MongoDB ingestion using Django ORM (best approach with django_mongodb_backend)"""
    
    if request.method == "POST":
        start_time = time.time()
        try:
            logger.info("🚀 Starting Django MongoDB processing")
            
            # Get optional target channels
            target_channels_str = request.POST.get('target_channels', '')
            target_channels = []
            if target_channels_str:
                target_channels = [ch.strip() for ch in target_channels_str.split(',') if ch.strip()]
            
            logger.info(f"🎯 Target channels: {target_channels or 'All channels'}")
            
            # Test Django MongoDB connection
            connection_status = test_mongodb_connection()
            if not connection_status["connected"]:
                error_msg = f"Django MongoDB connection failed: {connection_status['error']}"
                messages.error(request, error_msg)
                logger.error(error_msg)
                return redirect('tg:mongodb_ingestion')
            
            logger.info(f"✅ Django MongoDB verified: {connection_status['collections']}")
            
            # Check if we have data to process
            total_available = sum(connection_status["collections"].values())
            if total_available == 0:
                error_msg = "No data found in MongoDB collections. Please ensure data is loaded into your Django models."
                messages.warning(request, error_msg)
                logger.warning(error_msg)
                return redirect('tg:mongodb_ingestion')
            
            # Process using Django ORM approach
            result = process_mongodb_flexible_enhanced(target_channels)

            if result.get('success', False):
                success_msg = build_mongodb_success_message(result)
                messages.success(request, success_msg)
                logger.info("🎉 Django MongoDB processing successful")
            else:
                error_msg = f"MongoDB processing failed: {result.get('error', 'Unknown error')}"
                messages.error(request, error_msg)
                logger.error("❌ Django MongoDB processing failed")
            
            return redirect('tg:dashboard')
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"MongoDB processing failed: {str(e)} (after {processing_time:.1f}s)"
            messages.error(request, error_msg)
            logger.error(f"❌ MongoDB processing exception: {e}", exc_info=True)
    
    # GET request - show form with REAL data counts
    try:
        # Get real collection counts using Django ORM
        mongodb_stats = get_mongodb_stats()
        connection_status = test_mongodb_connection()
        
        if connection_status["connected"]:
            session_count = mongodb_stats.get("sessions", {}).get("total", 0)
            kpi_count = mongodb_stats.get("kpi", {}).get("total", 0)  
            metadata_count = mongodb_stats.get("advancetags", {}).get("total", 0)
            mongodb_connected = True
            
            logger.info(f"📊 Real MongoDB counts: Sessions={session_count}, KPI={kpi_count}, Meta={metadata_count}")
        else:
            session_count = kpi_count = metadata_count = 0
            mongodb_connected = False
            logger.warning(f"❌ Django MongoDB not connected: {connection_status.get('error', 'Unknown error')}")
        
        # Get recent processing logs
        recent_ingestions = DataIngestionLog.objects.filter(
            source_type='mongodb'
        ).order_by('-started_at')[:8]
        
        context = {
            "session_count": session_count,
            "kpi_count": kpi_count,
            "metadata_count": metadata_count,
            "mongodb_connected": mongodb_connected,
            "connection_status": connection_status,
            "mongodb_stats": mongodb_stats,
            "recent_ingestions": recent_ingestions,
            "mongodb_features": [
                "Django ORM MongoDB integration",
                "django_mongodb_backend compatibility", 
                "SESSION-ID ONLY ticket generation",
                "Variable channel filtering",
                "Flexible data processing",
                "Real collection statistics",
                "Enhanced error handling"
            ]
        }
        
        return render(request, "tg/mongodb_ingestion.html", context)
        
    except Exception as e:
        logger.error(f"❌ Error loading MongoDB page: {e}")
        context = {
            "error": str(e),
            "session_count": 0,
            "kpi_count": 0,
            "metadata_count": 0,
            "mongodb_connected": False
        }
        return render(request, "tg/mongodb_ingestion.html", context)

# Replace process_mongodb_flexible_enhanced function:

def process_mongodb_flexible_enhanced(target_channels=None):
    """FIXED: Enhanced MongoDB processing using Django ORM"""
    try:
        logger.info("🔄 Enhanced Django MongoDB processing starting...")
        
        # Test connection first
        connection_status = test_mongodb_connection()
        if not connection_status["connected"]:
            logger.error(f"❌ Django MongoDB connection failed: {connection_status['error']}")
            return {
                "success": False, 
                "error": f"Django MongoDB connection failed: {connection_status['error']}",
                "django_orm_attempted": True
            }
        
        logger.info("✅ Django MongoDB connection verified")
        
        # Use the flexible processor with Django ORM
        result = process_mongodb_flexible(target_channels)
        
        if result.get('success'):
            # Add enhanced information
            result['django_orm_processing'] = True
            result['flexible_processing_used'] = True
            result['connection_verified'] = True
            result['backend'] = 'django_mongodb_backend'
            logger.info(f"🎉 Django MongoDB processing successful: {result.get('tickets_generated', 0)} tickets")
        else:
            logger.error(f"❌ Django MongoDB processing failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Enhanced Django MongoDB processing failed: {e}", exc_info=True)
        return {
            "success": False, 
            "error": str(e),
            "django_orm_attempted": True
        }

# ============================================================================
# TICKET MANAGEMENT - MAINTAIN EXISTING FUNCTION NAMES
# ============================================================================

# CORRECTED ticket_list VIEW FUNCTION
# FIXED views.py functions - Replace the existing functions with these

def ticket_list(request):
    """FIXED ticket list - corrected field references and search"""
    try:
        # Get filter parameters - matching what template expects
        current_search = request.GET.get('search', '').strip()
        current_status = request.GET.get('status', '')
        current_priority = request.GET.get('priority', '')
        per_page = request.GET.get('per_page', '25')
        
        # Build queryset for displayed tickets
        tickets = Ticket.objects.all()
        
        # Apply filters
        if current_status and current_status != 'all':
            tickets = tickets.filter(status=current_status)
        if current_priority and current_priority != 'all':
            tickets = tickets.filter(priority=current_priority)
        
        # FIXED SEARCH - Use correct field names from your Ticket model
        if current_search:
            tickets = tickets.filter(
                Q(ticket_id__icontains=current_search) |
                Q(title__icontains=current_search) |           # Use 'title' instead of 'root_cause'
                Q(description__icontains=current_search) |     # Add description search
                Q(session_id__icontains=current_search) |
                Q(assign_team__icontains=current_search) |
                Q(issue_type__icontains=current_search)        # Add issue_type search
            )
        
        # Order tickets by creation date (newest first)
        tickets = tickets.order_by('-created_at')
        
        # Get status and priority choices from Ticket model safely
        status_choices = []
        priority_choices = []
        
        try:
            if hasattr(Ticket, 'STATUS_CHOICES'):
                status_choices = [choice[0] for choice in Ticket.STATUS_CHOICES]
            else:
                status_choices = ['new', 'in_progress', 'resolved', 'closed']
                
            if hasattr(Ticket, 'PRIORITY_CHOICES'):
                priority_choices = [choice[0] for choice in Ticket.PRIORITY_CHOICES]
            else:
                priority_choices = ['low', 'medium', 'high', 'critical']
        except Exception as e:
            logger.warning(f"Could not get model choices: {e}")
            status_choices = ['new', 'in_progress', 'resolved', 'closed']
            priority_choices = ['low', 'medium', 'high', 'critical']
        
        # CALCULATE REAL STATISTICS - NO FAKE DATA
        all_tickets = Ticket.objects.all()
        
        # Real open tickets count (not resolved or closed)
        open_tickets_count = all_tickets.exclude(status__in=['resolved', 'closed']).count()
        
        # Real high priority count
        high_priority_count = all_tickets.filter(priority='high').count()
        
        # Real teams involved count
        teams_involved = all_tickets.exclude(
            assign_team__isnull=True
        ).exclude(
            assign_team=''
        ).values_list('assign_team', flat=True).distinct().count()
        
        # Real average response time calculation
        from django.utils import timezone
        from datetime import timedelta
        
        resolved_tickets = all_tickets.filter(
            status__in=['resolved', 'closed'],
            resolved_at__isnull=False,
            created_at__isnull=False
        )
        
        avg_response_time = "No data"
        if resolved_tickets.exists():
            total_response_seconds = 0
            count = 0
            
            for ticket in resolved_tickets:
                if ticket.resolved_at and ticket.created_at:
                    response_time = ticket.resolved_at - ticket.created_at
                    total_response_seconds += response_time.total_seconds()
                    count += 1
            
            if count > 0:
                avg_seconds = total_response_seconds / count
                
                if avg_seconds < 3600:  # Less than 1 hour
                    avg_response_time = f"{int(avg_seconds // 60)}m"
                elif avg_seconds < 86400:  # Less than 1 day
                    hours = int(avg_seconds // 3600)
                    minutes = int((avg_seconds % 3600) // 60)
                    avg_response_time = f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
                else:  # 1 day or more
                    days = int(avg_seconds // 86400)
                    hours = int((avg_seconds % 86400) // 3600)
                    avg_response_time = f"{days}d {hours}h" if hours > 0 else f"{days}d"
        
        # PAGINATION HANDLING - Support for "show all" 
        from django.core.paginator import Paginator
        
        if per_page == 'all':
            # Show all tickets without pagination
            page_obj = tickets
            total_count = tickets.count()
            # Add pagination-like attributes for template compatibility
            page_obj.has_other_pages = False
            page_obj.has_previous = False
            page_obj.has_next = False
            page_obj.start_index = 1 if tickets.exists() else 0
            page_obj.end_index = total_count
            page_obj.number = 1
        else:
            # Normal pagination
            try:
                page_size = int(per_page)
                if page_size not in [25, 50, 100]:
                    page_size = 25
            except (ValueError, TypeError):
                page_size = 25
            
            paginator = Paginator(tickets, page_size)
            total_count = paginator.count
            page_obj = paginator.get_page(request.GET.get('page'))
        
        # Complete context with ALL required variables
        context = {
            # Paginated tickets (or all tickets if per_page='all')
            'tickets': page_obj,
            
            # Filter choices for dropdowns - REQUIRED by template
            'status_choices': status_choices,
            'priority_choices': priority_choices,
            
            # Current filter values - REQUIRED individual variables (not dictionary)
            'current_search': current_search,
            'current_status': current_status,
            'current_priority': current_priority,
            
            # Pagination info
            'total_count': total_count,
            
            # REAL CALCULATED STATISTICS - NO FAKE DATA
            'open_tickets_count': open_tickets_count,
            'high_priority_count': high_priority_count,
            'teams_involved': teams_involved,
            'avg_response_time': avg_response_time,
            
            # Additional real statistics for enhanced display
            'total_tickets': all_tickets.count(),
            'resolved_tickets': all_tickets.filter(status='resolved').count(),
            'closed_tickets': all_tickets.filter(status='closed').count(),
            'new_tickets': all_tickets.filter(status='new').count(),
            'in_progress_tickets': all_tickets.filter(status='in_progress').count(),
            'critical_tickets': all_tickets.filter(priority='critical').count(),
        }
        
        return render(request, "tg/ticket_list.html", context)
        
    except Exception as e:
        logger.error(f"Ticket list error: {e}", exc_info=True)
        messages.error(request, f"Error loading tickets: {e}")
        
        # Fallback context with real empty values (not fake data)
        return render(request, "tg/ticket_list.html", {
            'tickets': [],
            'status_choices': ['new', 'in_progress', 'resolved', 'closed'],
            'priority_choices': ['low', 'medium', 'high', 'critical'],
            'current_search': '',
            'current_status': '',
            'current_priority': '',
            'total_count': 0,
            'open_tickets_count': 0,
            'high_priority_count': 0,
            'teams_involved': 0,
            'avg_response_time': 'No data',
            'total_tickets': 0,
            'resolved_tickets': 0,
            'closed_tickets': 0,
        })

def ticket_detail(request, ticket_id):
    """FIXED ticket detail with proper field extraction"""
    try:
        # Get ticket with error handling
        ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
        
        # Get related session data if available using SESSION-ID ONLY
        related_session = None
        if ticket.session_id:
            try:
                related_session = Session.objects.get(session_id=ticket.session_id)
            except Session.DoesNotExist:
                logger.warning(f"Session {ticket.session_id} not found for ticket {ticket_id}")
            except Exception as e:
                logger.error(f"Error fetching session {ticket.session_id}: {e}")
        
        # Get related metadata if available
        related_metadata = None
        if ticket.session_id:
            try:
                related_metadata = Advancetags.objects.get(session_id=ticket.session_id)
            except Advancetags.DoesNotExist:
                logger.debug(f"No metadata found for session {ticket.session_id}")
            except Exception as e:
                logger.error(f"Error fetching metadata for session {ticket.session_id}: {e}")
        # FIXED: Extract channel name and other data from context_data or failure_details
        channel_name = None
        root_cause = None

        # Try multiple sources for channel name
        channel_sources = []

        # 1. Try context_data first
        if ticket.context_data:
            try:
                if isinstance(ticket.context_data, dict):
                    context_data = ticket.context_data
                elif isinstance(ticket.context_data, str):
                    import json
                    context_data = json.loads(ticket.context_data)
                else:
                    context_data = {}

                # Extract channel name from various possible fields
                channel_name = (
                    context_data.get('asset_name') or 
                    context_data.get('channel') or 
                    context_data.get('channel_name') or
                    context_data.get('Asset Name') or  # Case variations
                    context_data.get('Channel') or
                    context_data.get('Channel Name')
                )
                if channel_name:
                    channel_sources.append(f"context_data: {channel_name}")

            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Invalid context_data format for ticket {ticket_id}: {e}")
                context_data = {}
        else:
            context_data = {}

        # 2. Try failure_details if no channel from context_data
        if not channel_name and ticket.failure_details:
            try:
                if isinstance(ticket.failure_details, dict):
                    failure_details = ticket.failure_details
                elif isinstance(ticket.failure_details, str):
                    import json
                    failure_details = json.loads(ticket.failure_details)
                else:
                    failure_details = {}

                channel_name = (
                    failure_details.get('asset_name') or 
                    failure_details.get('channel') or 
                    failure_details.get('channel_name') or
                    failure_details.get('Asset Name') or
                    failure_details.get('Channel') or
                    failure_details.get('Channel Name')
                )
                if channel_name:
                    channel_sources.append(f"failure_details: {channel_name}")

                # Extract root cause from failure details
                root_cause = failure_details.get('root_cause') or failure_details.get('issue') or failure_details.get('error')

            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Invalid failure_details format for ticket {ticket_id}: {e}")
                failure_details = {}
        else:
            # Initialize failure_details if we didn't process it above
            if ticket.failure_details:
                try:
                    if isinstance(ticket.failure_details, dict):
                        failure_details = ticket.failure_details
                    elif isinstance(ticket.failure_details, str):
                        import json
                        failure_details = json.loads(ticket.failure_details)
                    else:
                        failure_details = {}
                    root_cause = failure_details.get('root_cause') or failure_details.get('issue') or failure_details.get('error')
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Invalid failure_details format for ticket {ticket_id}: {e}")
                    failure_details = {}
            else:
                failure_details = {}

        # 3. Try related session if still no channel
        if not channel_name and related_session and hasattr(related_session, 'asset_name'):
            channel_name = related_session.asset_name
            if channel_name:
                channel_sources.append(f"related_session: {channel_name}")
        
        # Fallback: Use ticket title/description as root cause if not found in failure_details
        if not root_cause:
            root_cause = ticket.title or ticket.description
        
        # Extract and validate suggested actions
        suggested_actions = []
        if ticket.suggested_actions:
            try:
                if isinstance(ticket.suggested_actions, list):
                    suggested_actions = ticket.suggested_actions
                elif isinstance(ticket.suggested_actions, str):
                    import json
                    suggested_actions = json.loads(ticket.suggested_actions)
                # Ensure it's a list
                if not isinstance(suggested_actions, list):
                    suggested_actions = []
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Invalid suggested_actions format for ticket {ticket_id}: {e}")
                suggested_actions = []
        
        # Enhanced diagnosis check
        mvp_diagnosis = bool(
            root_cause or 
            failure_details or
            ticket.title or 
            ticket.description
        )
        
        # Enhanced context with additional useful data
        context = {
            # Core ticket data
            'ticket': ticket,
            'related_session': related_session,
            'related_metadata': related_metadata,
            
            # FIXED: Extracted data for template display
            'root_cause': root_cause,
            'channel_name': channel_name,
            'confidence_score': ticket.confidence_score,
            
            # Processed data
            'failure_details': failure_details,
            'context_data': context_data,
            'suggested_actions': suggested_actions,
            
            # Permissions and modes
            'can_edit': True,
            'session_only_mode': True,  # SESSION-ONLY mode
            'mvp_diagnosis': mvp_diagnosis,
            
            # Additional helpful context
            'has_failure_details': bool(failure_details),
            'has_suggested_actions': bool(suggested_actions),
            'has_related_session': bool(related_session),
            'ticket_age_days': (timezone.now() - ticket.created_at).days if ticket.created_at else 0,
            'is_resolved': ticket.status in ['resolved', 'closed'],
            'is_high_priority': ticket.priority in ['high', 'critical'],
        }
        
        return render(request, "tg/ticket_detail.html", context)
        
    except Exception as e:
        logger.error(f"Ticket detail error for {ticket_id}: {e}", exc_info=True)
        messages.error(request, f"Error loading ticket {ticket_id}: {str(e)}")
        return redirect("tg:ticket_list")
    
# Replace update_ticket_status function (around lines 454-504) with this:

@csrf_exempt
def update_ticket_status(request, ticket_id):
    """COMPLETE FIXED ticket update - status, priority, team assignment"""
    
    if request.method == "GET":
        # Handle GET request - redirect to ticket detail page
        try:
            ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
            return redirect('tg:ticket_detail', ticket_id=ticket_id)
        except Exception as e:
            logger.error(f"Error redirecting to ticket detail: {e}")
            messages.error(request, f"Error loading ticket: {str(e)}")
            return redirect('tg:ticket_list')
    
    elif request.method == "POST":
        try:
            ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
            
            # Get all update fields
            new_status = request.POST.get("status", "").strip()
            new_priority = request.POST.get("priority", "").strip()
            new_team = request.POST.get("assign_team", "").strip()
            resolution_notes = request.POST.get("resolution_notes", "").strip()
            
            # Track changes for detailed response
            changes = {}
            old_values = {}
            
            # Validate and update status
            if new_status:
                try:
                    if hasattr(Ticket, 'STATUS_CHOICES'):
                        valid_statuses = [choice[0] for choice in Ticket.STATUS_CHOICES]
                    else:
                        valid_statuses = ['new', 'in_progress', 'resolved', 'closed']
                except:
                    valid_statuses = ['new', 'in_progress', 'resolved', 'closed']
                    
                if new_status not in valid_statuses:
                    return JsonResponse({
                        "success": False, 
                        "error": f"Invalid status: {new_status}. Valid options: {', '.join(valid_statuses)}"
                    })
                
                if ticket.status != new_status:
                    old_values['status'] = ticket.status
                    ticket.status = new_status
                    changes['status'] = new_status
            
            # Validate and update priority
            if new_priority:
                try:
                    if hasattr(Ticket, 'PRIORITY_CHOICES'):
                        valid_priorities = [choice[0] for choice in Ticket.PRIORITY_CHOICES]
                    else:
                        valid_priorities = ['low', 'medium', 'high', 'critical']
                except:
                    valid_priorities = ['low', 'medium', 'high', 'critical']
                    
                if new_priority not in valid_priorities:
                    return JsonResponse({
                        "success": False, 
                        "error": f"Invalid priority: {new_priority}. Valid options: {', '.join(valid_priorities)}"
                    })
                
                if ticket.priority != new_priority:
                    old_values['priority'] = ticket.priority
                    ticket.priority = new_priority
                    changes['priority'] = new_priority
            
            # Update team assignment (can be empty to unassign)
            if 'assign_team' in request.POST:
                current_team = ticket.assign_team or ""
                if current_team != new_team:
                    old_values['assign_team'] = current_team or "Unassigned"
                    ticket.assign_team = new_team if new_team else None
                    changes['assign_team'] = new_team or "Unassigned"
            
            # Update resolution notes
            if resolution_notes and ticket.resolution_notes != resolution_notes:
                old_values['resolution_notes'] = ticket.resolution_notes or ""
                ticket.resolution_notes = resolution_notes
                changes['resolution_notes'] = "Updated"
            
            # Handle resolved timestamp automatically
            if 'status' in changes:
                from django.utils import timezone
                if new_status == 'resolved' and old_values.get('status') != 'resolved':
                    ticket.resolved_at = timezone.now()
                    changes['resolved_at'] = str(ticket.resolved_at)
                elif new_status != 'resolved' and old_values.get('status') == 'resolved':
                    ticket.resolved_at = None
                    changes['resolved_at'] = "Cleared"
            
            # Save changes if any were made
            if changes:
                ticket.save()
                
                # Create detailed change summary for user feedback
                change_details = []
                for field, new_value in changes.items():
                    old_val = old_values.get(field, "Unknown")
                    if field == 'status':
                        change_details.append(f"Status: {old_val.title()} → {new_value.title()}")
                    elif field == 'priority':
                        change_details.append(f"Priority: {old_val.title()} → {new_value.title()}")
                    elif field == 'assign_team':
                        change_details.append(f"Team: {old_val} → {new_value}")
                    elif field == 'resolution_notes':
                        change_details.append("Resolution notes updated")
                    elif field == 'resolved_at':
                        if new_value == "Cleared":
                            change_details.append("Resolved timestamp cleared")
                        else:
                            change_details.append("Marked as resolved")
                
                success_message = f"Ticket updated successfully! Changes: {', '.join(change_details)}"
                
                # Log the change for audit trail
                logger.info(f"Ticket {ticket_id} updated. Changes: {changes}")
                
                return JsonResponse({
                    "success": True,
                    "message": success_message,
                    "changes": changes,
                    "new_status": ticket.status,
                    "new_priority": ticket.priority,
                    "new_team": ticket.assign_team,
                    "session_id": ticket.session_id,
                    "resolved_at": str(ticket.resolved_at) if ticket.resolved_at else None,
                    "updated_at": str(ticket.updated_at)
                })
            else:
                return JsonResponse({
                    "success": True,
                    "message": "No changes were made to the ticket",
                    "changes": {}
                })
            
        except Exception as e:
            logger.error(f"Ticket update error for {ticket_id}: {e}", exc_info=True)
            return JsonResponse({
                "success": False,
                "error": f"Failed to update ticket: {str(e)}"
            })
    
    return JsonResponse({
        "success": False, 
        "error": "Invalid request method. Use GET to view or POST to update."
    })

@csrf_exempt
def update_ticket_status(request, ticket_id):
    """Complete enhanced ticket update - status, priority, team assignment"""
    
    if request.method == "GET":
        # Handle GET request - redirect to ticket detail page
        try:
            ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
            return redirect('tg:ticket_detail', ticket_id=ticket_id)
        except Exception as e:
            logger.error(f"Error redirecting to ticket detail: {e}")
            messages.error(request, f"Error loading ticket: {str(e)}")
            return redirect('tg:ticket_list')
    
    elif request.method == "POST":
        try:
            ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
            
            # Get all update fields
            new_status = request.POST.get("status", "").strip()
            new_priority = request.POST.get("priority", "").strip()
            new_team = request.POST.get("assign_team", "").strip()
            resolution_notes = request.POST.get("resolution_notes", "").strip()
            
            # Track changes
            changes = {}
            old_values = {}
            
            # Validate and update status
            if new_status:
                # Get valid statuses - try model choices first, fallback to common values
                try:
                    if hasattr(Ticket, 'STATUS_CHOICES'):
                        valid_statuses = [choice[0] for choice in Ticket.STATUS_CHOICES]
                    else:
                        valid_statuses = ['new', 'in_progress', 'resolved', 'closed']
                except:
                    valid_statuses = ['new', 'in_progress', 'resolved', 'closed']
                    
                if new_status not in valid_statuses:
                    return JsonResponse({
                        "success": False, 
                        "error": f"Invalid status: {new_status}. Valid options: {', '.join(valid_statuses)}"
                    })
                
                if ticket.status != new_status:
                    old_values['status'] = ticket.status
                    ticket.status = new_status
                    changes['status'] = new_status
            
            # Validate and update priority
            if new_priority:
                # Get valid priorities - try model choices first, fallback to common values
                try:
                    if hasattr(Ticket, 'PRIORITY_CHOICES'):
                        valid_priorities = [choice[0] for choice in Ticket.PRIORITY_CHOICES]
                    else:
                        valid_priorities = ['low', 'medium', 'high', 'critical']
                except:
                    valid_priorities = ['low', 'medium', 'high', 'critical']
                    
                if new_priority not in valid_priorities:
                    return JsonResponse({
                        "success": False, 
                        "error": f"Invalid priority: {new_priority}. Valid options: {', '.join(valid_priorities)}"
                    })
                
                if ticket.priority != new_priority:
                    old_values['priority'] = ticket.priority
                    ticket.priority = new_priority
                    changes['priority'] = new_priority
            
            # Update team assignment (can be empty to unassign)
            if 'assign_team' in request.POST:
                current_team = ticket.assign_team or ""
                if current_team != new_team:
                    old_values['assign_team'] = current_team or "Unassigned"
                    ticket.assign_team = new_team if new_team else None
                    changes['assign_team'] = new_team or "Unassigned"
            
            # Update resolution notes
            if resolution_notes and ticket.resolution_notes != resolution_notes:
                old_values['resolution_notes'] = ticket.resolution_notes or ""
                ticket.resolution_notes = resolution_notes
                changes['resolution_notes'] = "Updated"
            
            # Handle resolved timestamp
            if 'status' in changes:
                from django.utils import timezone
                if new_status == 'resolved' and old_values.get('status') != 'resolved':
                    ticket.resolved_at = timezone.now()
                    changes['resolved_at'] = str(ticket.resolved_at)
                elif new_status != 'resolved' and old_values.get('status') == 'resolved':
                    ticket.resolved_at = None
                    changes['resolved_at'] = "Cleared"
            
            # Save if there are changes
            if changes:
                ticket.save()
                
                # Create detailed change summary
                change_details = []
                for field, new_value in changes.items():
                    old_val = old_values.get(field, "Unknown")
                    if field == 'status':
                        change_details.append(f"Status: {old_val} → {new_value}")
                    elif field == 'priority':
                        change_details.append(f"Priority: {old_val} → {new_value}")
                    elif field == 'assign_team':
                        change_details.append(f"Team: {old_val} → {new_value}")
                    elif field == 'resolution_notes':
                        change_details.append("Resolution notes updated")
                    elif field == 'resolved_at':
                        if new_value == "Cleared":
                            change_details.append("Resolved timestamp cleared")
                        else:
                            change_details.append("Marked as resolved")
                
                success_message = f"Ticket updated successfully. {', '.join(change_details)}"
                
                # Log the change
                logger.info(f"Ticket {ticket_id} updated. Changes: {changes}")
                
                return JsonResponse({
                    "success": True,
                    "message": success_message,
                    "changes": changes,
                    "new_status": ticket.status,
                    "new_priority": ticket.priority,
                    "new_team": ticket.assign_team,
                    "session_id": ticket.session_id,
                    "resolved_at": str(ticket.resolved_at) if ticket.resolved_at else None,
                    "updated_at": str(ticket.updated_at)
                })
            else:
                return JsonResponse({
                    "success": True,
                    "message": "No changes were made to the ticket",
                    "changes": {}
                })
            
        except Exception as e:
            logger.error(f"Ticket update error for {ticket_id}: {e}", exc_info=True)
            return JsonResponse({
                "success": False,
                "error": f"Failed to update ticket: {str(e)}"
            })
    
    return JsonResponse({
        "success": False, 
        "error": "Invalid request method. Use GET to view or POST to update."
    })

# ============================================================================
# ANALYTICS - MAINTAIN EXISTING FUNCTION NAME
# ============================================================================

def analytics_view(request):
    """MAINTAIN existing function name - Enhanced analytics with session-only insights"""
    try:
        # Use existing utility function
        stats = UnifiedDataManager.get_data_statistics()
        
        # Enhanced analytics with session-based insights
        recent_logs = DataIngestionLog.objects.filter(
            started_at__gte=timezone.now() - timedelta(days=7),
            status='success'
        ).order_by('-started_at')
        
        # SESSION-ONLY ticket analytics
        ticket_analytics = get_session_ticket_analytics()
        
        # Processing performance analytics
        processing_analytics = get_processing_performance_analytics()
        
        # Channel analytics (VARIABLE channels)
        channel_analytics = get_channel_analytics()
        
        context = {
            'stats': stats,
            'ticket_analytics': ticket_analytics,
            'processing_analytics': processing_analytics,
            'channel_analytics': channel_analytics,
            'recent_processing': recent_logs[:15],
            'session_only_mode': True,
            'flexible_processing': True
        }
        
        return render(request, "tg/analytics.html", context)
        
    except Exception as e:
        logger.error(f"Analytics error: {e}", exc_info=True)
        messages.error(request, f"Error loading analytics: {str(e)}")
        return render(request, "tg/analytics.html", {'error': str(e)})

# ============================================================================
# UTILITY FUNCTIONS - UPDATED FOR NEW RESULT STRUCTURE
# ============================================================================

def build_success_message(result):
    """Build success message from processing result - UPDATED"""
    # CHANGED: data_counts instead of save_results
    data_counts = result.get('data_counts', {})
    total_count = sum(data_counts.values()) if data_counts else result.get('total_records', 0)
    tickets_count = result.get('tickets_generated', 0)
    
    msg_parts = [f"Processing completed successfully!"]
    
    if total_count:
        msg_parts.append(f"Processed {total_count:,} records")
    
    if tickets_count:
        msg_parts.append(f"Generated {tickets_count:,} SESSION-ONLY tickets")
    
    if result.get('flexible_mapping_used'):
        msg_parts.append("Used flexible column mapping")
    
    return " • ".join(msg_parts)

def build_mongodb_success_message(result):
    """Build success message for MongoDB processing - UPDATED"""
    # CHANGED: data_counts instead of save_results
    data_counts = result.get('data_counts', {})
    total_records = sum(data_counts.values()) if data_counts else 0
    tickets_count = result.get('tickets_generated', 0)
    
    msg_parts = ["MongoDB ingestion completed successfully!"]
    
    if total_records:
        msg_parts.append(f"Analyzed {total_records:,} existing records")
    
    if tickets_count:
        msg_parts.append(f"Generated {tickets_count:,} SESSION-ONLY tickets")
    
    if result.get('target_channels'):
        channels = result['target_channels']
        msg_parts.append(f"Focused on {len(channels)} channels: {', '.join(channels[:3])}")
    
    return " • ".join(msg_parts)

def get_session_ticket_analytics():
    """Get SESSION-ONLY ticket analytics"""
    try:
        from django.db.models import Avg, Max, Min
        
        # Basic stats
        total_tickets = Ticket.objects.count()
        session_tickets = Ticket.objects.exclude(session_id__isnull=True).count()
        
        # Confidence distribution
        high_conf = Ticket.objects.filter(confidence_score__gte=0.8).count()
        med_conf = Ticket.objects.filter(confidence_score__gte=0.5, confidence_score__lt=0.8).count()
        low_conf = Ticket.objects.filter(confidence_score__lt=0.5).count()
        
        # Root cause distribution
        failure_details = Ticket.objects.exclude(failure_details__isnull=True).values_list('failure_details', flat=True)
        root_causes = {}
        for details in failure_details:
            if isinstance(details, dict) and 'root_cause' in details:
                cause = details['root_cause']
                root_causes[cause] = root_causes.get(cause, 0) + 1
        
        return {
            'total_tickets': total_tickets,
            'session_tickets': session_tickets,
            'session_percentage': (session_tickets / total_tickets * 100) if total_tickets else 0,
            'confidence_distribution': {
                'high': high_conf,
                'medium': med_conf,
                'low': low_conf
            },
            'root_causes': root_causes,
            'avg_confidence': Ticket.objects.aggregate(avg=Avg('confidence_score'))['avg'] or 0
        }
    except Exception as e:
        logger.error(f"Error calculating session ticket analytics: {e}")
        return {}

def get_processing_performance_analytics():
    """Get processing performance analytics"""
    try:
        from django.db.models import Avg, Count, Sum
        
        # Recent processing stats
        recent_logs = DataIngestionLog.objects.filter(
            started_at__gte=timezone.now() - timedelta(days=7)
        )
        
        performance = {
            'total_batches': recent_logs.count(),
            'successful_batches': recent_logs.filter(status='success').count(),
            'failed_batches': recent_logs.filter(status='failed').count(),
            'avg_processing_time': recent_logs.filter(
                processing_time_seconds__isnull=False
            ).aggregate(avg=Avg('processing_time_seconds'))['avg'] or 0,
            'total_records_processed': recent_logs.aggregate(
                total=Sum('records_processed')
            )['total'] or 0,
        }
        
        # Success rate
        if performance['total_batches'] > 0:
            performance['success_rate'] = (
                performance['successful_batches'] / performance['total_batches'] * 100
            )
        else:
            performance['success_rate'] = 0
        
        return performance
        
    except Exception as e:
        logger.error(f"Error calculating processing performance: {e}")
        return {}

def get_channel_analytics():
    """Get VARIABLE channel analytics"""
    try:
        from django.db.models import Count
        
        # Channel distribution from tickets
        ticket_channels = Ticket.objects.exclude(
            context_data__isnull=True
        ).values_list('context_data', flat=True)
        
        channel_counts = {}
        for context in ticket_channels:
            if isinstance(context, dict):
                channel = context.get('asset_name') or context.get('channel')
                if channel:
                    channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        # Session distribution by asset name
        session_channels = Session.objects.exclude(
            asset_name__isnull=True
        ).values('asset_name').annotate(count=Count('id')).order_by('-count')[:10]
        
        return {
            'ticket_channels': dict(sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'session_channels': {item['asset_name']: item['count'] for item in session_channels},
            'total_channels': len(channel_counts)
        }
        
    except Exception as e:
        logger.error(f"Error calculating channel analytics: {e}")
        return {}

def _calculate_avg_processing_time():
    """Calculate average processing time from recent logs"""
    try:
        from django.db.models import Avg
        
        recent_logs = DataIngestionLog.objects.filter(
            started_at__gte=timezone.now() - timedelta(days=7),
            processing_time_seconds__isnull=False
        )
        
        result = recent_logs.aggregate(avg_time=Avg('processing_time_seconds'))
        return round(result['avg_time'] or 0, 2)
        
    except Exception:
        return 0

def _calculate_recent_success_rate():
    """Calculate recent processing success rate"""
    try:
        recent_logs = DataIngestionLog.objects.filter(
            started_at__gte=timezone.now() - timedelta(days=7)
        )
        
        total = recent_logs.count()
        successful = recent_logs.filter(status='success').count()
        
        return round((successful / total * 100) if total > 0 else 100, 1)
        
    except Exception:
        return 0
