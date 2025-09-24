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

def dashboard_view(request):
    """MAINTAIN existing function - Enhanced dashboard with unified pipeline stats"""
    try:
        # Use existing utility function to get comprehensive stats
        stats = UnifiedDataManager.get_data_statistics()
        
        # Get recent processing activity
        recent_ingestions = DataIngestionLog.objects.order_by('-started_at')[:10]
        
        # Enhanced ticket statistics with session-only focus
        ticket_stats = Ticket.objects.aggregate(
            total=Count('id'),
            new=Count('id', filter=Q(status='new')),
            in_progress=Count('id', filter=Q(status='in_progress')),
            resolved=Count('id', filter=Q(status='resolved')),
            high_priority=Count('id', filter=Q(priority='high')),
            critical=Count('id', filter=Q(priority='critical')),
            session_based=Count('id', filter=Q(session_id__isnull=False)),  # SESSION-ONLY tracking
            high_confidence=Count('id', filter=Q(confidence_score__gte=0.8))
        )
        
        # Processing performance metrics
        processing_metrics = {
            'total_batches': DataIngestionLog.objects.count(),
            'successful_batches': DataIngestionLog.objects.filter(status='success').count(),
            'failed_batches': DataIngestionLog.objects.filter(status='failed').count(),
            'avg_processing_time': _calculate_avg_processing_time(),
            'recent_success_rate': _calculate_recent_success_rate()
        }
        
        context = {
            'stats': stats,
            'ticket_stats': ticket_stats,
            'processing_metrics': processing_metrics,
            'recent_ingestions': recent_ingestions,
            'can_ingest_mongodb': True,
            'can_upload_files': True,
            'session_only_mode': True,  # Indicate SESSION-ONLY mode
            'flexible_processing': True,  # Indicate flexible processing available
        }
        
        return render(request, "tg/dashboard.html", context)
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        messages.error(request, f"Dashboard loading error: {str(e)}")
        
        # Fallback context
        context = {
            'stats': {
                'sessions': {'total': 0}, 
                'tickets': {'total': 0},
                'kpi': {'total': 0},
                'advancetags': {'total': 0}
            },
            'recent_ingestions': [],
            'error': str(e)
        }
        return render(request, "tg/dashboard.html", context)

# ============================================================================
# ENHANCED FILE UPLOAD PROCESSING - MAINTAIN EXISTING FUNCTION NAME
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
    """MAINTAIN existing function name - Enhanced with flexible pipeline"""
    if request.method == "POST":
        start_time = time.time()
        
        try:
            logger.info("Starting flexible MongoDB ingestion processing")
            
            # Get optional target channels from request (VARIABLE channels)
            target_channels_str = request.POST.get('target_channels', '')
            target_channels = []
            if target_channels_str:
                target_channels = [ch.strip() for ch in target_channels_str.split(',') if ch.strip()]
            
            logger.info(f"Target channels for MongoDB ingestion: {target_channels or 'All channels'}")
            
            # Process using enhanced MongoDB pipeline with SESSION-ONLY tickets
            result = process_mongodb_flexible_enhanced(target_channels)
            
            # Handle result - MAINTAIN existing structure
            if result.get('success', False):
                success_msg = build_mongodb_success_message(result)
                messages.success(request, success_msg)
                logger.info("Flexible MongoDB ingestion successful")
            else:
                error_msg = f"MongoDB ingestion failed: {'; '.join(result.get('errors', ['Unknown error']))}"
                messages.error(request, error_msg)
                logger.error("Flexible MongoDB ingestion failed")
            
            return redirect("tg:dashboard")
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"MongoDB ingestion failed: {str(e)} (after {processing_time:.1f}s)"
            messages.error(request, error_msg)
            logger.error(f"MongoDB ingestion exception: {e}", exc_info=True)
    
    # Get current data counts for display using existing function
    try:
        stats = UnifiedDataManager.get_data_statistics()
        
        context = {
            'stats': stats,
            'recent_ingestions': DataIngestionLog.objects.filter(
                source_type='mongodb'
            ).order_by('-started_at')[:8],
            'mongodb_features': [
                'SESSION-ID ONLY ticket generation',
                'Variable channel filtering',
                'Flexible data processing',
                'MVP diagnosis rules',
                'Batch processing tracking'
            ]
        }
    except Exception as e:
        logger.error(f"Error loading MongoDB ingestion context: {e}")
        context = {'error': str(e)}
    
    return render(request, "tg/mongodb_ingestion.html", context)

def process_mongodb_flexible_enhanced(target_channels=None):
    """
    Enhanced MongoDB processing with flexible pipeline
    Uses imported functions to prevent duplicacy
    """
    try:
        # Use existing flexible MongoDB processing
        result = process_mongodb_flexible(target_channels)
        
        # Add enhanced information
        if result.get('success'):
            result['session_only_processing'] = True
            result['flexible_processing_used'] = True
            
        return result
        
    except Exception as e:
        logger.error(f"Enhanced MongoDB processing failed: {e}")
        return {
            'success': False,
            'errors': [str(e)]
        }

# ============================================================================
# TICKET MANAGEMENT - MAINTAIN EXISTING FUNCTION NAMES
# ============================================================================

def ticket_list(request):
    """MAINTAIN existing function name - Enhanced with session-only filtering"""
    try:
        # Get filter parameters
        status_filter = request.GET.get('status', 'all')
        priority_filter = request.GET.get('priority', 'all')
        search_query = request.GET.get('search', '').strip()
        confidence_filter = request.GET.get('confidence', 'all')  # New filter
        
        # Build queryset
        tickets = Ticket.objects.all()
        
        # Apply filters
        if status_filter != 'all':
            tickets = tickets.filter(status=status_filter)
        if priority_filter != 'all':
            tickets = tickets.filter(priority=priority_filter)
        if confidence_filter == 'high':
            tickets = tickets.filter(confidence_score__gte=0.8)
        elif confidence_filter == 'low':
            tickets = tickets.filter(confidence_score__lt=0.5)
        
        if search_query:
            tickets = tickets.filter(
                Q(ticket_id__icontains=search_query) |
                Q(title__icontains=search_query) |
                Q(session_id__icontains=search_query)  # SESSION-ID ONLY search
            )
        
        # Order by priority, confidence, and creation time
        tickets = tickets.order_by('-priority', '-confidence_score', '-created_at')
        
        # Pagination
        paginator = Paginator(tickets, 25)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        # SESSION-ONLY statistics
        session_stats = {
            'total_with_sessions': tickets.exclude(session_id__isnull=True).count(),
            'high_confidence': tickets.filter(confidence_score__gte=0.8).count(),
            'pending_review': tickets.filter(status='new').count(),
        }
        
        context = {
            'tickets': page_obj,
            'current_filters': {
                'status': status_filter,
                'priority': priority_filter,
                'confidence': confidence_filter,
                'search': search_query,
            },
            'total_count': paginator.count,
            'session_stats': session_stats,
            'session_only_mode': True  # Indicate SESSION-ONLY mode
        }
        
        return render(request, "tg/ticket_list.html", context)
        
    except Exception as e:
        logger.error(f"Ticket list error: {e}", exc_info=True)
        messages.error(request, f"Error loading tickets: {str(e)}")
        return render(request, "tg/ticket_list.html", {'tickets': []})

def ticket_detail(request, ticket_id):
    """MAINTAIN existing function name - Enhanced with session-only details"""
    try:
        ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
        
        # Get related session data if available using SESSION-ID ONLY
        related_session = None
        if ticket.session_id:
            try:
                related_session = Session.objects.get(session_id=ticket.session_id)
            except Session.DoesNotExist:
                logger.warning(f"Session {ticket.session_id} not found for ticket {ticket_id}")
        
        # Get related metadata if available
        related_metadata = None
        if ticket.session_id:
            try:
                related_metadata = Advancetags.objects.get(session_id=ticket.session_id)
            except Advancetags.DoesNotExist:
                pass
        
        # Extract failure details
        failure_details = ticket.failure_details or {}
        context_data = ticket.context_data or {}
        suggested_actions = ticket.suggested_actions or []
        
        context = {
            'ticket': ticket,
            'related_session': related_session,
            'related_metadata': related_metadata,
            'failure_details': failure_details,
            'context_data': context_data,
            'suggested_actions': suggested_actions,
            'can_edit': True,
            'session_only_mode': True,  # SESSION-ONLY mode
            'mvp_diagnosis': True if failure_details.get('root_cause') else False
        }
        
        return render(request, "tg/ticket_detail.html", context)
        
    except Exception as e:
        logger.error(f"Ticket detail error: {e}", exc_info=True)
        messages.error(request, f"Error loading ticket: {str(e)}")
        return redirect("tg:ticket_list")

@csrf_exempt
def update_ticket_status(request, ticket_id):
    """MAINTAIN existing function name - Enhanced status update with session tracking"""
    if request.method == "POST":
        try:
            ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
            new_status = request.POST.get("status")
            resolution_notes = request.POST.get("resolution_notes", "")
            
            # Validate status
            valid_statuses = [choice[0] for choice in Ticket.STATUS_CHOICES]
            if new_status not in valid_statuses:
                return JsonResponse({
                    "success": False, 
                    "error": f"Invalid status: {new_status}"
                })
            
            # Update ticket using existing model method
            old_status = ticket.status
            ticket.status = new_status
            
            if resolution_notes:
                ticket.resolution_notes = resolution_notes
            
            # Set resolved timestamp
            if new_status == 'resolved' and old_status != 'resolved':
                ticket.resolved_at = timezone.now()
            elif new_status != 'resolved':
                ticket.resolved_at = None
            
            ticket.save()  # Uses existing save method
            
            logger.info(f"Ticket {ticket_id} status updated: {old_status} -> {new_status} (SESSION-ONLY mode)")
            
            return JsonResponse({
                "success": True,
                "message": f"Ticket status updated to {new_status}",
                "new_status": new_status,
                "old_status": old_status,
                "session_id": ticket.session_id,  # Include SESSION-ID in response
                "resolved_at": str(ticket.resolved_at) if ticket.resolved_at else None
            })
            
        except Exception as e:
            logger.error(f"Ticket status update error: {e}", exc_info=True)
            return JsonResponse({
                "success": False,
                "error": str(e)
            })
    
    return JsonResponse({"success": False, "error": "Invalid request method"})

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
