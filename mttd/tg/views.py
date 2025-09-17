# views.py - COMPLETE FIX WITH FIELD NAME CONSISTENCY
"""
FIXED: Complete views.py with proper field name mapping for all Django models
- Robust field mapping for Session, KPI, Advancetags models
- Proper Ticket creation with JSON field structure
- Timezone-aware datetime handling
- Complete error handling and logging
"""

import logging
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta
import json
import tempfile
import os

from .forms import UploadFilesForm, AutoUploadConfigForm
from .models import Session, KPI, Advancetags, Ticket
from .data_processing import DataProcessor
from .data_validation import DataValidator

logger = logging.getLogger(__name__)

# FIELD MAPPING DICTIONARIES FOR EXACT MODEL COMPATIBILITY
def get_kpi_field_mapping():
    """Map KPI CSV columns to Django model fields"""
    return {
        # Exact CSV column names -> Django model field names
        'Timestamp': 'timestamp',
        'Plays': 'plays',
        'Playing Time (Ended) (mins)': 'playing_time_ended_mins',
        'Streaming Performance Index': 'streaming_performance_index',
        'Video Start Failures Technical': 'video_start_failures_technical',
        'Video Start Failures Business': 'video_start_failures_business',
        'Exit Before Video Starts': 'exit_before_video_starts',
        'Video Playback Failures Technical': 'video_playback_failures_technical',
        'Video Playback Failures Business': 'video_playback_failures_business',
        'Video Start Time(sec)': 'video_start_time_sec',
        'Rebuffering Ratio(%)': 'rebuffering_ratio_pct',
        'Connection Induced Rebuffering Ratio(%)': 'connection_induced_rebuffering_ratio_pct',
        'Video Restart Time(sec)': 'video_restart_time_sec',
        'Avg. Peak Bitrate(Mbps)': 'avg_peak_bitrate_mbps',
        
        # Alternative column names that might appear
        'playing_time_mins': 'playing_time_ended_mins',
        'connection_induced_rebuffering_pct': 'connection_induced_rebuffering_ratio_pct',
        'playingtimemins': 'playing_time_ended_mins',
        'connectioninducedrebufferingpct': 'connection_induced_rebuffering_ratio_pct'
    }

def get_session_field_mapping():
    """Map Session CSV columns to Django model fields"""
    return {
        # Exact CSV column names -> Django model field names
        'Session ID': 'session_id',
        'Session Start Time': 'session_start_time',
        'Session End Time': 'session_end_time',
        'Playing Time': 'playing_time',
        'Asset Name': 'asset_name',
        'Ended Session': 'ended_session',
        'Impacted Session': 'impacted_session',
        'Video Start Time': 'video_start_time',
        'Rebuffering Ratio': 'rebuffering_ratio',
        'Connection Induced Rebuffering Ratio': 'connection_induced_rebuffering_ratio',
        'Total Video Restart Time': 'total_video_restart_time',
        'Avg. Peak Bitrate': 'avg_peak_bitrate',
        'Avg. Average Bitrate': 'avg_average_bitrate',
        'Average Framerate': 'average_framerate',
        'Starting Bitrate': 'starting_bitrate',
        'channel': 'channel',
        'Bitrate Switches': 'bitrate_switches',
        'Ended Status': 'ended_status',
        'Exit Before Video Starts': 'exit_before_video_starts',
        'Status': 'status',
        'Video Start Failure': 'video_start_failure',
        
        # Alternative names
        'sessionid': 'session_id',
        'sessionstarttime': 'session_start_time',
        'sessionendtime': 'session_end_time',
        'assetname': 'asset_name'
    }

def get_advancetags_field_mapping():
    """Map Advancetags CSV columns to Django model fields"""
    return {
        # Exact CSV column names -> Django model field names
        'Session Id': 'session_id',
        'Asset Name': 'asset_name',
        'Content Category': 'content_category',
        'Browser Name': 'browser_name',
        'Browser Version': 'browser_version',
        'Device Hardware Type': 'device_hardware_type',
        'Device Manufacturer': 'device_manufacturer',
        'Device Marketing Name': 'device_marketing_name',
        'Device Model': 'device_model',
        'Device Name': 'device_name',
        'Device Operating System': 'device_operating_system',
        'Device Operating System Family': 'device_operating_system_family',
        'Device Operating System Version': 'device_operating_system_version',
        'App Name': 'app_name',
        'App Version': 'app_version',
        'Player Framework Name': 'player_framework_name',
        'Player Framework Version': 'player_framework_version',
        'Last CDN': 'last_cdn',
        'Channel': 'channel',
        'city': 'city',
        'ip': 'ip_address',
        'ipv6': 'ipv6_address',
        'cdn': 'cdn',
        'state': 'state',
        'country': 'country',
        'address': 'address',
        'asnName': 'asn_name',
        'ispName': 'isp_name',
        'streamUrl': 'stream_url',
        
        # Alternative problematic names that were causing errors
        'deviceos': 'device_operating_system',
        'deviceosfamily': 'device_operating_system_family',
        'deviceosversion': 'device_operating_system_version',
        'asname': 'asn_name',
        'sessionid': 'session_id'
    }

def map_and_clean_model_data(data_dict: dict, model_class, field_mapping: dict) -> dict:
    """
    Map field names and clean data for Django model compatibility
    
    Args:
        data_dict: Raw data dictionary
        model_class: Django model class
        field_mapping: Dictionary mapping source -> target field names
    
    Returns:
        Cleaned dictionary with only valid model fields
    """
    # Get valid model field names
    valid_fields = {f.name for f in model_class._meta.get_fields()}
    
    # Apply field mapping
    mapped_data = {}
    for source_field, value in data_dict.items():
        # Skip None values and empty strings for optional fields
        if pd.isna(value) or (isinstance(value, str) and value.strip() == ''):
            continue
            
        # Map field name if mapping exists
        target_field = field_mapping.get(source_field, source_field)
        
        # Only include valid model fields
        if target_field in valid_fields:
            mapped_data[target_field] = value
    
    return mapped_data

def make_timezone_aware(datetime_value):
    """Convert naive datetime to timezone-aware"""
    if pd.isna(datetime_value):
        return None
    
    if isinstance(datetime_value, str):
        try:
            datetime_value = pd.to_datetime(datetime_value)
        except:
            return None
    
    if datetime_value and timezone.is_naive(datetime_value):
        return timezone.make_aware(datetime_value)
    
    return datetime_value

def create_ticket_from_engine_data(ticket_data: dict) -> dict:
    """
    Convert ticket engine output to Django Ticket model format
    
    Args:
        ticket_data: Raw ticket data from AutoTicketMVP
    
    Returns:
        Dictionary compatible with Ticket model
    """
    # Map ticket engine fields to Ticket model structure
    ticket_dict = {
        'ticket_id': ticket_data.get('ticket_id'),
        'viewer_id': ticket_data.get('viewer_id'),
        'session_id': ticket_data.get('session_id'),
        'priority': ticket_data.get('priority', 'medium'),
        'status': 'new',  # Always start as new
        'assign_team': ticket_data.get('assign_team', 'technical'),
        'issue_type': 'video_start_failure',
        'description': ticket_data.get('description', ''),
        'failure_details': {
            'root_cause': ticket_data.get('root_cause'),
            'confidence': ticket_data.get('confidence_score', 0.0),
            'evidence': ticket_data.get('evidence', []),
            'correlation_count': ticket_data.get('correlation_count', 0),
            'failure_code': ticket_data.get('failure_code'),
            'failure_type': 'video_start_failure'
        },
        'context_data': {
            'asset_name': ticket_data.get('asset_name'),
            'channel': ticket_data.get('channel'),
            'failure_time': ticket_data.get('failure_time'),
            'formatted_failure_time': ticket_data.get('formatted_failure_time'),
            'deep_link': ticket_data.get('deep_link'),
            'title': ticket_data.get('title')
        }
    }
    
    # Remove None values
    return {k: v for k, v in ticket_dict.items() if v is not None}

# DASHBOARD VIEW - FIXED created_at field issue
def dashboard_view(request):
    """Enhanced dashboard with MongoDB-optimized queries and analytics"""
    try:
        # Get counts for dashboard cards
        total_tickets = Ticket.objects.count()
        open_tickets = Ticket.objects.filter(status__in=['new', 'in_progress']).count()
        high_priority_tickets = Ticket.objects.filter(priority='high', status__in=['new', 'in_progress']).count()
        
        # FIXED: Use correct timestamp fields for each model
        recent_sessions = Session.objects.filter(created_at__gte=timezone.now() - timedelta(hours=24)).count()
        recent_kpis = KPI.objects.filter(timestamp__gte=timezone.now() - timedelta(hours=24)).count()  # Use timestamp, not created_at
        recent_metadata = Advancetags.objects.filter(created_at__gte=timezone.now() - timedelta(hours=24)).count()
        
        # Recent activity
        recent_tickets = Ticket.objects.all().order_by('-created_at')[:5]
        
        # Status distribution for charts
        status_distribution = list(Ticket.objects.values('status').annotate(count=Count('id')).order_by('status'))
        priority_distribution = list(Ticket.objects.values('priority').annotate(count=Count('id')).order_by('priority'))
        
        # Weekly ticket trend (last 7 days)
        week_ago = timezone.now() - timedelta(days=7)
        weekly_tickets = []
        for i in range(7):
            day = week_ago + timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            count = Ticket.objects.filter(created_at__gte=day_start, created_at__lt=day_end).count()
            weekly_tickets.append({
                'date': day.strftime('%m/%d'),
                'count': count
            })
        
        context = {
            'total_tickets': total_tickets,
            'open_tickets': open_tickets,
            'high_priority_tickets': high_priority_tickets,
            'recent_tickets': recent_tickets,
            'status_distribution': json.dumps(status_distribution),
            'priority_distribution': json.dumps(priority_distribution),
            'weekly_tickets': json.dumps(weekly_tickets),
            'recent_sessions': recent_sessions,
            'recent_kpis': recent_kpis,
            'recent_metadata': recent_metadata,
            'mongodb_connected': True,  # You can add actual health check here
        }
        
        return render(request, 'tg/dashboard.html', context)
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        messages.error(request, f"Dashboard loading error: {str(e)}")
        return render(request, 'tg/dashboard.html', {
            'total_tickets': 0,
            'open_tickets': 0,
            'high_priority_tickets': 0,
            'recent_tickets': [],
            'status_distribution': '[]',
            'priority_distribution': '[]',
            'weekly_tickets': '[]',
            'mongodb_connected': False,
        })

# FILE UPLOAD VIEW - COMPLETELY FIXED WITH FIELD MAPPING
@csrf_exempt
def upload_files(request):
    """FIXED: Enhanced file upload with proper field mapping and model compatibility"""
    if request.method == 'POST':
        form = UploadFilesForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                processor = DataProcessor()
                file_mapping = form.get_file_mapping()
                
                df_sessions, df_kpis, df_metadata = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                
                for file_obj, data_types in file_mapping.items():
                    filename = getattr(file_obj, 'name', 'uploaded_file')
                    logger.info(f"Processing {filename} with data types: {data_types}")
                    
                    try:
                        # Process file with proper temp file handling
                        temp_file_path = None
                        
                        if hasattr(file_obj, 'temporary_file_path'):
                            temp_file_path = file_obj.temporary_file_path()
                        elif hasattr(file_obj, 'read'):
                            temp_fd, temp_file_path = tempfile.mkstemp(suffix=os.path.splitext(filename)[1])
                            try:
                                with os.fdopen(temp_fd, 'wb') as temp_file:
                                    for chunk in file_obj.chunks():
                                        temp_file.write(chunk)
                            except:
                                os.close(temp_fd)
                                raise
                        else:
                            temp_file_path = str(file_obj)
                        
                        if not temp_file_path or not os.path.exists(temp_file_path):
                            raise ValueError(f"Could not access file: {filename}")
                        
                        file_size = os.path.getsize(temp_file_path)
                        if file_size == 0:
                            raise ValueError(f"File {filename} is empty (0 bytes)")
                        
                        logger.info(f"Processing file {filename}, size: {file_size} bytes, path: {temp_file_path}")
                        
                        # Process the file
                        file_results = processor.intelligently_process_any_file(temp_file_path, filename)
                        
                        # Clean up temporary file if we created it
                        if hasattr(file_obj, 'read') and not hasattr(file_obj, 'temporary_file_path'):
                            try:
                                os.unlink(temp_file_path)
                            except:
                                pass
                        
                        # Combine data based on selected types
                        for data_type in data_types:
                            if data_type == 'sessions' and 'sessions' in file_results and not file_results['sessions'].empty:
                                df_sessions = pd.concat([df_sessions, file_results['sessions']], ignore_index=True)
                                logger.info(f"Added {len(file_results['sessions'])} session records from {filename}")
                            elif data_type == 'kpi_data' and 'kpi_data' in file_results and not file_results['kpi_data'].empty:
                                df_kpis = pd.concat([df_kpis, file_results['kpi_data']], ignore_index=True)
                                logger.info(f"Added {len(file_results['kpi_data'])} KPI records from {filename}")
                            elif data_type == 'advancetags' and 'advancetags' in file_results and not file_results['advancetags'].empty:
                                df_metadata = pd.concat([df_metadata, file_results['advancetags']], ignore_index=True)
                                logger.info(f"Added {len(file_results['advancetags'])} advancetags records from {filename}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
                        messages.error(request, f"Error processing {filename}: {str(e)}")
                        continue
                
                total_records = len(df_sessions) + len(df_kpis) + len(df_metadata)
                if total_records == 0:
                    messages.error(request, "No valid data found in uploaded files. Please check file format and content.")
                    return render(request, 'tg/upload.html', {'form': form})
                
                # FIXED: Save data with proper field mapping
                sessions_saved = 0
                kpis_saved = 0
                metadata_saved = 0
                
                # Get field mappings
                session_mapping = get_session_field_mapping()
                kpi_mapping = get_kpi_field_mapping()
                advancetags_mapping = get_advancetags_field_mapping()
                
                # Save sessions with field mapping
                if not df_sessions.empty:
                    for _, row in df_sessions.iterrows():
                        try:
                            row_dict = row.to_dict()
                            # Apply field mapping and clean data
                            clean_data = map_and_clean_model_data(row_dict, Session, session_mapping)
                            
                            # Handle timezone-aware datetime fields
                            for field in ['session_start_time', 'session_end_time']:
                                if field in clean_data:
                                    clean_data[field] = make_timezone_aware(clean_data[field])
                            
                            session_id = clean_data.get('session_id')
                            if session_id:
                                Session.objects.update_or_create(
                                    session_id=session_id,
                                    defaults=clean_data
                                )
                                sessions_saved += 1
                                
                        except Exception as e:
                            logger.error(f"Error saving session record: {e}")
                            continue
                
                # Save KPIs with field mapping
                if not df_kpis.empty:
                    for _, row in df_kpis.iterrows():
                        try:
                            row_dict = row.to_dict()
                            # Apply field mapping and clean data
                            clean_data = map_and_clean_model_data(row_dict, KPI, kpi_mapping)
                            
                            # Handle timezone-aware datetime fields
                            if 'timestamp' in clean_data:
                                clean_data['timestamp'] = make_timezone_aware(clean_data['timestamp'])
                            
                            if clean_data.get('timestamp'):
                                KPI.objects.create(**clean_data)
                                kpis_saved += 1
                                
                        except Exception as e:
                            logger.error(f"Error saving KPI record: {e}")
                            continue
                
                # Save metadata/advancetags with field mapping
                if not df_metadata.empty:
                    for _, row in df_metadata.iterrows():
                        try:
                            row_dict = row.to_dict()
                            # Apply field mapping and clean data
                            clean_data = map_and_clean_model_data(row_dict, Advancetags, advancetags_mapping)
                            
                            session_id = clean_data.get('session_id')
                            if session_id:
                                Advancetags.objects.create(**clean_data)
                                metadata_saved += 1
                                
                        except Exception as e:
                            logger.error(f"Error saving metadata record: {e}")
                            continue
                
                # FIXED: Generate tickets with proper model structure
                tickets_generated = 0
                try:
                    if not df_sessions.empty:
                        try:
                            from .operation.ticket_engine import AutoTicketMVP
                        except ImportError:
                            try:
                                from .ticket_engine import AutoTicketMVP
                            except ImportError:
                                raise ImportError("AutoTicketMVP engine not found. Please check ticket_engine.py location.")

                        engine = AutoTicketMVP(df_sessions, df_kpi=df_kpis, df_advancetags=df_metadata)
                        tickets = engine.process()
                        
                        for ticket_data in tickets:
                            try:
                                # FIXED: Create ticket with proper model structure
                                ticket_dict = create_ticket_from_engine_data(ticket_data)
                                
                                ticket_obj, created = Ticket.objects.update_or_create(
                                    ticket_id=ticket_dict.get('ticket_id'),
                                    defaults=ticket_dict
                                )
                                if created:
                                    tickets_generated += 1
                                    
                            except Exception as e:
                                logger.error(f"Error saving ticket: {e}")
                                continue
                                
                except Exception as e:
                    logger.error(f"Ticket generation failed: {e}")
                    messages.warning(request, f"Ticket generation partially failed: {str(e)}")
                
                success_msg = f"File upload completed successfully!\n"
                success_msg += f"Saved {sessions_saved} sessions, {kpis_saved} KPIs, {metadata_saved} metadata records to MongoDB.\n"
                success_msg += f"Generated {tickets_generated} tickets with status 'new' and team assignments."
                messages.success(request, success_msg)
                
                return redirect('tg:ticket_list')
                
            except Exception as e:
                logger.error(f"Upload processing failed: {e}")
                messages.error(request, f"Upload failed: {str(e)}")
        else:
            # Form validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = UploadFilesForm()
    
    context = {
        'form': form,
        'mongodb_stats': {
            'sessions': Session.objects.count(),
            'kpis': KPI.objects.count(),
            'metadata': Advancetags.objects.count(),
            'tickets': Ticket.objects.count(),
        }
    }
    
    return render(request, 'tg/upload.html', context)

# MONGODB INGESTION VIEW - FIXED WITH FIELD MAPPING
@csrf_exempt
def mongodb_ingestion_view(request):
    """FIXED: Process data directly from MongoDB collections with proper field mapping"""
    if request.method == 'POST':
        try:
            processor = DataProcessor()
            
            # Fetch data from MongoDB models (already properly structured)
            sessions_df = processor.fetch_database_df(Session)
            kpi_df = processor.fetch_database_df(KPI)
            metadata_df = processor.fetch_database_df(Advancetags)

            logger.info(f"MongoDB ingestion: {len(sessions_df)} sessions, {len(kpi_df)} KPIs, {len(metadata_df)} metadata")

            tickets_generated = 0
            if not sessions_df.empty:
                try:
                    from .operation.ticket_engine import AutoTicketMVP
                    engine = AutoTicketMVP(
                                sessions_df, 
                                df_kpi=kpi_df,
                                df_advancetags=metadata_df  # Correct parameter name
                    )
                    tickets = engine.process()
                    
                    for ticket_data in tickets:
                        try:
                            # FIXED: Create ticket with proper model structure
                            ticket_dict = create_ticket_from_engine_data(ticket_data)
                            
                            ticket_obj, created = Ticket.objects.update_or_create(
                                ticket_id=ticket_dict.get('ticket_id'),
                                defaults=ticket_dict
                            )
                            if created:
                                tickets_generated += 1
                                
                        except Exception as e:
                            logger.error(f"Error saving MongoDB-generated ticket: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"MongoDB ticket generation failed: {e}")
                    messages.error(request, f"Ticket generation failed: {str(e)}")
            
            if tickets_generated > 0:
                messages.success(request, f"MongoDB ingestion completed! Generated {tickets_generated} new tickets from existing data.")
            else:
                messages.warning(request, "MongoDB ingestion completed, but no new tickets were generated. Data may already be processed.")
                
            return redirect('tg:dashboard')
            
        except Exception as e:
            logger.error(f"MongoDB ingestion failed: {e}")
            messages.error(request, f"MongoDB ingestion failed: {str(e)}")
    
    # GET request - show MongoDB ingestion options
    context = {
        'session_count': Session.objects.count(),
        'kpi_count': KPI.objects.count(),
        'metadata_count': Advancetags.objects.count(),
        'mongodb_form': AutoUploadConfigForm(),
    }
    
    return render(request, 'tg/mongodb_ingestion.html', context)

# OTHER VIEWS (unchanged but with proper error handling)
def analytics_view(request):
    """Analytics dashboard view"""
    try:
        total_tickets = Ticket.objects.count()
        
        # Team performance
        team_stats = list(Ticket.objects.values('assign_team').annotate(
            total=Count('id'),
            resolved=Count('id', filter=Q(status='resolved')),
            high_priority=Count('id', filter=Q(priority='high'))
        ).order_by('-total'))
        
        # Status trends
        status_trends = []
        for i in range(30):
            day = timezone.now() - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            day_stats = {
                'date': day.strftime('%Y-%m-%d'),
                'new': Ticket.objects.filter(created_at__gte=day_start, created_at__lt=day_end, status='new').count(),
                'resolved': Ticket.objects.filter(created_at__gte=day_start, created_at__lt=day_end, status='resolved').count(),
            }
            status_trends.append(day_stats)
        
        context = {
            'total_tickets': total_tickets,
            'team_stats': team_stats,
            'status_trends': json.dumps(status_trends),
        }
        
        return render(request, 'tg/analytics.html', context)
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        messages.error(request, f"Analytics loading error: {str(e)}")
        return render(request, 'tg/analytics.html', {'total_tickets': 0, 'team_stats': []})

def ticket_list(request):
    """Enhanced ticket list with filtering"""
    try:
        tickets_queryset = Ticket.objects.all().order_by('-created_at')
        
        # Apply filters
        status_filter = request.GET.get('status')
        priority_filter = request.GET.get('priority')
        team_filter = request.GET.get('team')
        search_query = request.GET.get('search')
        
        if status_filter:
            tickets_queryset = tickets_queryset.filter(status=status_filter)
        if priority_filter:
            tickets_queryset = tickets_queryset.filter(priority=priority_filter)
        if team_filter:
            tickets_queryset = tickets_queryset.filter(assign_team=team_filter)
        if search_query:
            tickets_queryset = tickets_queryset.filter(
                Q(ticket_id__icontains=search_query) |
                Q(session_id__icontains=search_query) |
                Q(viewer_id__icontains=search_query)
            )
        
        # Get distinct values for filter dropdowns
        status_choices = list(Ticket.objects.values_list('status', flat=True).distinct())
        priority_choices = list(Ticket.objects.values_list('priority', flat=True).distinct())
        team_choices = list(Ticket.objects.values_list('assign_team', flat=True).distinct().exclude(assign_team__isnull=True))
        
        context = {
            'tickets': tickets_queryset[:100],  # Limit for performance
            'status_choices': status_choices,
            'priority_choices': priority_choices,
            'team_choices': team_choices,
            'current_status': status_filter,
            'current_priority': priority_filter,
            'current_team': team_filter,
            'current_search': search_query or '',
        }
        
        return render(request, 'tg/ticket_list.html', context)
        
    except Exception as e:
        logger.error(f"Ticket list error: {e}")
        messages.error(request, f"Error loading tickets: {str(e)}")
        return render(request, 'tg/ticket_list.html', {'tickets': []})

def ticket_detail(request, ticket_id):
    """Enhanced ticket detail with MongoDB data context"""
    try:
        ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
        
        # Get related session data if available
        related_session = None
        if ticket.session_id:
            try:
                related_session = Session.objects.filter(session_id=ticket.session_id).first()
            except:
                pass
        
        # Get related metadata
        related_metadata = []
        if ticket.session_id:
            try:
                related_metadata = Advancetags.objects.filter(session_id=ticket.session_id)[:5]
            except:
                pass
        
        context = {
            'ticket': ticket,
            'related_session': related_session,
            'related_metadata': related_metadata,
            'mongodb_linked': related_session is not None or related_metadata,
        }
        
        return render(request, 'tg/ticket_detail.html', context)
        
    except Exception as e:
        logger.error(f"Ticket detail error: {e}")
        messages.error(request, f"Error loading ticket details: {str(e)}")
        return redirect('tg:ticket_list')

@csrf_exempt
def update_ticket_status(request, ticket_id):
    """Enhanced ticket update with status and assignment management"""
    try:
        ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
        
        if request.method == 'POST':
            old_status = ticket.status
            old_priority = ticket.priority
            old_team = ticket.assign_team
            
            new_status = request.POST.get('status')
            new_priority = request.POST.get('priority')
            new_team = request.POST.get('assign_team')
            resolution_notes = request.POST.get('resolution_notes', '')
            
            # Update fields
            if new_status and new_status != old_status:
                ticket.status = new_status
            if new_priority and new_priority != old_priority:
                ticket.priority = new_priority
            if new_team and new_team != old_team:
                ticket.assign_team = new_team
            if resolution_notes:
                ticket.resolution_notes = resolution_notes
            
            # Set resolved timestamp if status is resolved
            if new_status in ['resolved', 'closed']:
                ticket.resolved_at = timezone.now()
            
            ticket.save()
            
            # Track changes
            changes = []
            if new_status != old_status:
                changes.append(f"Status: {old_status} → {new_status}")
            if new_priority != old_priority:
                changes.append(f"Priority: {old_priority} → {new_priority}")
            if new_team != old_team:
                changes.append(f"Team: {old_team or 'None'} → {new_team}")
            
            change_msg = "Ticket updated successfully"
            if changes:
                change_msg += f": {', '.join(changes)}"
            
            messages.success(request, change_msg)
            return redirect('tg:ticket_detail', ticket_id=ticket.ticket_id)
        
        # GET request - show update form
        context = {
            'ticket': ticket,
            'status_choices': ['new', 'in_progress', 'resolved', 'closed'],
            'priority_choices': ['low', 'medium', 'high', 'critical'],
            'team_choices': ['technical', 'network', 'content', 'customer_service'],
        }
        
        return render(request, 'tg/update_ticket.html', context)
        
    except Exception as e:
        logger.error(f"Ticket update error: {e}")
        messages.error(request, f"Error updating ticket: {str(e)}")
        return redirect('tg:ticket_list')

def analytics_data_api(request):
    """API endpoint for analytics data"""
    try:
        data = {
            'tickets': {
                'total': Ticket.objects.count(),
                'by_status': list(Ticket.objects.values('status').annotate(count=Count('id'))),
                'by_priority': list(Ticket.objects.values('priority').annotate(count=Count('id'))),
                'by_team': list(Ticket.objects.values('assign_team').annotate(count=Count('id'))),
            }
        }
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def api_data_ingestion_status(request):
    """API endpoint to check data ingestion status"""
    try:
        stats = {
            'mongodb': {
                'sessions': Session.objects.count(),
                'kpis': KPI.objects.count(),
                'metadata': Advancetags.objects.count(),
                'tickets': Ticket.objects.count(),
                'last_session': Session.objects.order_by('-created_at').first().created_at.isoformat() if Session.objects.exists() else None,
                'connected': True,
            },
            'tickets': {
                'new': Ticket.objects.filter(status='new').count(),
                'in_progress': Ticket.objects.filter(status='in_progress').count(),
                'resolved': Ticket.objects.filter(status='resolved').count(),
                'closed': Ticket.objects.filter(status='closed').count(),
            }
        }
        return JsonResponse(stats)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def tg_home_redirect(request):
    """Redirect home to dashboard"""
    return redirect('tg:dashboard')
