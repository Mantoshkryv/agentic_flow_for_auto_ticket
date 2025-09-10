# tg/views.py
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from .models import Ticket, Session, KPI, Advancetags
from .forms import UploadFilesForm
import pandas as pd
import csv
import io
import json
from datetime import datetime
from django.core.paginator import Paginator
from django.db.models import Q
from django.utils import timezone
from bson import ObjectId
from pymongo import MongoClient
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# MongoDB connection
def get_mongo_client():
    """Get MongoDB client connection"""
    try:
        # You can adjust this connection string as per your MongoDB setup
        client = MongoClient(getattr(settings, 'MONGODB_URI', 'mongodb://localhost:27017/'))
        return client[getattr(settings, 'MONGODB_DATABASE', 'ticket_system')]
    except Exception as e:
        logger.error(f"MongoDB connection error: {e}")
        return None

def serialize_ticket(ticket_dict):
    """Convert ticket dict into JSON serializable format (convert datetime and ObjectId)."""
    ticket_copy = ticket_dict.copy()
    
    # Handle ObjectId
    if isinstance(ticket_copy.get("_id"), ObjectId):
        ticket_copy["id"] = str(ticket_copy["_id"])
        ticket_copy.pop("_id", None)
    
    # Handle datetime objects
    for field in ["created_at", "updated_at", "resolved_at", "failure_time"]:
        if isinstance(ticket_copy.get(field), datetime):
            ticket_copy[field] = ticket_copy[field].isoformat()
    
    return ticket_copy

def serialize_ticket_model(ticket):
    """Convert Ticket model instance to JSON serializable dict."""
    return {
        'id': str(ticket.pk),
        'ticket_id': ticket.ticket_id,
        'viewer_id': ticket.viewer_id,
        'channel': ticket.channel,
        'asset_name': ticket.asset_name,
        'failure_code': ticket.failure_code,
        'root_cause': ticket.root_cause,
        'confidence_score': ticket.confidence_score,
        'status': ticket.status,
        'priority': ticket.priority,
        'assign_team': ticket.assign_team,
        'assigned_to': ticket.assigned_to,
        'created_at': ticket.created_at.isoformat() if ticket.created_at else None,
        'updated_at': ticket.updated_at.isoformat() if ticket.updated_at else None,
        'internal_notes': ticket.resolution_notes or '',
        'status_history': getattr(ticket, 'status_history', [])
    }

def dashboard_view(request):
    """Original dashboard view for ticket generation."""
    error = None
    message = None
    tickets = []
    df_info = None
    tickets_to_save = []

    if request.method == "POST":
        try:
            # Generate tickets from MongoDB
            if "use_mongo" in request.POST:
                # Fetch data using Django ORM instead of mongo service
                sessions_qs = Session.objects.all()
                kpi_qs = KPI.objects.all()
                meta_qs = Advancetags.objects.all()

                # Convert to DataFrames
                df_sessions = pd.DataFrame(list(sessions_qs.values()))
                df_kpi = pd.DataFrame(list(kpi_qs.values()))
                df_meta = pd.DataFrame(list(meta_qs.values()))

                df_info = {
                    "sessions": len(df_sessions),
                    "kpi": len(df_kpi),
                    "meta": len(df_meta),
                    "tickets_generated": 0,
                    "tickets_saved": 0,
                }

                if df_sessions.empty or df_kpi.empty:
                    error = "Session or KPI data is empty in database."
                else:
                    from .operation.ticket_engine import AutoTicketMVP
                    engine = AutoTicketMVP(df_sessions, df_kpi)
                    tickets = engine.process()
                    df_info["tickets_generated"] = len(tickets)

                    # Parse tickets and prepare for saving
                    for t in tickets:
                        try:
                            # Extract information from ticket text
                            session_id = t.split("Viewer ID: ")[1].split("\n")[0] if "Viewer ID: " in t else None
                            asset_name = t.split("Impacted Channel: ")[1].split("\n")[0] if "Impacted Channel: " in t else None
                            root_cause = t.split("Auto-Diagnosis: ")[1].split(" (Confidence:")[0] if "Auto-Diagnosis: " in t else None
                            confidence_str = t.split("(Confidence: ")[1].split(")")[0] if "(Confidence: " in t else "0.7"
                            assign_team = t.split("Assign to: ")[1].split(" Team")[0] if "Assign to: " in t else "NOC"
                            
                            tickets_to_save.append({
                                "viewer_id": session_id,
                                "asset_name": asset_name,
                                "root_cause": root_cause,
                                "confidence_score": float(confidence_str),
                                "ticket_text": t,
                                "assign_team": assign_team,
                                "created_at": timezone.now(),
                                "status": "new",
                                "priority": "medium",
                                "failure_code": "VSF",
                                "channel": asset_name,
                            })
                        except Exception as e:
                            logger.warning(f"Error parsing ticket: {e}")
                            tickets_to_save.append({
                                "ticket_text": t,
                                "created_at": timezone.now(),
                                "status": "new",
                                "priority": "medium",
                                "assign_team": "NOC",
                                "failure_code": "VSF",
                            })

            # Generate tickets from uploaded files
            elif "use_upload" in request.POST:
                form = UploadFilesForm(request.POST, request.FILES)
                if form.is_valid():
                    session_file = form.cleaned_data["session_file"]
                    kpi_file = form.cleaned_data["kpi_file"]
                    meta_file = form.cleaned_data["meta_file"]

                    df_sessions = pd.read_csv(session_file) if session_file.name.endswith(".csv") else pd.read_excel(session_file)
                    df_kpi = pd.read_csv(kpi_file) if kpi_file.name.endswith(".csv") else pd.read_excel(kpi_file)
                    df_meta = pd.read_csv(meta_file) if meta_file.name.endswith(".csv") else pd.read_excel(meta_file)

                    df_info = {
                        "sessions": len(df_sessions),
                        "kpi": len(df_kpi),
                        "meta": len(df_meta),
                        "tickets_generated": 0,
                        "tickets_saved": 0,
                    }

                    from .operation.ticket_engine import AutoTicketMVP
                    engine = AutoTicketMVP(df_sessions, df_kpi)
                    tickets = engine.process()
                    df_info["tickets_generated"] = len(tickets)

                    for t in tickets:
                        tickets_to_save.append({
                            "ticket_text": t,
                            "created_at": timezone.now(),
                            "status": "new",
                            "priority": "medium",
                            "assign_team": "NOC",
                            "failure_code": "VSF",
                        })
                else:
                    error = "Invalid file upload."

            # Save tickets to MongoDB
            elif "save" in request.POST:
                if "tickets_to_save" in request.session:
                    tickets_data = request.session["tickets_to_save"]
                    saved_count = 0
                    
                    # Get MongoDB connection
                    db = get_mongo_client()
                    if db is None:
                        error = "MongoDB connection failed. Saving to Django ORM instead."
                        # Fallback to Django ORM
                        for ticket_data in tickets_data:
                            try:
                                ticket = Ticket.objects.create(
                                    viewer_id=ticket_data.get('viewer_id'),
                                    channel=ticket_data.get('channel'),
                                    asset_name=ticket_data.get('asset_name'),
                                    root_cause=ticket_data.get('root_cause'),
                                    confidence_score=ticket_data.get('confidence_score'),
                                    ticket_text=ticket_data.get('ticket_text'),
                                    status=ticket_data.get('status', 'new'),
                                    priority=ticket_data.get('priority', 'medium'),
                                    assign_team=ticket_data.get('assign_team', 'NOC'),
                                    failure_code=ticket_data.get('failure_code', 'VSF'),
                                )
                                saved_count += 1
                            except Exception as e:
                                logger.error(f"Error saving ticket to Django: {e}")
                    else:
                        # Save to MongoDB
                        tickets_collection = db.tickets
                        for ticket_data in tickets_data:
                            try:
                                # Generate ticket ID if not present
                                if 'ticket_id' not in ticket_data:
                                    ticket_data['ticket_id'] = f"TICKET-{timezone.now().strftime('%Y%m%d')}-{ObjectId()}"
                                
                                # Ensure datetime objects
                                if isinstance(ticket_data.get('created_at'), str):
                                    ticket_data['created_at'] = datetime.fromisoformat(ticket_data['created_at'].replace('Z', '+00:00'))
                                
                                # Initialize status history
                                ticket_data['status_history'] = [{
                                    'timestamp': ticket_data['created_at'],
                                    'status': ticket_data.get('status', 'new'),
                                    'updated_by': 'system',
                                    'notes': 'Ticket created automatically'
                                }]
                                
                                result = tickets_collection.insert_one(ticket_data)
                                if result.inserted_id:
                                    saved_count += 1
                            except Exception as e:
                                logger.error(f"Error saving ticket to MongoDB: {e}")
                    
                    message = f"✅ {saved_count} tickets saved successfully."
                    request.session.pop("tickets_to_save", None)
                    df_info = {"tickets_saved": saved_count}
                else:
                    error = "No tickets available to save."

            # Download tickets as CSV
            elif "download" in request.POST:
                if "tickets_to_save" in request.session:
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow([
                        "Viewer ID", "Asset Name", "Root Cause", "Confidence Score", 
                        "Ticket Text", "Created At", "Status", "Assigned Team", "Priority"
                    ])
                    for t in request.session["tickets_to_save"]:
                        writer.writerow([
                            t.get("viewer_id", ""),
                            t.get("asset_name", ""),
                            t.get("root_cause", ""),
                            t.get("confidence_score", ""),
                            t.get("ticket_text", ""),
                            t.get("created_at", ""),
                            t.get("status", ""),
                            t.get("assign_team", ""),
                            t.get("priority", ""),
                        ])
                    response = HttpResponse(output.getvalue(), content_type="text/csv")
                    response["Content-Disposition"] = f"attachment; filename=tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    return response
                else:
                    error = "No tickets available to download."

            # Keep tickets in session
            if tickets_to_save:
                request.session["tickets_to_save"] = [serialize_ticket(t) for t in tickets_to_save]

        except Exception as e:
            error = f"Error: {str(e)}"
            logger.error(f"Dashboard error: {str(e)}")

    # Show recent tickets (latest 10)
    recent_tickets = Ticket.objects.all().order_by('-created_at')[:10]

    return render(request, "tg/dashboard.html", {
        "error": error,
        "message": message,
        "tickets": tickets,
        "df_info": df_info,
        "recent_tickets": recent_tickets,
        "form": UploadFilesForm(),
    })

@require_http_methods(["GET"])
def tickets_api_view(request):
    """API endpoint to fetch all tickets for management page."""
    try:
        # Try MongoDB first
        db = get_mongo_client()
        if db is not None:
            tickets_collection = db.tickets
            
            # Build query
            query = {}
            
            # Apply filters if provided
            status_filter = request.GET.get('status')
            if status_filter and status_filter != 'all':
                query['status'] = status_filter
            
            search_query = request.GET.get('search')
            if search_query:
                query['$or'] = [
                    {'ticket_id': {'$regex': search_query, '$options': 'i'}},
                    {'viewer_id': {'$regex': search_query, '$options': 'i'}},
                    {'channel': {'$regex': search_query, '$options': 'i'}},
                    {'root_cause': {'$regex': search_query, '$options': 'i'}},
                    {'assign_team': {'$regex': search_query, '$options': 'i'}}
                ]

            # Get total count
            total = tickets_collection.count_documents(query)
            
            # Paginate
            page = int(request.GET.get('page', 1))
            per_page = int(request.GET.get('per_page', 50))
            skip = (page - 1) * per_page
            
            # Fetch tickets
            tickets_cursor = tickets_collection.find(query).sort('created_at', -1).skip(skip).limit(per_page)
            tickets_data = [serialize_ticket(ticket) for ticket in tickets_cursor]
            
            total_pages = (total + per_page - 1) // per_page
            
            return JsonResponse({
                'success': True,
                'tickets': tickets_data,
                'total': total,
                'page': page,
                'pages': total_pages,
                'per_page': per_page,
                'source': 'mongodb'
            })
        else:
            # Fallback to Django ORM
            tickets = Ticket.objects.all().order_by('-created_at')
            
            # Apply filters if provided
            status_filter = request.GET.get('status')
            if status_filter and status_filter != 'all':
                tickets = tickets.filter(status=status_filter)
            
            search_query = request.GET.get('search')
            if search_query:
                tickets = tickets.filter(
                    Q(ticket_id__icontains=search_query) |
                    Q(viewer_id__icontains=search_query) |
                    Q(channel__icontains=search_query) |
                    Q(root_cause__icontains=search_query) |
                    Q(assign_team__icontains=search_query)
                )

            # Paginate if requested
            page = request.GET.get('page', 1)
            per_page = int(request.GET.get('per_page', 50))
            
            paginator = Paginator(tickets, per_page)
            page_obj = paginator.get_page(page)
            
            tickets_data = [serialize_ticket_model(ticket) for ticket in page_obj]
            
            return JsonResponse({
                'success': True,
                'tickets': tickets_data,
                'total': paginator.count,
                'page': page_obj.number,
                'pages': paginator.num_pages,
                'per_page': per_page,
                'source': 'django'
            })

    except Exception as e:
        logger.error(f"Error fetching tickets: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def ticket_update_api_view(request):
    """API endpoint to update individual ticket status in MongoDB."""
    try:
        data = json.loads(request.body)
        ticket_id = data.get('ticket_id')
        
        if not ticket_id:
            return JsonResponse({
                'success': False,
                'error': 'Ticket ID is required'
            }, status=400)
        
        # Try MongoDB first
        db = get_mongo_client()
        if db is not None:
            tickets_collection = db.tickets
            
            # Find ticket
            ticket = tickets_collection.find_one({'$or': [
                {'_id': ObjectId(ticket_id) if ObjectId.is_valid(ticket_id) else None},
                {'ticket_id': ticket_id}
            ]})
            
            if not ticket:
                return JsonResponse({
                    'success': False,
                    'error': 'Ticket not found'
                }, status=404)
            
            # Prepare update data
            update_data = {
                'updated_at': datetime.now()
            }
            
            # Track status change for history
            old_status = ticket.get('status')
            new_status = data.get('status')
            
            if 'status' in data:
                update_data['status'] = data['status']
            if 'priority' in data:
                update_data['priority'] = data['priority']
            if 'assigned_to' in data:
                update_data['assigned_to'] = data['assigned_to']
            if 'assign_team' in data:
                update_data['assign_team'] = data['assign_team']
            if 'internal_notes' in data:
                update_data['resolution_notes'] = data['internal_notes']
            
            # Add to status history if status changed
            if new_status and new_status != old_status:
                if 'status_history' not in ticket:
                    ticket['status_history'] = []
                
                ticket['status_history'].append({
                    'timestamp': datetime.now(),
                    'from_status': old_status,
                    'to_status': new_status,
                    'updated_by': data.get('updated_by', 'user'),
                    'notes': data.get('internal_notes', ''),
                    'escalated': data.get('escalate', False)
                })
                update_data['status_history'] = ticket['status_history']
            
            # Update ticket
            result = tickets_collection.update_one(
                {'_id': ticket['_id']},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                # Fetch updated ticket
                updated_ticket = tickets_collection.find_one({'_id': ticket['_id']})
                return JsonResponse({
                    'success': True,
                    'ticket': serialize_ticket(updated_ticket),
                    'message': f'Ticket {ticket.get("ticket_id", ticket_id)} updated successfully'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'No changes made'
                }, status=400)
        
        else:
            # Fallback to Django ORM
            try:
                ticket = Ticket.objects.get(pk=ticket_id)
            except Ticket.DoesNotExist:
                try:
                    ticket = Ticket.objects.get(ticket_id=ticket_id)
                except Ticket.DoesNotExist:
                    return JsonResponse({
                        'success': False,
                        'error': 'Ticket not found'
                    }, status=404)
            
            # Update ticket fields
            if 'status' in data:
                ticket.status = data['status']
            if 'priority' in data:
                ticket.priority = data['priority']
            if 'assigned_to' in data:
                ticket.assigned_to = data['assigned_to']
            if 'assign_team' in data:
                ticket.assign_team = data['assign_team']
            if 'internal_notes' in data:
                ticket.resolution_notes = data['internal_notes']
            
            ticket.updated_at = timezone.now()
            ticket.save()
            
            return JsonResponse({
                'success': True,
                'ticket': serialize_ticket_model(ticket),
                'message': f'Ticket {ticket.ticket_id} updated successfully'
            })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Error updating ticket: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def ticket_bulk_update_api_view(request):
    """API endpoint to update multiple tickets at once in MongoDB."""
    try:
        data = json.loads(request.body)
        ticket_ids = data.get('ticket_ids', [])
        
        if not ticket_ids:
            return JsonResponse({
                'success': False,
                'error': 'Ticket IDs are required'
            }, status=400)
        
        # Try MongoDB first
        db = get_mongo_client()
        if db is not None:
            tickets_collection = db.tickets
            
            # Prepare query for multiple ticket IDs
            query = {'$or': []}
            for tid in ticket_ids:
                if ObjectId.is_valid(tid):
                    query['$or'].append({'_id': ObjectId(tid)})
                query['$or'].append({'ticket_id': tid})
            
            # Prepare update data
            update_data = {
                'updated_at': datetime.now()
            }
            
            if 'status' in data:
                update_data['status'] = data['status']
            if 'assign_team' in data:
                update_data['assign_team'] = data['assign_team']
            if 'priority' in data:
                update_data['priority'] = data['priority']
            
            # Bulk update
            result = tickets_collection.update_many(query, {'$set': update_data})
            
            return JsonResponse({
                'success': True,
                'updated_count': result.modified_count,
                'message': f'{result.modified_count} tickets updated successfully'
            })
        
        else:
            # Fallback to Django ORM
            tickets = Ticket.objects.filter(pk__in=ticket_ids)
            
            if not tickets.exists():
                return JsonResponse({
                    'success': False,
                    'error': 'No tickets found'
                }, status=404)
            
            update_fields = {}
            if 'status' in data:
                update_fields['status'] = data['status']
            if 'assign_team' in data:
                update_fields['assign_team'] = data['assign_team']
            if 'priority' in data:
                update_fields['priority'] = data['priority']
            
            update_fields['updated_at'] = timezone.now()
            
            # Bulk update
            updated_count = tickets.update(**update_fields)
            
            return JsonResponse({
                'success': True,
                'updated_count': updated_count,
                'message': f'{updated_count} tickets updated successfully'
            })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Error bulk updating tickets: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def ticket_stats_api_view(request):
    """API endpoint to get ticket statistics for analytics."""
    try:
        # Try MongoDB first
        db = get_mongo_client()
        if db is not None:
            tickets_collection = db.tickets
            
            # Basic counts by status
            status_pipeline = [
                {'$group': {'_id': '$status', 'count': {'$sum': 1}}},
                {'$sort': {'_id': 1}}
            ]
            status_counts = list(tickets_collection.aggregate(status_pipeline))
            
            # Priority counts
            priority_pipeline = [
                {'$group': {'_id': '$priority', 'count': {'$sum': 1}}},
                {'$sort': {'_id': 1}}
            ]
            priority_counts = list(tickets_collection.aggregate(priority_pipeline))
            
            # Team assignment counts
            team_pipeline = [
                {'$group': {'_id': '$assign_team', 'count': {'$sum': 1}}},
                {'$sort': {'_id': 1}}
            ]
            team_counts = list(tickets_collection.aggregate(team_pipeline))
            
            # Time-based statistics
            from datetime import date, timedelta
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            time_stats = {
                'today': tickets_collection.count_documents({'created_at': {'$gte': today}}),
                'this_week': tickets_collection.count_documents({'created_at': {'$gte': week_ago}}),
                'this_month': tickets_collection.count_documents({'created_at': {'$gte': month_ago}}),
                'total': tickets_collection.count_documents({})
            }
            
            # Resolution time statistics
            resolved_count = tickets_collection.count_documents({'status': 'issue_resolved'})
            
            return JsonResponse({
                'success': True,
                'stats': {
                    'status_counts': [{'status': item['_id'], 'count': item['count']} for item in status_counts],
                    'priority_counts': [{'priority': item['_id'], 'count': item['count']} for item in priority_counts],
                    'team_counts': [{'assign_team': item['_id'], 'count': item['count']} for item in team_counts],
                    'time_stats': time_stats,
                    'resolved_count': resolved_count
                },
                'source': 'mongodb'
            })
        
        else:
            # Fallback to Django ORM
            from django.db.models import Count
            from datetime import timedelta
            
            now = timezone.now()
            
            # Basic counts by status
            status_counts = Ticket.objects.values('status').annotate(count=Count('id')).order_by('status')
            
            # Priority counts
            priority_counts = Ticket.objects.values('priority').annotate(count=Count('id')).order_by('priority')
            
            # Team assignment counts
            team_counts = Ticket.objects.values('assign_team').annotate(count=Count('id')).order_by('assign_team')
            
            # Time-based statistics
            today = now.date()
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            time_stats = {
                'today': Ticket.objects.filter(created_at__date=today).count(),
                'this_week': Ticket.objects.filter(created_at__gte=week_ago).count(),
                'this_month': Ticket.objects.filter(created_at__gte=month_ago).count(),
                'total': Ticket.objects.count()
            }
            
            # Resolution time statistics (for resolved tickets)
            resolved_tickets = Ticket.objects.filter(status='issue_resolved')
            
            return JsonResponse({
                'success': True,
                'stats': {
                    'status_counts': list(status_counts),
                    'priority_counts': list(priority_counts),
                    'team_counts': list(team_counts),
                    'time_stats': time_stats,
                    'resolved_count': resolved_tickets.count()
                },
                'source': 'django'
            })

    except Exception as e:
        logger.error(f"Error fetching ticket stats: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def management_view(request):
    """Render the ticket management page."""
    return render(request, "tg/management.html")

def analytics_view(request):
    """Render the analytics page."""
    return render(request, "tg/analytics.html")
