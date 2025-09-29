# COMPLETE OPTIMIZED mongo_service.py
# This leverages your existing django_mongodb_backend setup and fixes all pipeline issues

import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from django.db import transaction
from django.db.models import Q, Count
from django.utils import timezone
from datetime import timedelta
from typing import Dict, List, Optional, Any, Union, Tuple  # Add Tuple to existing import
logger = logging.getLogger(__name__)

# Import your Django MongoDB models
try:
    from .models import Session, KPI, Advancetags, Ticket, DataIngestionLog
except ImportError:
    logger.error("Could not import Django MongoDB models")

class MongoDBService:
    """
    Enhanced MongoDB service using Django ORM with djongo/django_mongodb_backend
    This is the BEST approach since you're already using django_mongodb_backend
    """
    
    def __init__(self):
        self.stats = {
            "fetched_records": 0,
            "processed_records": 0,
            "tickets_created": 0,
            "errors": []
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Django MongoDB connection and get collection counts"""
        try:
            # Test connection by querying each model
            counts = {
                "sessions": Session.objects.count(),
                "kpi": KPI.objects.count(), 
                "advancetags": Advancetags.objects.count(),
                "tickets": Ticket.objects.count()
            }
            
            logger.info(f"✅ Django MongoDB connection successful: {counts}")
            
            return {
                "connected": True,
                "collections": counts,
                "backend": "django_mongodb_backend",
                "total_records": sum(counts.values())
            }
            
        except Exception as e:
            logger.error(f"❌ Django MongoDB connection failed: {e}")
            return {
                "connected": False,
                "error": str(e),
                "collections": {}
            }
        
    def fetch_sessions_data(self, target_channels: List[str] = None, limit: int = None) -> pd.DataFrame:
        """Fetch using Django raw() method"""
        try:
            # Use Django raw query
            raw_query = "SELECT * FROM tg_session"
            sessions = Session.objects.raw(raw_query)

            # Convert to list of dicts with actual field names
            data = []
            for session in sessions:
                session_dict = {}
                for field in session._meta.fields:
                    # Get the actual database column name
                    db_column = getattr(field, 'db_column', field.name)
                    value = getattr(session, field.name, None)
                    session_dict[db_column or field.name] = value
                data.append(session_dict)

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            logger.info(f"✅ Fetched {len(df)} sessions via raw query")

            return df

        except Exception as e:
            logger.error(f"❌ Raw query failed: {e}")
            return pd.DataFrame()
        
    def fetch_sessions_data(self, target_channels: List[str] = None, limit: int = None) -> pd.DataFrame:
        """Fetch sessions data using Django ORM - NO DOUBLE MAPPING"""
        try:
            query = Session.objects.all()
    
            if target_channels:
                query = query.filter(asset_name__in=target_channels)
    
            if limit:
                query = query[:limit]
    
            data = list(query.values())
            if not data:
                logger.warning("No sessions data found")
                return pd.DataFrame()
    
            df = pd.DataFrame(data)
            
            # DON'T RENAME COLUMNS - Keep Django field names as-is
            # Remove Django auto fields if present
            columns_to_remove = ['id', '_id', 'created_at', 'updated_at']
            for col in columns_to_remove:
                if col in df.columns:
                    df = df.drop(col, axis=1)
    
            logger.info(f"Fetched {len(df)} sessions using Django ORM")
            logger.info(f"Available columns: {list(df.columns)}")
    
            self.stats["fetched_records"] += len(df)
            return df
    
        except Exception as e:
            logger.error(f"Error fetching sessions data: {e}")
            return pd.DataFrame()

    def fetch_kpi_data(self, target_channels: List[str] = None, limit: int = None) -> pd.DataFrame:
        """Fetch KPI data via Django ORM"""
        try:
            # Build query  
            query = KPI.objects.all()
            
            # Apply limit if specified
            if limit:
                query = query[:limit]
            
            # Convert to DataFrame
            data = list(query.values())
            
            if not data:
                logger.warning("No KPI data found in MongoDB")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Fetched {len(df)} KPI records via Django ORM")
            
            # Remove MongoDB ObjectId field if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            self.stats["fetched_records"] += len(df)
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching KPI data: {e}")
            self.stats["errors"].append(f"KPI fetch error: {e}")
            return pd.DataFrame()
    
    def fetch_advancetags_data(self, target_channels: List[str] = None, limit: int = None) -> pd.DataFrame:
        """Fetch advancetags data via Django ORM"""
        try:
            # Build query
            query = Advancetags.objects.all()
            
            # Apply channel filter if specified
            if target_channels:
                channel_filter = Q()
                for channel in target_channels:
                    channel_filter |= Q(asset_name__icontains=channel)
                query = query.filter(channel_filter)
            
            # Apply limit if specified
            if limit:
                query = query[:limit]
            
            # Convert to DataFrame
            data = list(query.values())
            
            if not data:
                logger.warning("No advancetags data found in MongoDB")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Fetched {len(df)} advancetags via Django ORM")
            
            # Remove MongoDB ObjectId field if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            self.stats["fetched_records"] += len(df)
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching advancetags data: {e}")
            self.stats["errors"].append(f"Advancetags fetch error: {e}")
            return pd.DataFrame()
    
    def fetch_all_collections(self, target_channels: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        FIXED: Fetch sessions, KPI, and advancetags via Django ORM
        This replaces the broken _fetch_from_mongodb_flexible function
        """
        logger.info("🔄 Starting Django MongoDB data fetch...")
        
        # Reset stats
        self.stats = {
            "fetched_records": 0,
            "processed_records": 0, 
            "tickets_created": 0,
            "errors": []
        }
        
        # Fetch all data types
        data = {
            "sessions": self.fetch_sessions_data(target_channels),
            "kpi_data": self.fetch_kpi_data(target_channels),
            "advancetags": self.fetch_advancetags_data(target_channels)
        }
        
        # Log summary
        total_records = sum(len(df) for df in data.values())
        logger.info(f"🎉 Django MongoDB fetch complete: {total_records} total records")
        
        for data_type, df in data.items():
            logger.info(f"   - {data_type}: {len(df)} records")
            if not df.empty and len(df.columns) > 0:
                logger.info(f"     Sample columns: {list(df.columns)[:5]}")
        
        return data
    
    def _extract_nested_field(self, ticket_data: Dict[str, Any], field_name: str, default: Any = None) -> Any:
        """Extract field from nested failure_details if not found at root level"""
        
        # Try root level first
        if field_name in ticket_data and ticket_data[field_name]:
            return ticket_data[field_name]
        
        # Try nested in failure_details
        failure_details = ticket_data.get('failure_details', {})
        if isinstance(failure_details, dict) and field_name in failure_details:
            return failure_details[field_name]
        
        # Try nested context_data within failure_details
        if isinstance(failure_details, dict):
            nested_context = failure_details.get('context_data', {})
            if isinstance(nested_context, dict) and field_name in nested_context:
                return nested_context[field_name]
        
        return default

    def _extract_confidence_score(self, ticket_data: Dict[str, Any]) -> float:
        """Enhanced confidence score extraction with multiple fallback sources"""

        # Try direct confidence_score field
        if 'confidence_score' in ticket_data:
            try:
                score = float(ticket_data['confidence_score'])
                return max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                pass
            
        # Try failure_details.confidence_score
        failure_details = ticket_data.get('failure_details', {})
        if isinstance(failure_details, dict):
            if 'confidence_score' in failure_details:
                try:
                    score = float(failure_details['confidence_score'])
                    return max(0.0, min(1.0, score))
                except (ValueError, TypeError):
                    pass
                
            # Try failure_details.failure_details.confidence (nested structure)
            nested_failure = failure_details.get('failure_details', {})
            if isinstance(nested_failure, dict) and 'confidence' in nested_failure:
                try:
                    confidence_val = nested_failure['confidence']
                    if isinstance(confidence_val, (int, float)):
                        return max(0.0, min(1.0, float(confidence_val)))
                    elif isinstance(confidence_val, str):
                        confidence_map = {
                            'high': 0.9, 'High': 0.9, 'HIGH': 0.9,
                            'medium': 0.7, 'Medium': 0.7, 'MEDIUM': 0.7,
                            'low': 0.5, 'Low': 0.5, 'LOW': 0.5
                        }
                        return confidence_map.get(confidence_val, 0.6)
                except (ValueError, TypeError):
                    pass
                
        # Default confidence score
        return 0.6
    
    def validate_ticket_data(self, ticket_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate ticket data before saving"""
        errors = []
        
        # Check required fields
        if not ticket_data.get('session_id'):
            errors.append("Missing session_id")
        
        # Check data types
        if 'confidence_score' in ticket_data:
            try:
                score = float(ticket_data['confidence_score'])
                if not (0.0 <= score <= 1.0):
                    errors.append("confidence_score must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("confidence_score must be a number")
        
        # Check suggested_actions is a list
        if 'suggested_actions' in ticket_data and not isinstance(ticket_data['suggested_actions'], list):
            errors.append("suggested_actions must be a list")
        
        return len(errors) == 0, errors
    
    def bulk_create_tickets(self, tickets_data: List[Dict[str, Any]]) -> int:
        """
        Robust ticket creation with comprehensive error handling
        Returns: Number of tickets successfully saved
        """
        if not tickets_data:
            logger.warning("No tickets data to save")
            return 0
    
        logger.info(f"Starting ticket save process: {len(tickets_data)} tickets to process")
    
        try:
            # Step 1: Prepare and validate ticket objects
            ticket_objects = []
            validation_failures = []
    
            for i, ticket_data in enumerate(tickets_data):
                try:
                    # Extract session_id with multiple fallbacks
                    session_id = (
                        ticket_data.get('session_id') or 
                        ticket_data.get('Session ID') or
                        ticket_data.get('Session Id')
                    )
                    
                    # Strict validation
                    if not session_id:
                        validation_failures.append(f"Ticket {i}: No session_id found")
                        continue
                    
                    session_id = str(session_id).strip()
                    if not session_id or session_id.lower() in ['nan', 'none', 'null', '']:
                        validation_failures.append(f"Ticket {i}: Invalid session_id '{session_id}'")
                        continue
                    
                    # Generate ticket_id
                    ticket_id = ticket_data.get('ticket_id')
                    if not ticket_id:
                        ticket_id = f"TKT_{session_id}"
                    ticket_id = str(ticket_id).strip()
                    
                    # Extract and validate other fields
                    title = ticket_data.get('title', '')
                    if not title:
                        asset_name = ticket_data.get('Asset Name', 
                                      ticket_data.get('asset_name', 'Unknown Channel'))
                        title = f"Video Issue - {asset_name}"
                    
                    description = ticket_data.get('description', '')
                    if not description:
                        description = f"Auto-generated ticket for session {session_id}"
                    
                    # Create ticket object with all required fields
                    ticket_obj = Ticket(
                        ticket_id=ticket_id,
                        session_id=session_id,
                        title=title[:200],  # Truncate if needed
                        description=description,
                        priority=ticket_data.get('priority', 'medium'),
                        status=ticket_data.get('status', 'new'),
                        assign_team=ticket_data.get('assign_team', 'technical'),
                        issue_type=ticket_data.get('issue_type', 'video_start_failure'),
                        confidence_score=self._extract_confidence_score(ticket_data),
                        failure_details=ticket_data.get('failure_details', {}),
                        context_data=ticket_data.get('context_data', {}),
                        suggested_actions=ticket_data.get('suggested_actions', []),
                        data_source='auto'
                    )
    
                    ticket_objects.append(ticket_obj)
                    
                except Exception as e:
                    validation_failures.append(f"Ticket {i}: {type(e).__name__} - {str(e)}")
                    continue
                
            # Log validation results
            if validation_failures:
                logger.warning(f"Validation failures: {len(validation_failures)}")
                for failure in validation_failures[:5]:  # Show first 5
                    logger.warning(f"  - {failure}")
            
            if not ticket_objects:
                logger.error("No valid ticket objects after validation")
                return 0
    
            logger.info(f"Validated {len(ticket_objects)} tickets successfully")
    
            # Step 2: Check for duplicates
            ticket_ids = [t.ticket_id for t in ticket_objects]
            existing_tickets = Ticket.objects.filter(ticket_id__in=ticket_ids)
            existing_ids = set(existing_tickets.values_list('ticket_id', flat=True))
            
            if existing_ids:
                logger.info(f"Found {len(existing_ids)} existing tickets, will skip duplicates")
                ticket_objects = [t for t in ticket_objects if t.ticket_id not in existing_ids]
                
                if not ticket_objects:
                    logger.info("All tickets already exist in database")
                    return len(existing_ids)
    
            logger.info(f"Ready to save {len(ticket_objects)} new tickets")
    
            # Step 3: Attempt bulk save with fallback strategies
            
            # Strategy 1: Standard bulk_create
            try:
                with transaction.atomic():
                    created_tickets = Ticket.objects.bulk_create(
                        ticket_objects,
                        ignore_conflicts=True
                    )
                    
                tickets_saved = len(created_tickets) if created_tickets else len(ticket_objects)
                logger.info(f"SUCCESS: Bulk save completed - {tickets_saved} tickets saved")
                self.stats["tickets_created"] = tickets_saved
                return tickets_saved
                
            except Exception as bulk_error:
                logger.error(f"Bulk save failed: {type(bulk_error).__name__}: {str(bulk_error)}")
                
                # Strategy 2: Try smaller batches
                logger.info("Attempting batch save strategy...")
                batch_size = 50
                saved_count = 0
                
                for batch_start in range(0, len(ticket_objects), batch_size):
                    batch = ticket_objects[batch_start:batch_start + batch_size]
                    try:
                        with transaction.atomic():
                            created = Ticket.objects.bulk_create(batch, ignore_conflicts=True)
                        batch_saved = len(created) if created else len(batch)
                        saved_count += batch_saved
                        logger.info(f"  Batch {batch_start//batch_size + 1}: Saved {batch_saved} tickets")
                        
                    except Exception as batch_error:
                        logger.error(f"  Batch {batch_start//batch_size + 1} failed: {str(batch_error)}")
                        
                        # Strategy 3: Individual saves for failed batch
                        logger.info(f"  Attempting individual saves for failed batch...")
                        for ticket_obj in batch:
                            try:
                                ticket_obj.save()
                                saved_count += 1
                            except Exception as individual_error:
                                logger.error(f"    Failed to save {ticket_obj.ticket_id}: {str(individual_error)}")
                
                if saved_count > 0:
                    logger.info(f"PARTIAL SUCCESS: Saved {saved_count}/{len(ticket_objects)} tickets")
                    self.stats["tickets_created"] = saved_count
                    return saved_count
                else:
                    logger.error("FAILED: Could not save any tickets")
                    return 0
    
        except Exception as e:
            logger.error(f"CRITICAL ERROR in bulk_create_tickets: {type(e).__name__}: {str(e)}", 
                        exc_info=True)
            return 0


    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed collection statistics"""
        try:
            stats = {
                "sessions": {
                    "total": Session.objects.count(),
                    "with_failures": Session.objects.filter(
                        Q(video_start_failure=True) | 
                        Q(exit_before_video_starts=True)
                    ).count(),
                    "recent_24h": Session.objects.filter(
                        created_at__gte=timezone.now() - timedelta(hours=24)
                    ).count()
                },
                "kpi": {
                    "total": KPI.objects.count(),
                    "recent_24h": KPI.objects.filter(
                        created_at__gte=timezone.now() - timedelta(hours=24)
                    ).count()
                },
                "advancetags": {
                    "total": Advancetags.objects.count(),
                    "recent_24h": Advancetags.objects.filter(
                        created_at__gte=timezone.now() - timedelta(hours=24)
                    ).count()
                },
                "tickets": {
                    "total": Ticket.objects.count(),
                    "open": Ticket.objects.exclude(status__in=['resolved', 'closed']).count(),
                    "session_based": Ticket.objects.exclude(session_id__isnull=True).count()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 1000) -> Dict[str, int]:
        """Cleanup old data (optional maintenance function)"""
        try:
            cutoff_date = timezone.now() - timedelta(days=days)
            
            # Count records to be deleted
            old_sessions = Session.objects.filter(created_at__lt=cutoff_date)
            old_kpi = KPI.objects.filter(created_at__lt=cutoff_date)  
            old_advancetags = Advancetags.objects.filter(created_at__lt=cutoff_date)
            
            counts = {
                "sessions_deleted": old_sessions.count(),
                "kpi_deleted": old_kpi.count(),
                "advancetags_deleted": old_advancetags.count()
            }
            
            # Delete old records
            with transaction.atomic():
                old_sessions.delete()
                old_kpi.delete()
                old_advancetags.delete()
            
            logger.info(f"🧹 Cleanup completed: {counts}")
            return counts
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
            return {}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self.stats,
            "has_errors": len(self.stats["errors"]) > 0,
            "success_rate": (
                (self.stats["fetched_records"] - len(self.stats["errors"])) / 
                max(self.stats["fetched_records"], 1) * 100
            )
        }

# Create singleton instance
_mongodb_service = MongoDBService()

def get_mongodb_service() -> MongoDBService:
    """Get MongoDB service singleton"""
    return _mongodb_service

# Convenience functions that match your existing API
def fetch_collections(target_channels: List[str] = None) -> Dict[str, pd.DataFrame]:
    """FIXED: Fetch collections using Django ORM - replaces broken function"""
    service = get_mongodb_service()
    return service.fetch_all_collections(target_channels)

def save_tickets(tickets_data: List[Dict[str, Any]]) -> int:
    """ENHANCED: Save tickets using Django ORM - replaces your function"""
    service = get_mongodb_service()
    return service.bulk_create_tickets(tickets_data)

def test_mongodb_connection() -> Dict[str, Any]:
    """Test Django MongoDB connection"""
    service = get_mongodb_service()
    return service.test_connection()

def get_mongodb_stats() -> Dict[str, Any]:
    """Get MongoDB collection statistics"""
    service = get_mongodb_service()
    return service.get_collection_stats()
