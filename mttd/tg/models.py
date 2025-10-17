# models.py - ENHANCED WITH UNIFIED PIPELINE AND SESSION-ONLY TICKETS

"""
Enhanced Models with Unified Pipeline Support
============================================

Key Features:
- EXACT column names from your dataset maintained
- Session-only ticket generation (no viewer_id)
- Variable channel support (no hardcoded names)
- Unified pipeline tracking and batch processing
- Comprehensive indexing for performance
"""

from typing import Any
from django.db import models, transaction
from django.db.models import Q
from django.utils import timezone
import uuid
from django_mongodb_backend.fields import ObjectIdField
from django.utils.translation import gettext_lazy as _
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# ============================================================================
# BASE MODEL WITH COMMON FUNCTIONALITY
# ============================================================================

class BaseModel(models.Model):
    """Base model with common fields and methods"""
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        abstract = True
        
    def to_dict(self):
        """Convert model instance to dictionary"""
        return {field.name: getattr(self, field.name) for field in self._meta.fields}

# ============================================================================
# KPI Collection - EXACT COLUMN NAMES FROM YOUR DATASET
# ============================================================================

class KPI(BaseModel):
    """KPI metrics collection - EXACT column names maintained"""
    
    # Core fields - EXACT NAMES FROM YOUR DATASET
    timestamp = models.DateTimeField(db_index=True)
    plays = models.BigIntegerField(default=0)
    
    # Playing time metrics - EXACT COLUMN NAMES
    playing_time_ended_mins = models.FloatField(
        db_column="Playing Time (Ended) (mins)",
        null=True, blank=True
    )
    streaming_performance_index = models.FloatField(
        db_column="Streaming Performance Index", 
        null=True, blank=True
    )
    
    # Video Start Failures - EXACT COLUMN NAMES FROM DATASET
    video_start_failures_technical = models.IntegerField(
        db_column="Video Start Failures Technical",
        default=0, db_index=True
    )
    video_start_failures_business = models.IntegerField(
        db_column="Video Start Failures Business",
        default=0, db_index=True
    )
    exit_before_video_starts = models.IntegerField(
        db_column="Exit Before Video Starts", 
        default=0, db_index=True
    )
    
    # Video Playback Failures - EXACT COLUMN NAMES
    video_playback_failures_technical = models.IntegerField(
        db_column="Video Playback Failures Technical",
        default=0
    )
    video_playback_failures_business = models.IntegerField(
        db_column="Video Playback Failures Business",
        default=0
    )
    
    # Quality metrics - EXACT COLUMN NAMES FROM DATASET
    video_start_time_sec = models.FloatField(
        db_column="Video Start Time(sec)",
        null=True, blank=True
    )
    rebuffering_ratio_pct = models.FloatField(
        db_column="Rebuffering Ratio(%)",
        null=True, blank=True
    )
    connection_induced_rebuffering_ratio_pct = models.FloatField(
        db_column="Connection Induced Rebuffering Ratio(%)",
        null=True, blank=True
    )
    video_restart_time_sec = models.FloatField(
        db_column="Video Restart Time(sec)",
        null=True, blank=True
    )
    avg_peak_bitrate_mbps = models.FloatField(
        db_column="Avg. Peak Bitrate(Mbps)",
        null=True, blank=True
    )
    
    # Unified pipeline tracking
    data_source = models.CharField(max_length=50, default='manual', db_index=True)
    processing_batch = models.CharField(max_length=100, null=True, blank=True, db_index=True)

    class Meta:
        db_table = "kpi_data"
        indexes = [
            models.Index(fields=['timestamp', 'data_source']),
            models.Index(fields=['plays', 'video_start_failures_technical']),
            models.Index(fields=['processing_batch']),
            models.Index(fields=['video_start_failures_business', 'exit_before_video_starts']),
        ]
        verbose_name = "KPI Metric"
        verbose_name_plural = "KPI Metrics"

    def __str__(self):
        return f"KPI-{self.timestamp}-{self.plays}plays"

# ============================================================================
# Session Collection - EXACT COLUMN NAMES FROM YOUR DATASET
# ============================================================================

class Session(BaseModel):
    """Session data collection - EXACT column names from dataset maintained"""
    
    # Primary identifiers - EXACT NAMES FROM DATASET
    viewer_id = models.CharField(
        db_column="Viewer ID",
        max_length=255, null=False, blank=True, db_index=True
    )
    
    session_id = models.CharField(
        db_column="Session ID",
        max_length=255, unique=False, db_index=False
    )

    # Time fields - EXACT COLUMN NAMES FROM DATASET  
    session_start_time = models.CharField(
        db_column="Session Start Time",
        max_length=255, null=True, blank=True, db_index=True
    )
    session_end_time = models.CharField(
        db_column="Session End Time",
        max_length=255, null=True, blank=True, db_index=True
    )
    
    # Content information - EXACT COLUMN NAMES
    asset_name = models.CharField(
        db_column="Asset Name",
        max_length=500, null=True, blank=True, db_index=True
    )
    
    # Status fields - EXACT COLUMN NAMES FROM DATASET
    status = models.CharField(
        db_column="Status",
        max_length=50, null=True, blank=True, db_index=True
    )
    ended_status = models.CharField(
        db_column="Ended Status",
        max_length=50, null=True, blank=True, db_index=True
    )
    ended_session = models.BooleanField(
        db_column="Ended Session",
        null=True, blank=True
    )
    impacted_session = models.BooleanField(
        db_column="Impacted Session",
        null=True, blank=True
    )
    
    # Quality metrics - EXACT COLUMN NAMES FROM DATASET
    playing_time = models.FloatField(
        db_column="Playing Time",
        null=True, blank=True
    )
    video_start_time = models.FloatField(
        db_column="Video Start Time",
        null=True, blank=True
    )
    total_video_restart_time = models.FloatField(
        db_column="Total Video Restart Time",
        null=True, blank=True
    )
    rebuffering_ratio = models.FloatField(
        db_column="Rebuffering Ratio",
        null=True, blank=True
    )
    connection_induced_rebuffering_ratio = models.FloatField(
        db_column="Connection Induced Rebuffering Ratio",
        null=True, blank=True
    )
    avg_peak_bitrate = models.FloatField(
        db_column="Avg. Peak Bitrate",
        null=True, blank=True
    )
    avg_average_bitrate = models.FloatField(
        db_column="Avg. Average Bitrate",
        null=True, blank=True
    )
    average_framerate = models.FloatField(
        db_column="Average Framerate",
        null=True, blank=True
    )
    starting_bitrate = models.FloatField(
        db_column="Starting Bitrate",
        null=True, blank=True
    )
    bitrate_switches = models.IntegerField(
        db_column="Bitrate Switches",
        null=True, blank=True
    )
    
    # Failure indicators - EXACT COLUMN NAMES FROM DATASET
    video_start_failure = models.BooleanField(
        db_column="Video Start Failure",
        null=True, blank=True, db_index=True
    )
    exit_before_video_starts = models.BooleanField(
        db_column="Exit Before Video Starts",
        null=True, blank=True, db_index=True
    )
    
    # Unified pipeline tracking
    data_source = models.CharField(max_length=50, default='manual', db_index=True)
    processing_batch = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    ticket_generated = models.BooleanField(default=False, db_index=True)

    class Meta:
        db_table = "sessions"
        indexes = [
            models.Index(fields=['session_id', 'data_source']),
            models.Index(fields=['viewer_id', 'session_start_time']),
            models.Index(fields=['asset_name', 'status']),
            models.Index(fields=['video_start_failure', 'ended_status']),
            models.Index(fields=['exit_before_video_starts', 'status']),
            models.Index(fields=['ticket_generated', 'processing_batch']),
        ]
        verbose_name = "Sessions"
        verbose_name_plural = "Sessions"

    def __str__(self):
        return f"Session-{self.session_id}-{self.asset_name or 'Unknown'}"
    
    @property
    def is_failure(self):
        """Check if this session represents a video start failure"""
        return (
            self.video_start_failure or
            self.status in ['VSF-T', 'VSF-B', 'EBVS'] or
            self.ended_status in ['VSF-T', 'VSF-B', 'EBVS'] or
            self.exit_before_video_starts
        )
    
    @property
    def failure_type(self):
        """Get the specific failure type"""
        if self.status in ['VSF-T', 'VSF-B', 'EBVS']:
            return self.status
        elif self.ended_status in ['VSF-T', 'VSF-B', 'EBVS']:
            return self.ended_status
        elif self.exit_before_video_starts:
            return 'EBVS'
        elif self.video_start_failure:
            return 'VSF-T'
        return None

# ============================================================================
# Advancetags Collection - EXACT COLUMN NAMES FROM YOUR DATASET
# ============================================================================

class Advancetags(BaseModel):
    """Advanced metadata collection - EXACT column names from dataset"""
    
    # Primary identifier - EXACT NAME FROM DATASET
    viewer_id = models.CharField(
        db_column="Viewer ID",
        max_length=255, null=False, blank=True, db_index=True
    )
    
    session_id = models.CharField(
        db_column="Session ID",
        max_length=255, db_index=True
    )
    
    # Content information - EXACT COLUMN NAMES  
    asset_name = models.CharField(
        db_column="Asset Name",
        max_length=500, null=True, blank=True
    )
    content_category = models.CharField(
        db_column="Content Category",
        max_length=200, null=True, blank=True
    )
    
    # Browser information - EXACT COLUMN NAMES FROM DATASET
    browser_name = models.CharField(
        db_column="Browser Name", 
        max_length=200, null=True, blank=True, db_index=True
    )
    browser_version = models.CharField(
        db_column="Browser Version",
        max_length=100, null=True, blank=True
    )
    
    # Device information - EXACT COLUMN NAMES FROM DATASET
    device_hardware_type = models.CharField(
        db_column="Device Hardware Type",
        max_length=200, null=True, blank=True
    )
    device_manufacturer = models.CharField(
        db_column="Device Manufacturer", 
        max_length=200, null=True, blank=True
    )
    device_marketing_name = models.CharField(
        db_column="Device Marketing Name",
        max_length=200, null=True, blank=True
    )
    device_model = models.CharField(
        db_column="Device Model",
        max_length=200, null=True, blank=True
    )
    device_name = models.CharField(
        db_column="Device Name",
        max_length=200, null=True, blank=True
    )
    device_operating_system = models.CharField(
        db_column="Device Operating System",
        max_length=200, null=True, blank=True, db_index=True
    )
    device_operating_system_family = models.CharField(
        db_column="Device Operating System Family",
        max_length=200, null=True, blank=True
    )
    device_operating_system_version = models.CharField(
        db_column="Device Operating System Version",
        max_length=100, null=True, blank=True
    )
    
    # App and Player information - EXACT COLUMN NAMES
    app_name = models.CharField(
        db_column="App Name",
        max_length=200, null=True, blank=True
    )
    app_version = models.CharField(
        db_column="App Version", 
        max_length=100, null=True, blank=True
    )
    player_framework_name = models.CharField(
        db_column="Player Framework Name",
        max_length=200, null=True, blank=True
    )
    player_framework_version = models.CharField(
        db_column="Player Framework Version",
        max_length=100, null=True, blank=True
    )
    
    # Network and CDN information - EXACT COLUMN NAMES
    cdn = models.CharField(
        db_column="CDN", 
        max_length=100, null=True, blank=True, db_index=True
    )
    last_cdn = models.CharField(
        db_column="Last CDN",
        max_length=100, null=True, blank=True
    )
    
    # Geographic information - EXACT COLUMN NAMES FROM DATASET
    city = models.CharField(
        db_column="City",
        max_length=100, null=True, blank=True, db_index=True
    )
    state = models.CharField(
        db_column="State",
        max_length=100, null=True, blank=True
    )
    country = models.CharField(
        db_column="Country",
        max_length=100, null=True, blank=True, db_index=True
    )
    address = models.TextField(
        db_column="Address",
        null=True, blank=True
    )
    
    # Network details - EXACT COLUMN NAMES FROM DATASET
    ip = models.GenericIPAddressField(
        db_column="IP",
        null=True, blank=True, protocol='both'
    )
    ipv6 = models.GenericIPAddressField(
        db_column="IPv6", 
        null=True, blank=True, protocol='IPv6'
    )
    asnname = models.CharField(
        db_column="ASNName",
        max_length=200, null=True, blank=True
    )
    ispname = models.CharField(
        db_column="ISPName",
        max_length=200, null=True, blank=True, db_index=True
    )
    
    # Stream information - EXACT COLUMN NAME
    streamurl = models.URLField(
        db_column="StreamURL",
        max_length=1000, null=True, blank=True
    )
    
    # Unified pipeline tracking
    data_source = models.CharField(max_length=50, default='manual', db_index=True)
    processing_batch = models.CharField(max_length=100, null=True, blank=True, db_index=True)

    class Meta:
        db_table = "advancetags"
        indexes = [
            models.Index(fields=['session_id', 'data_source']),
            models.Index(fields=['ispname', 'city']),
            models.Index(fields=['cdn', 'device_operating_system']),
            models.Index(fields=['processing_batch']),
        ]
        verbose_name = "Advanced Tags"
        verbose_name_plural = "Advanced Tags"

    def __str__(self):
        return f"Meta-{self.session_id}-{self.ispname or 'Unknown'}"

# ============================================================================
# ENHANCED TICKET MODEL - SESSION-ONLY, VARIABLE CHANNELS
# ============================================================================

class Ticket(BaseModel):
    """Enhanced ticket model - SESSION-ONLY tickets with variable channel support"""
    
    CURRENT_STATUS = [
        ('new', _('New')),
        ('InProgress', _('InProgress')),
        ('resolved', _('Resolved')),
        ('closed', _('Closed')),
    ]
    
    PRIORITY_CHOICES = [
        ('low', _('Low')),
        ('medium', _('Medium')),
        ('high', _('High')),
        ('critical', _('Critical')),
    ]
    
    TEAM_CHOICES = [
        ('technical', _('Technical Team')),
        ('network', _('Network Team')),
        ('content', _('Content Team')),
        ('customer_service', _('Customer Service')),
    ]
    
    # Ticket identification
    ticket_id = models.CharField(max_length=100, unique=True, db_index=True)
    
    # SESSION-ONLY approach - no viewer_id as requested
    session_id = models.CharField(max_length=255, null=True, blank=True, db_index=True)
    
    # Ticket details
    title = models.CharField(max_length=500, null=True, blank=True)
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, db_index=True)
    status = models.CharField(max_length=20, choices=CURRENT_STATUS, db_index=True)
    assign_team = models.CharField(max_length=50, choices=TEAM_CHOICES, db_index=True)
    
    # Issue information
    issue_type = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    description = models.TextField(null=True, blank=True)
    resolution_notes = models.TextField(null=True, blank=True)
    
    # ENHANCED: Multi-layer diagnostic fields
    confidence_score = models.FloatField(null=True, blank=True, db_index=True)  # 0.0-1.0
    severity_score = models.IntegerField(db_index=True)  # 0-10 scale
    
    # Enhanced data fields for MVP ticket engine
    failure_details = models.JSONField(default=dict, null=True, blank=True)
    context_data = models.JSONField(default=dict, null=True, blank=True)
    suggested_actions = models.JSONField(default=list, null=True, blank=True)
    
    # Unified pipeline tracking
    data_source = models.CharField(max_length=50, default='auto', db_index=True)
    processing_batch = models.CharField(max_length=100, null=True, blank=True, db_index=True)

    # Timestamps
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "tickets"
        indexes = [
            models.Index(fields=['ticket_id', 'status']),
            models.Index(fields=['session_id']),  # SESSION-ONLY indexing
            models.Index(fields=['priority', 'assign_team']),
            models.Index(fields=['processing_batch', 'data_source']),
            models.Index(fields=['created_at', 'status']),
            models.Index(fields=['issue_type', 'confidence_score']),
            models.Index(fields=['severity_score', 'status']),
            models.Index(fields=['confidence_score', 'severity_score']),
            models.Index(fields=['severity_score', 'created_at']),
        ]
        verbose_name = "Ticket"
        verbose_name_plural = "Tickets"

    def save(self, *args, **kwargs):
        """Enhanced save with auto-generation of ticket_id"""
        if not self.ticket_id:
            self.ticket_id = f"TKT_{self.viewer_id}_{self.session_id}"

        # ENHANCED: Validate severity_score range (0-10)
        if self.severity_score is not None:
            self.severity_score = max(0, min(10, int(self.severity_score)))
        
        # ENHANCED: Validate confidence_score range (0.0-1.0)
        if self.confidence_score is not None:
            self.confidence_score = max(0.0, min(1.0, float(self.confidence_score)))
        
        # ENHANCED: Ensure failure_details is a valid dict
        if not isinstance(self.failure_details, dict):
            self.failure_details = {}
            
        # Set resolved_at when status changes to resolved
        if self.status == 'resolved' and not self.resolved_at:
            self.resolved_at = timezone.now()
        elif self.status != 'resolved':
            self.resolved_at = None
            
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Ticket-{self.ticket_id}-{self.status}-{self.severity_label}"
    
    @property
    def severity_label(self) -> str:
        """Get human-readable severity label from score"""
        if self.severity_score >= 8:
            return 'CRITICAL'
        elif self.severity_score >= 6:
            return 'HIGH'
        elif self.severity_score >= 4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    @property
    def confidence_percentage(self) -> int:
        """Get confidence as percentage"""
        if self.confidence_score is not None:
            return int(self.confidence_score * 100)
        return 0
    
    @property
    def root_cause(self) -> str:
        """Extract root cause from failure_details"""
        if isinstance(self.failure_details, dict):
            return self.failure_details.get('root_cause', 'Unknown')
        return 'Unknown'
    
    @property
    def user_behavior_pattern(self) -> str:
        """Extract user behavior pattern from failure_details"""
        if isinstance(self.failure_details, dict):
            user_behavior = self.failure_details.get('user_behavior', {})
            if isinstance(user_behavior, dict):
                return user_behavior.get('pattern', 'unknown')
        return 'unknown'
    
    @property
    def user_value_tier(self) -> str:
        """Extract user value tier from failure_details"""
        if isinstance(self.failure_details, dict):
            user_behavior = self.failure_details.get('user_behavior', {})
            if isinstance(user_behavior, dict):
                return user_behavior.get('user_value_tier', 'unknown')
        return 'unknown'
    
    @property
    def is_deteriorating_user(self) -> bool:
        """Check if ticket represents deteriorating user experience"""
        return self.user_behavior_pattern == 'deteriorating_experience'
    
    @property
    def is_reliable_user_affected(self) -> bool:
        """Check if reliable user was affected"""
        return self.user_behavior_pattern == 'reliable_user_hit_issue'
    
    @property
    def is_premium_user(self) -> bool:
        """Check if premium user was affected"""
        return self.user_value_tier == 'premium'
    
    @property
    def temporal_trend(self) -> str:
        """Get temporal trend from failure_details"""
        if isinstance(self.failure_details, dict):
            temporal = self.failure_details.get('temporal_analysis', {})
            if isinstance(temporal, dict):
                return temporal.get('trend', 'unknown')
        return 'unknown'
    
    @property
    def temporal_pattern(self) -> str:
        """Get temporal pattern from failure_details"""
        if isinstance(self.failure_details, dict):
            temporal = self.failure_details.get('temporal_analysis', {})
            if isinstance(temporal, dict):
                return temporal.get('pattern', 'unknown')
        return 'unknown'
    
    @property
    def has_geographic_correlation(self) -> bool:
        """Check if ticket has geographic correlation data"""
        if isinstance(self.failure_details, dict):
            geo_analysis = self.failure_details.get('geographic_analysis', {})
            if isinstance(geo_analysis, dict):
                return geo_analysis.get('is_widespread', False)
        return False
    
    @property
    def concurrent_failures_count(self) -> int:
        """Get count of concurrent failures from geographic analysis"""
        if isinstance(self.failure_details, dict):
            geo_analysis = self.failure_details.get('geographic_analysis', {})
            if isinstance(geo_analysis, dict):
                return geo_analysis.get('concurrent_count', 0)
        return 0
    
    @property
    def affected_location(self) -> str:
        """Get affected location from geographic analysis"""
        if isinstance(self.failure_details, dict):
            geo_analysis = self.failure_details.get('geographic_analysis', {})
            if isinstance(geo_analysis, dict):
                city = geo_analysis.get('affected_city', '')
                isp = geo_analysis.get('affected_isp', '')
                if city and isp:
                    return f"{city} ({isp})"
                return city or isp or 'Unknown'
        return 'Unknown'

    # ========================================================================
    # ENHANCED ANALYSIS METHODS
    # ========================================================================
    
    def get_diagnostic_summary(self) -> dict:
        """Get comprehensive diagnostic summary"""
        return {
            'ticket_id': self.ticket_id,
            'session_id': self.session_id,
            'root_cause': self.root_cause,
            'confidence': self.confidence_score,
            'confidence_percentage': self.confidence_percentage,
            'severity_score': self.severity_score,
            'severity_label': self.severity_label,
            'user_pattern': self.user_behavior_pattern,
            'user_value_tier': self.user_value_tier,
            'temporal_trend': self.temporal_trend,
            'temporal_pattern': self.temporal_pattern,
            'geographic_widespread': self.has_geographic_correlation,
            'concurrent_failures': self.concurrent_failures_count,
            'affected_location': self.affected_location,
            'status': self.status,
            'priority': self.priority,
            'assign_team': self.assign_team
        }
    
    def get_user_impact_analysis(self) -> dict:
        """Get detailed user impact analysis"""
        if not isinstance(self.failure_details, dict):
            return {'available': False}
        
        user_behavior = self.failure_details.get('user_behavior', {})
        if not isinstance(user_behavior, dict):
            return {'available': False}
        
        return {
            'available': True,
            'pattern': user_behavior.get('pattern', 'unknown'),
            'behavior_trend': user_behavior.get('behavior_trend', 'unknown'),
            'user_value_tier': user_behavior.get('user_value_tier', 'unknown'),
            'suggested_action': user_behavior.get('suggested_action', 'standard_triage'),
            'confidence_modifier': user_behavior.get('confidence_modifier', 0.0),
            'severity_boost': user_behavior.get('severity_boost', False),
            'consecutive_failures': user_behavior.get('consecutive_failures', 0)
        }
    
    def get_temporal_impact_analysis(self) -> dict:
        """Get detailed temporal impact analysis"""
        if not isinstance(self.failure_details, dict):
            return {'available': False}
        
        temporal = self.failure_details.get('temporal_analysis', {})
        if not isinstance(temporal, dict):
            return {'available': False}
        
        return {
            'available': True,
            'trend': temporal.get('trend', 'unknown'),
            'pattern': temporal.get('pattern', 'unknown'),
            'confidence_boost': temporal.get('confidence_boost', 0.0),
            'consecutive_failures': temporal.get('consecutive_failures', 0)
        }
    
    def get_geographic_impact_analysis(self) -> dict:
        """Get detailed geographic impact analysis"""
        if not isinstance(self.failure_details, dict):
            return {'available': False}
        
        geo = self.failure_details.get('geographic_analysis', {})
        if not isinstance(geo, dict):
            return {'available': False}
        
        return {
            'available': True,
            'is_widespread': geo.get('is_widespread', False),
            'concurrent_count': geo.get('concurrent_count', 0),
            'affected_isp': geo.get('affected_isp', 'Unknown'),
            'affected_city': geo.get('affected_city', 'Unknown'),
            'confidence_boost': geo.get('confidence_boost', 0.0)
        }
    
    def requires_immediate_action(self) -> bool:
        """Check if ticket requires immediate action"""
        return (
            self.severity_score >= 8 or
            self.is_deteriorating_user or
            (self.is_reliable_user_affected and self.is_premium_user) or
            self.has_geographic_correlation
        )
    
    def get_escalation_recommendation(self) -> dict:
        """Get escalation recommendation based on multi-layer analysis"""
        should_escalate = False
        reasons = []
        urgency = 'normal'
        
        # Check severity
        if self.severity_score >= 8:
            should_escalate = True
            reasons.append(f"Critical severity ({self.severity_score}/10)")
            urgency = 'critical'
        
        # Check user impact
        if self.is_deteriorating_user:
            should_escalate = True
            reasons.append("Deteriorating user experience detected")
            if urgency == 'normal':
                urgency = 'high'
        
        if self.is_reliable_user_affected and self.is_premium_user:
            should_escalate = True
            reasons.append("Premium reliable user affected")
            if urgency == 'normal':
                urgency = 'high'
        
        # Check geographic impact
        if self.has_geographic_correlation:
            should_escalate = True
            reasons.append(f"Widespread issue: {self.concurrent_failures_count} concurrent failures")
            if urgency != 'critical':
                urgency = 'high'
        
        # Check temporal patterns
        if self.temporal_pattern == 'service_degradation':
            should_escalate = True
            reasons.append("Service degradation pattern detected")
            if urgency == 'normal':
                urgency = 'high'
        
        return {
            'should_escalate': should_escalate,
            'urgency': urgency,
            'reasons': reasons,
            'recommended_team': self.assign_team,
            'confidence': self.confidence_score
        }
# ============================================================================
# UNIFIED PIPELINE SUPPORT MODELS
# ============================================================================

class ProcessingBatch(BaseModel):
    """Track processing batches across all operations"""
    
    batch_id = models.CharField(max_length=100, unique=True, db_index=True)
    description = models.CharField(max_length=500, null=True, blank=True)
    source_type = models.CharField(max_length=50, db_index=True)  # 'manual', 'mongodb'
    status = models.CharField(max_length=20, default='active', db_index=True)
    
    total_records = models.BigIntegerField(default=0)
    processed_records = models.BigIntegerField(default=0)
    failed_records = models.BigIntegerField(default=0)

    class Meta:
        db_table = "processing_batches"
        verbose_name = "Processing Batch"
        verbose_name_plural = "Processing Batches"

    def __str__(self):
        return f"Batch-{self.batch_id}-{self.status}"

class DataIngestionLog(BaseModel):
    """Comprehensive logging for data ingestion operations"""
    
    ingestion_id = models.CharField(max_length=100, unique=True, db_index=True)
    processing_batch = models.CharField(max_length=100, db_index=True)
    source_type = models.CharField(max_length=50, db_index=True)  # 'manual', 'mongodb'
    status = models.CharField(max_length=20, db_index=True)  # 'processing', 'success', 'failed'
    
    # File processing details
    total_files_processed = models.IntegerField(default=0)
    uploaded_files = models.JSONField(default=list, null=True, blank=True)
    
    # Record processing statistics
    records_processed = models.BigIntegerField(default=0)
    kpi_records = models.BigIntegerField(default=0)
    session_records = models.BigIntegerField(default=0)  
    advancetags_records = models.BigIntegerField(default=0)
    
    # Ticket generation statistics
    tickets_generated = models.IntegerField(default=0)
    tickets_saved = models.IntegerField(default=0)
    
    # Quality and performance metrics
    data_quality_score = models.FloatField(null=True, blank=True)
    validation_errors = models.IntegerField(default=0)
    validation_warnings = models.IntegerField(default=0)
    processing_time_seconds = models.FloatField(null=True, blank=True)
    memory_usage_mb = models.FloatField(null=True, blank=True)
    
    # Error details
    error_details = models.TextField(null=True, blank=True)
    
    # Timestamps
    started_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "data_ingestion_logs"
        indexes = [
            models.Index(fields=['ingestion_id', 'status']),
            models.Index(fields=['processing_batch', 'source_type']),
            models.Index(fields=['started_at', 'completed_at']),
        ]
        verbose_name = "Data Ingestion Log"
        verbose_name_plural = "Data Ingestion Logs"

    def __str__(self):
        return f"Ingestion-{self.ingestion_id}-{self.status}"
    
    def complete_processing(self, status='success'):
        """Mark processing as complete"""
        self.status = status
        self.completed_at = timezone.now()
        if self.started_at:
            self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()
        self.save()

# ============================================================================
# UNIFIED DATA MANAGER CLASS
# ============================================================================

class UnifiedDataManager:
    """Unified manager for all data operations and statistics"""
    
    @staticmethod
    def generate_batch_id():
        """Generate unique processing batch ID"""
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        return f"BATCH_{timestamp}_{str(uuid.uuid4())[:8]}"
    
    @staticmethod
    def create_processing_batch(description, source_type):
        """Create a new processing batch"""
        batch_id = UnifiedDataManager.generate_batch_id()
        batch = ProcessingBatch.objects.create(
            batch_id=batch_id,
            description=description,
            source_type=source_type
        )
        return batch
    
    @staticmethod
    @transaction.atomic
    def bulk_create_with_batch(model_class, data_list, batch_id, data_source='manual'):
        """Bulk create records with batch tracking"""
        if not data_list:
            return 0
            
        # Add batch tracking to each record
        for record in data_list:
            record.processing_batch = batch_id
            record.data_source = data_source
            
        # Bulk create with conflict handling
        created_objects = model_class.objects.bulk_create(
            data_list, 
            ignore_conflicts=True,
            batch_size=1000
        )
        
        return len(created_objects)
    
    @staticmethod
    def get_data_statistics():
        """Get comprehensive data statistics for dashboard"""
        from django.db.models import Q, Count
        
        return {
            'kpi': {
                'total': KPI.objects.count(),
                'manual': KPI.objects.filter(data_source='manual').count(),
                'mongodb': KPI.objects.filter(data_source='mongodb').count(),
                'recent': KPI.objects.filter(
                    created_at__gte=timezone.now() - timezone.timedelta(hours=24)
                ).count(),
            },
            'sessions': {
                'total': Session.objects.count(),
                'manual': Session.objects.filter(data_source='manual').count(),
                'mongodb': Session.objects.filter(data_source='mongodb').count(),
                'failures': Session.objects.filter(
                    Q(video_start_failure=True) | 
                    Q(status__in=['VSF-T', 'VSF-B', 'EBVS']) |
                    Q(ended_status__in=['VSF-T', 'VSF-B', 'EBVS']) |
                    Q(exit_before_video_starts=True)
                ).count(),
                'recent': Session.objects.filter(
                    created_at__gte=timezone.now() - timezone.timedelta(hours=24)
                ).count(),
            },
            'advancetags': {
                'total': Advancetags.objects.count(),
                'manual': Advancetags.objects.filter(data_source='manual').count(),
                'mongodb': Advancetags.objects.filter(data_source='mongodb').count(),
                'recent': Advancetags.objects.filter(
                    created_at__gte=timezone.now() - timezone.timedelta(hours=24)
                ).count(),
            },
            'tickets': {
                'total': Ticket.objects.count(),
                'new': Ticket.objects.filter(status='new').count(),
                'InProgress': Ticket.objects.filter(status='InProgress').count(),
                'resolved': Ticket.objects.filter(status='resolved').count(),
                'session_based': Ticket.objects.exclude(session_id__isnull=True).count(),
                'high_confidence': Ticket.objects.filter(confidence_score__gte=0.8).count(),
            },
            'processing': {
                'active_batches': ProcessingBatch.objects.filter(status='active').count(),
                'recent_ingestions': DataIngestionLog.objects.filter(
                    started_at__gte=timezone.now() - timezone.timedelta(hours=24)
                ).count(),
                'success_rate': UnifiedDataManager._calculate_success_rate(),
            }
        }
    
    @staticmethod 
    def _calculate_success_rate():
        """Calculate processing success rate"""
        total_logs = DataIngestionLog.objects.count()
        if total_logs == 0:
            return 100.0
        success_logs = DataIngestionLog.objects.filter(status='success').count()
        return round((success_logs / total_logs) * 100, 2)
    
    @staticmethod
    def get_enhanced_ticket_statistics() -> Dict[str, Any]:
        """Get enhanced ticket statistics with multi-layer diagnostics"""
        try:
            from django.db.models import Count, Avg, Q
            
            tickets = Ticket.objects.all()
            total = tickets.count()
            
            if total == 0:
                return {
                    'total': 0,
                    'message': 'No tickets available'
                }
            
            # Severity distribution
            severity_dist = {
                'critical': tickets.filter(severity_score__gte=8).count(),
                'high': tickets.filter(severity_score__gte=6, severity_score__lt=8).count(),
                'medium': tickets.filter(severity_score__gte=4, severity_score__lt=6).count(),
                'low': tickets.filter(severity_score__lt=4).count()
            }
            
            # Confidence distribution
            confidence_dist = {
                'high': tickets.filter(confidence_score__gte=0.8).count(),
                'medium': tickets.filter(confidence_score__gte=0.6, confidence_score__lt=0.8).count(),
                'low': tickets.filter(confidence_score__lt=0.6).count()
            }
            
            # User impact analysis
            deteriorating_users = sum(1 for t in tickets if t.is_deteriorating_user)
            reliable_users_affected = sum(1 for t in tickets if t.is_reliable_user_affected)
            premium_users_affected = sum(1 for t in tickets if t.is_premium_user)
            
            # Geographic impact
            widespread_issues = sum(1 for t in tickets if t.has_geographic_correlation)
            
            # Temporal patterns
            service_degradation = sum(
                1 for t in tickets 
                if t.temporal_pattern == 'service_degradation'
            )
            
            # Averages
            avg_confidence = tickets.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
            avg_severity = tickets.aggregate(Avg('severity_score'))['severity_score__avg'] or 0
            
            # Tickets requiring immediate action
            immediate_action_needed = sum(1 for t in tickets if t.requires_immediate_action())
            
            return {
                'total': total,
                'by_severity': severity_dist,
                'by_confidence': confidence_dist,
                'averages': {
                    'confidence': round(avg_confidence, 2),
                    'severity': round(avg_severity, 1)
                },
                'user_impact': {
                    'deteriorating_users': deteriorating_users,
                    'reliable_users_affected': reliable_users_affected,
                    'premium_users_affected': premium_users_affected
                },
                'patterns': {
                    'widespread_issues': widespread_issues,
                    'service_degradation': service_degradation
                },
                'actionable': {
                    'immediate_action_needed': immediate_action_needed,
                    'percentage': round((immediate_action_needed / total) * 100, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced ticket statistics: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def get_critical_tickets_for_review() -> List[Ticket]:
        """Get tickets that need immediate review"""
        try:
            tickets = Ticket.objects.filter(
                Q(severity_score__gte=8) |
                Q(confidence_score__gte=0.8, severity_score__gte=6)
            ).exclude(
                status__in=['resolved', 'closed']
            ).order_by('-severity_score', '-confidence_score', '-created_at')
            
            # Additional filtering using properties
            critical_tickets = [
                t for t in tickets 
                if t.requires_immediate_action()
            ]
            
            return critical_tickets
            
        except Exception as e:
            logger.error(f"Error getting critical tickets: {e}")
            return []
        
    @staticmethod
    def cleanup_old_batches(days=7):
        """Cleanup old processing batches and logs"""
        cutoff_date = timezone.now() - timezone.timedelta(days=days)
        
        # Cleanup completed batches
        old_batches = ProcessingBatch.objects.filter(
            created_at__lt=cutoff_date,
            status__in=['completed', 'failed']
        )
        count = old_batches.count()
        old_batches.delete()
        
        return count
    
    

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_batch_id():
    """Generate unique batch ID - Wrapper for compatibility"""
    return UnifiedDataManager.generate_batch_id()

def get_data_statistics():
    """Get data statistics - Wrapper for compatibility"""
    return UnifiedDataManager.get_data_statistics()
