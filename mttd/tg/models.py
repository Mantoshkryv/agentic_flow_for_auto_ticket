# UPDATED models.py - COMPLETE WITH EXACT COLUMN MAPPINGS

from django.db import models
from django.utils import timezone
import uuid
from django_mongodb_backend.fields import ObjectIdField
from django.utils.translation import gettext_lazy as _

# ---------------- KPI Collection - UPDATED WITH EXACT COLUMNS ----------------
class KPI(models.Model):
    """KPI metrics collection in MongoDB - UPDATED FOR EXACT COLUMN MAPPING"""
    # Core fields
    timestamp = models.DateTimeField(db_index=True)
    created_at = models.DateTimeField(default=timezone.now)  
    plays = models.IntegerField(default=0)
    playing_time_ended_mins = models.FloatField(null=True, blank=True)
    streaming_performance_index = models.FloatField(null=True, blank=True)
    
    # Video Start Failures - EXACT COLUMN NAMES
    video_start_failures_technical = models.IntegerField(default=0)
    video_start_failures_business = models.IntegerField(default=0)
    exit_before_video_starts = models.IntegerField(default=0)
    
    # Video Playback Failures - EXACT COLUMN NAMES
    video_playback_failures_technical = models.IntegerField(default=0)
    video_playback_failures_business = models.IntegerField(default=0)
    
    # Streaming QoE metrics - UPDATED WITH EXACT COLUMNS
    video_start_time_sec = models.FloatField(null=True, blank=True)
    rebuffering_ratio_pct = models.FloatField(null=True, blank=True)
    connection_induced_rebuffering_ratio_pct = models.FloatField(null=True, blank=True)  # NEW FIELD
    video_restart_time_sec = models.FloatField(null=True, blank=True)  # NEW FIELD
    avg_peak_bitrate_mbps = models.FloatField(null=True, blank=True)  # NEW FIELD
    
    class Meta:
        db_table = "kpi_data"
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['plays']),
        ]
        verbose_name = "KPI Metric"
        verbose_name_plural = "KPI Metrics"
    
    def __str__(self):
        return f"KPI - {self.timestamp} - {self.plays} plays"

# ---------------- Sessions Collection - UPDATED WITH ALL COLUMNS ----------------
class Session(models.Model):
    """Session data collection in MongoDB - UPDATED FOR ALL COLUMNS"""
    # Primary identifiers
    session_id = models.CharField(max_length=200, unique=True, db_index=True)
    viewer_id = models.CharField(max_length=200, null=True, blank=True, db_index=True)
    
    # Time fields - UPDATED WITH EXACT COLUMNS
    session_start_time = models.DateTimeField(null=True, blank=True, db_index=True)
    session_end_time = models.DateTimeField(null=True, blank=True)
    playing_time = models.FloatField(null=True, blank=True)
    video_start_time = models.FloatField(null=True, blank=True)
    total_video_restart_time = models.FloatField(null=True, blank=True)
    
    # Content information
    asset_name = models.CharField(max_length=500, null=True, blank=True, db_index=True)
    channel = models.CharField(max_length=200, null=True, blank=True, db_index=True)
    
    # Session status fields - EXACT COLUMN NAMES
    status = models.CharField(max_length=50, null=True, blank=True, db_index=True)
    ended_status = models.CharField(max_length=50, null=True, blank=True)
    ended_session = models.CharField(max_length=50, null=True, blank=True)
    impacted_session = models.CharField(max_length=50, null=True, blank=True)
    
    # Quality metrics - UPDATED WITH EXACT COLUMNS
    rebuffering_ratio = models.FloatField(null=True, blank=True)
    connection_induced_rebuffering_ratio = models.FloatField(null=True, blank=True)
    avg_peak_bitrate = models.FloatField(null=True, blank=True)
    avg_average_bitrate = models.FloatField(null=True, blank=True)
    average_framerate = models.FloatField(null=True, blank=True)
    starting_bitrate = models.FloatField(null=True, blank=True)
    bitrate_switches = models.IntegerField(null=True, blank=True)
    
    # Failure indicators - EXACT COLUMN NAMES
    video_start_failure = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    exit_before_video_starts = models.CharField(max_length=50, null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = "sessions"
        indexes = [
            models.Index(fields=['session_id']),
            models.Index(fields=['viewer_id']),
            models.Index(fields=['session_start_time']),
            models.Index(fields=['asset_name']),
            models.Index(fields=['channel']),
            models.Index(fields=['status']),
            models.Index(fields=['video_start_failure']),
        ]
        verbose_name = "Session"
        verbose_name_plural = "Sessions"
    
    def __str__(self):
        return f"Session {self.session_id} - {self.asset_name}"

# ---------------- Advancetags Collection - COMPLETELY UPDATED ----------------
class Advancetags(models.Model):
    """Advanced tags/metadata collection in MongoDB - COMPLETE UPDATE FOR ALL COLUMNS"""
    # Primary identifier
    session_id = models.CharField(max_length=200, db_index=True)
    
    # Content information
    asset_name = models.CharField(max_length=500, null=True, blank=True)
    content_category = models.CharField(max_length=200, null=True, blank=True)
    channel = models.CharField(max_length=200, null=True, blank=True)
    
    # Browser information - EXACT COLUMN NAMES
    browser_name = models.CharField(max_length=200, null=True, blank=True)
    browser_version = models.CharField(max_length=100, null=True, blank=True)
    
    # Device information - EXACT COLUMN NAMES
    device_hardware_type = models.CharField(max_length=200, null=True, blank=True)
    device_manufacturer = models.CharField(max_length=200, null=True, blank=True)
    device_marketing_name = models.CharField(max_length=200, null=True, blank=True)
    device_model = models.CharField(max_length=200, null=True, blank=True)
    device_name = models.CharField(max_length=200, null=True, blank=True)
    device_operating_system = models.CharField(max_length=200, null=True, blank=True)
    device_operating_system_family = models.CharField(max_length=200, null=True, blank=True)
    device_operating_system_version = models.CharField(max_length=100, null=True, blank=True)
    
    # App and Player information - EXACT COLUMN NAMES
    app_name = models.CharField(max_length=200, null=True, blank=True)
    app_version = models.CharField(max_length=100, null=True, blank=True)
    player_framework_name = models.CharField(max_length=200, null=True, blank=True)
    player_framework_version = models.CharField(max_length=100, null=True, blank=True)
    
    # Network and CDN information - EXACT COLUMN NAMES
    cdn = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    last_cdn = models.CharField(max_length=100, null=True, blank=True)
    
    # Geographic information - EXACT COLUMN NAMES
    city = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    state = models.CharField(max_length=100, null=True, blank=True)
    country = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    address = models.TextField(null=True, blank=True)
    
    # Network details - EXACT COLUMN NAMES
    ip_address = models.GenericIPAddressField(null=True, blank=True, protocol='IPv4')
    ipv6_address = models.GenericIPAddressField(null=True, blank=True, protocol='IPv6')
    asn_name = models.CharField(max_length=200, null=True, blank=True)  # asnName from CSV
    isp_name = models.CharField(max_length=200, null=True, blank=True, db_index=True)  # ispName from CSV
    
    # Stream information - EXACT COLUMN NAMES
    stream_url = models.URLField(max_length=1000, null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = "advancetags"
        indexes = [
            models.Index(fields=['session_id']),
            models.Index(fields=['isp_name']),
            models.Index(fields=['city', 'country']),
            models.Index(fields=['cdn']),
            models.Index(fields=['device_operating_system']),
            models.Index(fields=['browser_name']),
        ]
        verbose_name = "Advanced Tag"
        verbose_name_plural = "Advanced Tags"
    
    def __str__(self):
        return f"Advancetags {self.session_id} - {self.isp_name}"

# ---------------- Tickets Collection - SAME AS BEFORE ----------------
class Ticket(models.Model):
    """Ticket collection in MongoDB - NO CHANGES NEEDED"""
    
    # Status choices
    STATUS_CHOICES = [
        ('new', _('New')),
        ('in_progress', _('In Progress')),
        ('resolved', _('Resolved')),
        ('closed', _('Closed')),
        ('on_hold', _('On Hold')),
    ]
    
    # Priority choices
    PRIORITY_CHOICES = [
        ('low', _('Low')),
        ('medium', _('Medium')),
        ('high', _('High')),
        ('critical', _('Critical')),
    ]
    
    # Team choices
    TEAM_CHOICES = [
        ('technical', _('Technical Team')),
        ('network', _('Network Team')),
        ('content', _('Content Team')),
        ('customer_service', _('Customer Service')),
    ]
    
    # Ticket identification
    ticket_id = models.CharField(max_length=100, unique=True, db_index=True)
    
    # Related data
    viewer_id = models.CharField(max_length=200, null=True, blank=True, db_index=True)
    session_id = models.CharField(max_length=200, null=True, blank=True, db_index=True)
    
    # Ticket details
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default='medium', db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new', db_index=True)
    assign_team = models.CharField(max_length=50, choices=TEAM_CHOICES, default='technical', db_index=True)
    
    # Issue information
    issue_type = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    description = models.TextField(null=True, blank=True)
    resolution_notes = models.TextField(null=True, blank=True)
    
    # JSON fields for complex data
    failure_details = models.JSONField(default=dict, null=True, blank=True)
    context_data = models.JSONField(default=dict, null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = "tickets"
        indexes = [
            models.Index(fields=['ticket_id']),
            models.Index(fields=['viewer_id']),
            models.Index(fields=['session_id']),
            models.Index(fields=['status']),
            models.Index(fields=['priority']),
            models.Index(fields=['assign_team']),
            models.Index(fields=['issue_type']),
            models.Index(fields=['created_at']),
        ]
        verbose_name = "Ticket"
        verbose_name_plural = "Tickets"
    
    def __str__(self):
        return f"Ticket {self.ticket_id} - {self.status} - {self.priority}"
    
    def save(self, *args, **kwargs):
        """Override save to auto-generate ticket_id if not provided"""
        if not self.ticket_id:
            # Generate unique ticket ID
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            self.ticket_id = f"TKT_{timestamp}_{str(uuid.uuid4())[:8]}"
        
        # Set resolved_at when status changes to resolved
        if self.status == 'resolved' and not self.resolved_at:
            self.resolved_at = timezone.now()
        
        super().save(*args, **kwargs)

# ---------------- Data Ingestion Log - SAME AS BEFORE ----------------
class DataIngestionLog(models.Model):
    """Log data ingestion activities"""
    ingestion_id = models.CharField(max_length=100, unique=True)
    source_type = models.CharField(max_length=50)  # 'file_upload' or 'mongodb_ingestion'
    status = models.CharField(max_length=20)  # 'success', 'partial', 'failed'
    
    # Statistics
    records_processed = models.IntegerField(default=0)
    kpi_records = models.IntegerField(default=0)
    session_records = models.IntegerField(default=0)
    advancetags_records = models.IntegerField(default=0)
    tickets_generated = models.IntegerField(default=0)
    
    # Details
    error_details = models.TextField(null=True, blank=True)
    processing_time_seconds = models.FloatField(null=True, blank=True)
    
    # Timestamps
    started_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = "data_ingestion_logs"
        verbose_name = "Data Ingestion Log"
        verbose_name_plural = "Data Ingestion Logs"
    
    def __str__(self):
        return f"Ingestion {self.ingestion_id} - {self.status}"
