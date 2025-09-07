from django.db import models
from django.utils import timezone
import uuid
from django_mongodb_backend.fields import ObjectIdField
from django.utils.translation import gettext_lazy as _

# ---------------- KPI ----------------
class KPI(models.Model):
    # Let Django/MongoDB auto-generate the primary key
    timestamp = models.DateTimeField(db_index=True)
    plays = models.IntegerField(default=0)
    playing_time_mins = models.FloatField(null=True, blank=True)
    streaming_performance_index = models.FloatField(null=True, blank=True)

    # Failures
    video_start_failures_technical = models.IntegerField(default=0)
    video_start_failures_business = models.IntegerField(default=0)
    exit_before_video_starts = models.IntegerField(default=0)
    video_playback_failures_technical = models.IntegerField(default=0)
    video_playback_failures_business = models.IntegerField(default=0)

    # Streaming QoE metrics
    video_start_time_sec = models.FloatField(null=True, blank=True)
    rebuffering_ratio_pct = models.FloatField(null=True, blank=True)
    connection_induced_rebuffering_pct = models.FloatField(null=True, blank=True)
    video_restart_time_sec = models.FloatField(null=True, blank=True)
    avg_peak_bitrate_mbps = models.FloatField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "kpi_data"
        ordering = ['-timestamp']


# ---------------- SESSION ----------------
class Session(models.Model):
    # Let Django/MongoDB auto-generate the primary key
    session_id = models.CharField(max_length=200, unique=True, db_index=True)
    session_end_time = models.DateTimeField(null=True, blank=True)
    playing_time = models.FloatField(null=True, blank=True)
    asset_name = models.CharField(max_length=300, null=True, blank=True, db_index=True)

    # Session status fields
    ended_session = models.BooleanField(default=False)
    impacted_session = models.BooleanField(default=False)
    exit_before_video_starts = models.BooleanField(default=False)

    # Video metrics
    video_start_time = models.DateTimeField(null=True, blank=True)
    rebuffering_ratio = models.FloatField(null=True, blank=True)
    connection_induced_rebuffering_ratio = models.FloatField(null=True, blank=True)
    total_video_restart_time = models.FloatField(null=True, blank=True)

    # Bitrate metrics
    avg_peak_bitrate = models.FloatField(null=True, blank=True)
    avg_average_bitrate = models.FloatField(null=True, blank=True)
    average_framerate = models.FloatField(null=True, blank=True)
    starting_bitrate = models.FloatField(null=True, blank=True)
    bitrate_switches = models.IntegerField(null=True, blank=True)

    # Session details
    channel = models.CharField(max_length=200, null=True, blank=True, db_index=True)
    ended_status = models.CharField(max_length=200, null=True, blank=True)
    session_start_time = models.DateTimeField(db_index=True)
    status = models.CharField(max_length=50, db_index=True)
    video_start_failure = models.BooleanField(default=False, db_index=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "sessions"
        ordering = ['-session_start_time']


# ---------------- ADVANCETAGS ----------------
class Advancetags(models.Model):
    # Let Django/MongoDB auto-generate the primary key
    session_id = models.CharField(max_length=200, db_index=True)
    asset_name = models.CharField(max_length=300, null=True, blank=True)
    content_category = models.CharField(max_length=200, null=True, blank=True)

    # Device/Browser info
    browser_name = models.CharField(max_length=100, null=True, blank=True)
    browser_version = models.CharField(max_length=100, null=True, blank=True)
    device_hardware_type = models.CharField(max_length=100, null=True, blank=True)
    device_manufacturer = models.CharField(max_length=100, null=True, blank=True)
    device_marketing_name = models.CharField(max_length=200, null=True, blank=True)
    device_model = models.CharField(max_length=100, null=True, blank=True)
    device_name = models.CharField(max_length=200, null=True, blank=True)
    device_os = models.CharField(max_length=100, null=True, blank=True)
    device_os_family = models.CharField(max_length=100, null=True, blank=True)
    device_os_version = models.CharField(max_length=100, null=True, blank=True)

    # App / Player
    app_name = models.CharField(max_length=100, null=True, blank=True)
    app_version = models.CharField(max_length=100, null=True, blank=True)
    player_framework_name = models.CharField(max_length=100, null=True, blank=True)
    player_framework_version = models.CharField(max_length=100, null=True, blank=True)

    # CDN/Network
    last_cdn = models.CharField(max_length=200, null=True, blank=True)
    cdn = models.CharField(max_length=200, null=True, blank=True, db_index=True)
    channel = models.CharField(max_length=200, null=True, blank=True)
    stream_url = models.TextField(null=True, blank=True)

    # Geo/IP
    city = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    state = models.CharField(max_length=100, null=True, blank=True)
    country = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    address = models.CharField(max_length=300, null=True, blank=True)
    ip = models.GenericIPAddressField(null=True, blank=True)
    ipv6 = models.GenericIPAddressField(null=True, blank=True)
    asname = models.CharField(max_length=200, null=True, blank=True)
    isp_name = models.CharField(max_length=200, null=True, blank=True, db_index=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "advancetags"


# ---------------- TICKET ----------------
class Ticket(models.Model):
    # Let Django/MongoDB auto-generate the primary key
    ticket_id = models.CharField(max_length=100, unique=True, db_index=True)

    # Keep relation, remove duplicate session_id
    session = models.ForeignKey(Session, on_delete=models.SET_NULL, null=True, blank=True)

    viewer_id = models.CharField(max_length=200, null=True, blank=True)
    channel = models.CharField(max_length=200, null=True, blank=True, db_index=True)
    asset_name = models.CharField(max_length=300, null=True, blank=True)
    failure_code = models.CharField(max_length=50, null=True, blank=True, db_index=True)
    root_cause = models.CharField(max_length=200, null=True, blank=True, db_index=True)
    confidence = models.CharField(max_length=50, null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    failure_time = models.DateTimeField(null=True, blank=True)
    evidence = models.TextField(null=True, blank=True)
    deep_link = models.URLField(null=True, blank=True)
    assign_team = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    ticket_text = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=50, default='new', db_index=True)
    priority = models.CharField(max_length=20, default='medium', db_index=True)
    processed = models.BooleanField(default=False, db_index=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    assigned_to = models.CharField(max_length=100, null=True, blank=True)
    resolved_by = models.CharField(max_length=100, null=True, blank=True)
    resolution_notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "tickets"
        ordering = ['-created_at']

    def save(self, *args, **kwargs):
        if not self.ticket_id:
            self.ticket_id = f"TICKET-{timezone.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Ticket [{self.ticket_id}] {self.failure_code} - {self.root_cause}"