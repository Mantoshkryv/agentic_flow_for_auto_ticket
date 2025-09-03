from django.db import models

# Create your models here.
from django.db import models

class Ticket(models.Model):
    session_id = models.CharField(max_length=128, null=True, blank=True)
    asset_name = models.CharField(max_length=256, null=True, blank=True)
    root_cause = models.CharField(max_length=200, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    evidence = models.TextField(null=True, blank=True)
    assign_team = models.CharField(max_length=100, null=True, blank=True)
    ticket_text = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.root_cause} ({self.session_id})"
