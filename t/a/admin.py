from django.contrib import admin

# Register your models here.

from .models import Ticket

@admin.register(Ticket)
class TicketAdmin(admin.ModelAdmin):
    list_display = ("created_at", "session_id", "asset_name", "root_cause")
    search_fields = ("session_id", "asset_name", "root_cause", "ticket_text")
    readonly_fields = ("created_at",)
