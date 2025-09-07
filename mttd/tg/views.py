# tg/views.py
from django.shortcuts import render
from django.http import HttpResponse
from .mongo import MongoService
from .operation.ticket_engine import AutoTicketMVP
import pandas as pd
import csv
import io
from datetime import datetime


def dashboard_view(request):
    mongo_service = MongoService()
    error = None
    tickets = []
    df_info = None

    if request.method == "POST":
        try:
            # Fetch data from MongoDB using correct method names
            df_sessions = mongo_service.fetch_sessions()
            df_kpi = mongo_service.fetch_kpi()
            df_meta = mongo_service.fetch_advancetags()

            # Store info for dashboard
            df_info = {
                "sessions": len(df_sessions),
                "kpi": len(df_kpi),
                "meta": len(df_meta),
                "tickets_generated": 0,
                "tickets_saved": 0
            }

            if df_sessions.empty or df_kpi.empty:
                error = "Session or KPI data is empty in MongoDB."
            else:
                # Initialize ticket engine (assuming df_meta is optional)
                engine = AutoTicketMVP(df_sessions, df_kpi)
                tickets = engine.process()
                df_info["tickets_generated"] = len(tickets)

                # Optional: save tickets to MongoDB
                tickets_to_save = []
                for t in tickets:
                    try:
                        tickets_to_save.append({
                            "session_id": t.split("Session ")[1].split(")")[0],
                            "asset_name": t.split("on ")[1].split(" (Session")[0],
                            "root_cause": t.split("[")[2].split("]")[0],
                            "ticket_text": t,
                            "created_at": datetime.now()
                        })
                    except Exception:
                        continue  # skip parsing errors
                if tickets_to_save:
                    mongo_service.db["tickets"].insert_many(tickets_to_save)
                    df_info["tickets_saved"] = len(tickets_to_save)

            # Handle CSV download
            if "download" in request.POST:
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["Session ID", "Asset Name", "Root Cause", "Ticket Text", "Created At"])
                for t in tickets_to_save:
                    writer.writerow([t["session_id"], t["asset_name"], t["root_cause"], t["ticket_text"], t["created_at"]])
                response = HttpResponse(output.getvalue(), content_type="text/csv")
                response["Content-Disposition"] = f"attachment; filename=tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                return response

        except Exception as e:
            error = f"Error fetching data from MongoDB: {str(e)}"

    # Fetch recent tickets to show in dashboard
    recent_tickets = list(mongo_service.db["tickets"].find().sort("created_at", -1).limit(10))

    return render(request, "tg/dashboard.html", {
        "error": error,
        "tickets": tickets,
        "df_info": df_info,
        "recent_tickets": recent_tickets
    })
