# tg/views.py
from django.shortcuts import render
from django.http import HttpResponse
from .mongo import MongoService
from .operation.ticket_engine import AutoTicketMVP
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
            # Fetch data from MongoDB
            df_sessions = mongo_service.fetch_sessions()
            df_kpi = mongo_service.fetch_kpi()
            df_meta = mongo_service.fetch_advancetags()

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
                # Run ticket engine
                engine = AutoTicketMVP(df_sessions, df_kpi)
                tickets = engine.process()
                df_info["tickets_generated"] = len(tickets)

                tickets_to_save = []
                for t in tickets:
                    try:
                        # Try structured parsing
                        session_id = t.split("Session ")[1].split(")")[0]
                        asset_name = t.split("on ")[1].split(" (Session")[0]
                        root_cause = t.split("[")[2].split("]")[0]
                        tickets_to_save.append({
                            "session_id": session_id,
                            "asset_name": asset_name,
                            "root_cause": root_cause,
                            "ticket_text": t,
                            "created_at": datetime.now()
                        })
                    except Exception:
                        # Fallback: store raw text only
                        tickets_to_save.append({
                            "ticket_text": t,
                            "created_at": datetime.now()
                        })

                if tickets_to_save:
                    mongo_service.db["tickets"].insert_many(tickets_to_save)
                    df_info["tickets_saved"] = len(tickets_to_save)

            # CSV download
            if "download" in request.POST:
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["Session ID", "Asset Name", "Root Cause", "Ticket Text", "Created At"])
                for t in tickets_to_save:
                    writer.writerow([
                        t.get("session_id", ""),
                        t.get("asset_name", ""),
                        t.get("root_cause", ""),
                        t["ticket_text"],
                        t["created_at"]
                    ])
                response = HttpResponse(output.getvalue(), content_type="text/csv")
                response["Content-Disposition"] = f"attachment; filename=tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                return response

        except Exception as e:
            error = f"Error fetching data from MongoDB: {str(e)}"

    # Show recent tickets
    recent_tickets = list(mongo_service.db["tickets"].find().sort("created_at", -1).limit(10))

    return render(request, "tg/dashboard.html", {
        "error": error,
        "tickets": tickets,
        "df_info": df_info,
        "recent_tickets": recent_tickets
    })
