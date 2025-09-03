from django.shortcuts import render

# Create your views here.
import io
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadFilesForm
from .ticket_engine import AutoTicketMVP
from .models import Ticket
from django.utils import timezone

def ticket_dashboard(request):
    form = UploadFilesForm()
    tickets = []
    df_info = None

    if request.method == "POST":
        form = UploadFilesForm(request.POST, request.FILES)
        if form.is_valid():
            # Read uploaded files into DataFrames
            session_file = form.cleaned_data["session_file"]
            kpi_file = form.cleaned_data["kpi_file"]
            meta_file = form.cleaned_data["meta_file"]

            try:
                df_sess = pd.read_excel(session_file)
            except Exception as e:
                return render(request, "a/dashboard.html", {"form": form, "error": f"Error reading session file: {e}"})

            try:
                df_kpi = pd.read_excel(kpi_file)
            except Exception as e:
                return render(request, "a/dashboard.html", {"form": form, "error": f"Error reading KPI file: {e}"})

            try:
                df_meta = pd.read_excel(meta_file)
            except Exception as e:
                return render(request, "a/dashboard.html", {"form": form, "error": f"Error reading metadata file: {e}"})

            # Ensure required columns exist (best-effort)
            # Merge session + metadata
            try:
                df_full = df_sess.merge(
                    df_meta,
                    how="left",
                    left_on=["Session ID", "Asset Name"],
                    right_on=["Session Id", "Asset Name"]
                )
            except Exception:
                # fallback: try merging only on Session ID if Asset Name mismatch
                try:
                    df_full = df_sess.merge(df_meta, how="left", left_on=["Session ID"], right_on=["Session Id"])
                except Exception as e:
                    return render(request, "a/dashboard.html", {"form": form, "error": f"Merge failed: {e}"})

            # Run ticket engine
            engine = AutoTicketMVP(df_full, df_kpi)
            tickets = engine.process()

            # Save tickets to DB
            saved_count = 0
            for t in tickets:
                # Each ticket is a string block — try to extract session_id and asset_name from engine outputs if possible.
                # The engine returns full text; we have engine.process returning ticket strings
                # We also try to parse some fields using simple heuristics.
                session_id = None
                asset_name = None
                # try parse "Session {id}" from title line
                try:
                    # find 'Session ' token
                    first_line = t.splitlines()[1] if len(t.splitlines()) > 1 else t.splitlines()[0]
                    # look for "(Session <id>)"
                    import re
                    m = re.search(r"Session\s+([^\)\s]+)", t)
                    if m:
                        session_id = m.group(1)
                    m2 = re.search(r"on\s+(.*?)\s+\(Session", t)
                    if m2:
                        asset_name = m2.group(1).strip()
                except Exception:
                    pass

                ticket_obj = Ticket.objects.create(
                    session_id=session_id,
                    asset_name=asset_name,
                    root_cause="",     # optional: populate by parsing if you want
                    confidence=None,
                    evidence="",
                    assign_team="",
                    ticket_text=t,
                    created_at=timezone.now()
                )
                saved_count += 1

            df_info = {
                "sessions": df_sess.shape,
                "kpi": df_kpi.shape,
                "meta": df_meta.shape,
                "tickets_generated": len(tickets),
                "tickets_saved": saved_count
            }

            # Download CSV if requested
            if "download" in request.POST:
                df_tickets = pd.DataFrame({"ticket_text": tickets})
                buffer = io.StringIO()
                df_tickets.to_csv(buffer, index=False)
                response = HttpResponse(buffer.getvalue(), content_type="text/csv")
                response["Content-Disposition"] = 'attachment; filename="tickets.csv"'
                return response

    # show recent saved tickets
    recent_tickets = Ticket.objects.all().order_by("-created_at")[:200]
    return render(request, "a/dashboard.html", {
        "form": form,
        "tickets": tickets,
        "df_info": df_info,
        "recent_tickets": recent_tickets,
    })
