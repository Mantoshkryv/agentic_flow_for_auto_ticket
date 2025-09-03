# ticket_engine.py
import pandas as pd
from datetime import timedelta
from dataclasses import dataclass
import textwrap
from collections import defaultdict
import bisect

# Failure status codes
FAILURE_STATES = {"VSF-T", "VSF-B", "EBVS"}

@dataclass
class Diagnosis:
    root_cause: str
    confidence: float
    evidence: str
    assign_team: str

class AutoTicketMVP:
    def __init__(self, df_full: pd.DataFrame, df_kpi: pd.DataFrame):
        self.df = df_full.copy()
        self.df["Session Start Time"] = pd.to_datetime(self.df["Session Start Time"], errors="coerce")

        # Prepare KPI dataframe
        self.df_kpi = df_kpi.copy()
        kpi_col = self.df_kpi.columns[0]
        self.df_kpi["Timestamp"] = pd.to_datetime(self.df_kpi[kpi_col].iloc[1:], errors="coerce")

        # Convert timezone-aware to naive if needed
        if self.df_kpi["Timestamp"].dt.tz is not None:
            self.df_kpi["Timestamp"] = self.df_kpi["Timestamp"].dt.tz_convert(None)

        # --- Merge KPI context using merge_asof ---
        self.df = self.df.sort_values("Session Start Time")
        self.df_kpi = self.df_kpi.dropna(subset=["Timestamp"]).sort_values("Timestamp")

        self.df = pd.merge_asof(
            self.df,
            self.df_kpi,
            left_on="Session Start Time",
            right_on="Timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("5min")
        )

        # Define failure flag
        self.df["is_failure"] = (
            self.df["Status"].isin(FAILURE_STATES) |
            (self.df["Video Start Failure"].astype(str).str.lower() == "true")
        )

        # Pre-build correlation index: (ISP, CDN, City) -> sorted timestamps
        self.failures_by_group = defaultdict(list)
        for _, row in self.df[self.df["is_failure"]].iterrows():
            key = (row.get("ispName", "NA"), row.get("cdn", "NA"), row.get("city", "NA"))
            self.failures_by_group[key].append(row["Session Start Time"])
        for k in self.failures_by_group:
            self.failures_by_group[k].sort()

    def correlate(self, isp, cdn, city, ts, window=5):
        """Find failures around given timestamp in same ISP/CDN/City group"""
        times = self.failures_by_group.get((isp, cdn, city), [])
        if not times:
            return 0
        t0, t1 = ts - timedelta(minutes=window), ts + timedelta(minutes=window)
        left = bisect.bisect_left(times, t0)
        right = bisect.bisect_right(times, t1)
        return right - left

    def diagnose(self, row):
        status = row["Status"]
        has_zero_bitrate = str(row.get("Starting Bitrate", "")).strip().startswith("0")
        isp, cdn, city, ts = (
            row.get("ispName", "NA"),
            row.get("cdn", "NA"),
            row.get("city", "NA"),
            row["Session Start Time"],
        )

        # Rule 1: Business failures
        if status in {"VSF-B", "EBVS"}:
            return Diagnosis("Entitlement/Auth Issue", 0.8, f"Status={status}", "Platform Auth")

        # Rule 2: Tech with 0 bitrate
        if status == "VSF-T" and has_zero_bitrate:
            count = self.correlate(isp, cdn, city, ts)
            if count > 3:
                return Diagnosis("CDN Issue", 0.9, f"{count} failures from CDN={cdn} in {city}", "CDN")
            return Diagnosis("CDN/Manifest Issue", 0.8, "Starting bitrate=0 bps", "CDN")

        # Rule 3: ISP clustering
        count = self.correlate(isp, cdn, city, ts)
        if count > 5:
            return Diagnosis("ISP/Network Issue", 0.85, f"{count} failures from ISP={isp} in {city}", "NOC")

        # Rule 4: KPI anomaly context
        if row.get("Streaming Performance Index", 100) < 70:
            return Diagnosis("Channel Health Degraded", 0.8, f"SPI={row['Streaming Performance Index']}", "NOC")

        return Diagnosis("Technical Investigation Needed", 0.7, "No clear indicators", "NOC")

    def process(self):
        """Generate all failure tickets"""
        failures = self.df[self.df["is_failure"]]
        tickets = []
        for _, row in failures.iterrows():
            diag = self.diagnose(row)
            if diag.confidence >= 0.7:
                tickets.append(self.build_ticket_text(row, diag))
        return tickets

    def build_ticket_text(self, row, diag: Diagnosis):
        """Build ticket description"""
        return textwrap.dedent(f"""
        === NEW FAILURE TICKET ===
        TITLE: [VSF] [{diag.root_cause}] on {row['Asset Name']} (Session {row['Session ID']})

        BODY:
        - Session ID: {row['Session ID']}
        - Channel: {row.get('Channel','NA')}
        - Content Category: {row.get('Content Category','NA')}
        - Device: {row.get('Device Marketing Name','NA')} ({row.get('Device Operating System','NA')} {row.get('Device Operating System Version','')})
        - Browser: {row.get('Browser Name','NA')} {row.get('Browser Version','')}
        - ISP: {row.get('ispName','NA')} (ASN: {row.get('asnName','NA')}), City: {row.get('city','NA')}, Country: {row.get('country','NA')}
        - CDN: {row.get('cdn','NA')} (Last CDN: {row.get('Last CDN','NA')})
        - Session Metrics: StartBitrate={row.get('Starting Bitrate','NA')}, Rebuffer={row.get('Rebuffering Ratio','NA')}, Avg. Peak Bitrate={row.get('Avg. Peak Bitrate','NA')}
        - Channel KPIs: SPI={row.get('Streaming Performance Index','NA')}, Plays={row.get('Plays','NA')}, Rebuffering={row.get('Rebuffering Ratio(%)','NA')}
        - Auto-Diagnosis: {diag.root_cause} (Confidence: {diag.confidence:.2f})
        - Evidence: {diag.evidence}
        - Assign to: {diag.assign_team} Team
        """).strip()
