import pandas as pd
from datetime import timedelta
from dataclasses import dataclass
import textwrap
from collections import defaultdict
import bisect
import logging

logger = logging.getLogger(__name__)

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
        """
        Initialize AutoTicket engine with session and KPI data
        
        Args:
            df_full: DataFrame with session data
            df_kpi: DataFrame with KPI data
        """
        try:
            self.df = df_full.copy()
            
            # Handle datetime conversion with error checking
            if "Session Start Time" in self.df.columns:
                self.df["Session Start Time"] = pd.to_datetime(
                    self.df["Session Start Time"], errors="coerce"
                )
            elif "session_start_time" in self.df.columns:
                self.df["Session Start Time"] = pd.to_datetime(
                    self.df["session_start_time"], errors="coerce"
                )
            else:
                raise ValueError("No session start time column found")

            # Prepare KPI dataframe
            self.df_kpi = df_kpi.copy()
            
            # Handle KPI timestamp column
            timestamp_col = None
            for col in ["timestamp", "Timestamp"]:
                if col in self.df_kpi.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                # If no timestamp column found, use first column and skip first row
                timestamp_col = self.df_kpi.columns[0]
                self.df_kpi["Timestamp"] = pd.to_datetime(
                    self.df_kpi[timestamp_col].iloc[1:], errors="coerce"
                )
            else:
                self.df_kpi["Timestamp"] = pd.to_datetime(
                    self.df_kpi[timestamp_col], errors="coerce"
                )

            # Convert timezone-aware to naive if needed
            if hasattr(self.df_kpi["Timestamp"].dtype, 'tz') and self.df_kpi["Timestamp"].dt.tz is not None:
                self.df_kpi["Timestamp"] = self.df_kpi["Timestamp"].dt.tz_convert(None)

            # Merge KPI context using merge_asof
            self.df = self.df.sort_values("Session Start Time")
            self.df_kpi = self.df_kpi.dropna(subset=["Timestamp"]).sort_values("Timestamp")

            if not self.df_kpi.empty and not self.df.empty:
                self.df = pd.merge_asof(
                    self.df,
                    self.df_kpi,
                    left_on="Session Start Time",
                    right_on="Timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta("5min")
                )

            # Define failure flag with proper column handling
            status_col = "Status" if "Status" in self.df.columns else "status"
            vsf_col = "Video Start Failure" if "Video Start Failure" in self.df.columns else "video_start_failure"
            
            self.df["is_failure"] = (
                self.df[status_col].isin(FAILURE_STATES) |
                (self.df[vsf_col].astype(str).str.lower() == "true")
            )

            # Pre-build correlation index: (ISP, CDN, City) -> sorted timestamps
            self.failures_by_group = defaultdict(list)
            failure_df = self.df[self.df["is_failure"]]
            
            for _, row in failure_df.iterrows():
                isp = row.get("ispName", row.get("isp_name", "NA"))
                cdn = row.get("cdn", row.get("CDN", "NA"))
                city = row.get("city", row.get("City", "NA"))
                key = (isp, cdn, city)
                self.failures_by_group[key].append(row["Session Start Time"])
                
            for k in self.failures_by_group:
                self.failures_by_group[k].sort()
                
        except Exception as e:
            logger.error(f"Error initializing AutoTicketMVP: {str(e)}")
            raise

    def correlate(self, isp, cdn, city, ts, window=5):
        """Find failures around given timestamp in same ISP/CDN/City group"""
        if pd.isna(ts):
            return 0
            
        times = self.failures_by_group.get((isp, cdn, city), [])
        if not times:
            return 0
            
        t0 = ts - timedelta(minutes=window)
        t1 = ts + timedelta(minutes=window)
        
        left = bisect.bisect_left(times, t0)
        right = bisect.bisect_right(times, t1)
        return right - left

    def diagnose(self, row):
        """Diagnose failure based on row data"""
        try:
            status = row.get("Status", row.get("status", ""))
            
            # Handle bitrate checking
            starting_bitrate = row.get("Starting Bitrate", row.get("starting_bitrate", ""))
            has_zero_bitrate = str(starting_bitrate).strip().startswith("0")
            
            # Get location/network info with fallbacks
            isp = row.get("ispName", row.get("isp_name", "NA"))
            cdn = row.get("cdn", row.get("CDN", "NA"))
            city = row.get("city", row.get("City", "NA"))
            ts = row.get("Session Start Time", row.get("session_start_time"))

            # Rule 1: Business failures
            if status in {"VSF-B", "EBVS"}:
                return Diagnosis(
                    "Entitlement/Auth Issue", 
                    0.8, 
                    f"Status={status}", 
                    "Platform Auth"
                )

            # Rule 2: Technical failures with 0 bitrate
            if status == "VSF-T" and has_zero_bitrate:
                count = self.correlate(isp, cdn, city, ts)
                if count > 3:
                    return Diagnosis(
                        "CDN Issue", 
                        0.9, 
                        f"{count} failures from CDN={cdn} in {city}", 
                        "CDN"
                    )
                return Diagnosis(
                    "CDN/Manifest Issue", 
                    0.8, 
                    "Starting bitrate=0 bps", 
                    "CDN"
                )

            # Rule 3: ISP clustering
            count = self.correlate(isp, cdn, city, ts)
            if count > 5:
                return Diagnosis(
                    "ISP/Network Issue", 
                    0.85, 
                    f"{count} failures from ISP={isp} in {city}", 
                    "NOC"
                )

            # Rule 4: KPI anomaly context
            spi = row.get("Streaming Performance Index", row.get("streaming_performance_index", 100))
            if pd.notna(spi) and float(spi) < 70:
                return Diagnosis(
                    "Channel Health Degraded", 
                    0.8, 
                    f"SPI={spi}", 
                    "NOC"
                )

            return Diagnosis(
                "Technical Investigation Needed", 
                0.7, 
                "No clear indicators", 
                "NOC"
            )
            
        except Exception as e:
            logger.error(f"Error in diagnosis: {str(e)}")
            return Diagnosis(
                "Diagnosis Error", 
                0.5, 
                f"Error during diagnosis: {str(e)}", 
                "NOC"
            )

    def process(self):
        """Generate all failure tickets"""
        try:
            if "is_failure" not in self.df.columns:
                logger.warning("No failure flag found in data")
                return []
                
            failures = self.df[self.df["is_failure"]]
            tickets = []
            
            for _, row in failures.iterrows():
                diag = self.diagnose(row)
                if diag.confidence >= 0.7:
                    ticket_text = self.build_ticket_text(row, diag)
                    tickets.append(ticket_text)
                    
            return tickets
            
        except Exception as e:
            logger.error(f"Error processing tickets: {str(e)}")
            return []

    def build_ticket_text(self, row, diag: Diagnosis):
        """Build failure ticket in required format"""
        try:
            # Extract key fields
            session_id = str(row.get("Session ID", row.get("session_id", "N/A")))
            channel = row.get("Channel", row.get("channel", row.get("Asset Name", "N/A")))
            ts = row.get("Session Start Time", row.get("session_start_time"))

            # Format time: "Aug 29 2025, 19:28:30"
            if pd.notna(ts):
                try:
                    ts_str = pd.to_datetime(ts).strftime("%b %d %Y, %H:%M:%S")
                except Exception:
                    ts_str = str(ts)
            else:
                ts_str = "N/A"

            # Build ticket in required format
            return textwrap.dedent(f"""
            === NEW FAILURE TICKET === 
            TITLE: [VSF] [{diag.root_cause}] for User {session_id} on {channel}

            BODY:
            - Viewer ID: {session_id}
            - Impacted Channel: {channel}
            - Time of Failure: {ts_str}
            - Auto-Diagnosis: {diag.root_cause} (Confidence: {diag.confidence})
            - Evidence: {diag.evidence}
            - Deep Link: https://example.com/session/{session_id}
            - Assign to: {diag.assign_team} Team
            """).strip()

        except Exception as e:
            logger.error(f"Error building ticket text: {str(e)}")
            return f"ERROR: Could not generate ticket for session {row.get('Session ID', 'Unknown')}: {str(e)}"
