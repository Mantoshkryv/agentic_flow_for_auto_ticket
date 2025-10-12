# ticket_engine.py - FIXED: Eliminated duplicate _find_column() methods

"""
Enhanced Viewer-Centric Ticket Engine for Video Start Failures
==============================================================

FIXED: Replaced duplicate _find_column() methods with shared module-level helper
"""

import pandas as pd
import textwrap
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import hashlib
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# SHARED HELPER FUNCTIONS - MODULE LEVEL
# ============================================================================

def _find_column_helper(columns: List[str], possible_names: List[str]) -> Optional[str]:
    """
    Shared column finder helper - used by multiple classes
    Finds column name from variations (case-insensitive)
    """
    for name in possible_names:
        for col in columns:
            if str(col).lower() == name.lower():
                return col
    return None

# ============================================================================
# ENHANCED DIAGNOSIS CLASS
# ============================================================================

class Diagnosis:
    """Enhanced diagnosis with multi-layer analysis"""
    def __init__(self, root_cause: str, confidence: float, evidence: str, assign_team: str,
                 severity_score: int, user_behavior: Dict[str, Any] = None,
                 temporal_analysis: Dict[str, Any] = None,
                 geographic_analysis: Dict[str, Any] = None):
        self.root_cause = root_cause
        self.confidence = confidence  # Float 0.0-1.0
        self.evidence = evidence
        self.assign_team = assign_team
        self.severity_score = severity_score  # 0-10 scale
        self.user_behavior = user_behavior or {}
        self.temporal_analysis = temporal_analysis or {}
        self.geographic_analysis = geographic_analysis or {}

    def __repr__(self):
        return (f"Diagnosis(root_cause={self.root_cause}, confidence={self.confidence:.2f}, "
                f"severity={self.severity_score})")

# ============================================================================
# STREAMING HEALTH MONITOR - REAL-TIME ALERTING
# ============================================================================

class StreamingHealthMonitor:
    """Real-time alerting and triage system"""
    
    def __init__(self):
        self.failure_thresholds = {
            'city_isp_failure_rate': 0.15,
            'concurrent_failures': 5,
            'user_failure_streak': 3
        }
    
    def monitor_stream(self, new_session: pd.Series, historical_data: pd.DataFrame,
                      advancetags: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Monitor for real-time patterns"""
        alerts = []
        
        # Detect emerging outage
        if self._detect_emerging_outage(new_session, historical_data, advancetags):
            alerts.append({
                'type': 'OUTAGE_ALERT',
                'severity': 'CRITICAL',
                'affected_areas': self._get_affected_regions(new_session, historical_data, advancetags),
                'likely_root_cause': 'CDN_OUTAGE'
            })
        
        # Detect user experience degradation
        viewer_id = new_session.get('Viewer ID', 'unknown')
        if self._detect_user_degradation(viewer_id, historical_data):
            alerts.append({
                'type': 'USER_EXPERIENCE_DEGRADATION',
                'viewer_id': viewer_id,
                'trend': 'deteriorating',
                'suggested_action': 'proactive_support_outreach'
            })
        
        return alerts
    
    def _detect_emerging_outage(self, session: pd.Series, historical: pd.DataFrame,
                               advancetags: pd.DataFrame = None) -> bool:
        """Detect if this is part of an emerging outage"""
        if historical.empty or advancetags is None or advancetags.empty:
            return False
        
        session_time = session.get('Session Start Time')
        if not session_time:
            return False
        
        recent = historical[
            historical['Session Start Time'] >= (session_time - timedelta(minutes=5))
        ]
        
        failures = recent[recent['Status'].isin(['VSF-T', 'VSF-B', 'EBVS'])]
        
        return len(failures) >= self.failure_thresholds['concurrent_failures']
    
    def _detect_user_degradation(self, viewer_id: str, historical: pd.DataFrame) -> bool:
        """Detect if user is experiencing service degradation - FIXED with shared helper"""
        if historical.empty:
            return False

        # ✅ FIXED: Use shared module-level helper
        viewer_col = _find_column_helper(historical.columns, [
            'viewer_id', 'Viewer ID', 'ViewerID', 'user_id'
        ])

        if not viewer_col:
            logger.warning("No viewer_id column found for degradation detection")
            return False

        try:
            viewer_sessions = historical[historical[viewer_col] == viewer_id].tail(10)

            if len(viewer_sessions) < 5:
                return False

            # ✅ FIXED: Use shared module-level helper
            status_col = _find_column_helper(viewer_sessions.columns, [
                'status', 'Status', 'Ended Status'
            ])

            if not status_col:
                return False

            recent_failures = viewer_sessions.tail(5)[status_col].isin(['VSF-T', 'VSF-B', 'EBVS']).sum()

            return recent_failures >= self.failure_thresholds['user_failure_streak']

        except Exception as e:
            logger.error(f"Error in degradation detection: {e}")
            return False
    
    def _get_affected_regions(self, session: pd.Series, historical: pd.DataFrame,
                             advancetags: pd.DataFrame = None) -> List[str]:
        """Get affected geographic regions"""
        return ['Multiple regions detected']

# ============================================================================
# ENHANCED VIEWER-CENTRIC TICKET ENGINE
# ============================================================================

class AutoTicketMVP:
    """
    Enhanced Viewer-Centric Ticket Engine
    
    Flow: viewer_id → session_ids → multi-layer diagnosis → tickets
    """
    
    def __init__(self, df_sessions: pd.DataFrame, df_kpi: pd.DataFrame = None,
                 df_advancetags: pd.DataFrame = None, target_channels: List[str] = None):
        """
        Initialize Enhanced Ticket Engine - FIXED: Session-first architecture
        """

        self.df_sessions = df_sessions if df_sessions is not None and not df_sessions.empty else pd.DataFrame()
        self.df_kpi = df_kpi if df_kpi is not None and not df_kpi.empty else pd.DataFrame()
        self.df_advancetags = df_advancetags if df_advancetags is not None and not df_advancetags.empty else pd.DataFrame()
        self.target_channels = target_channels or []

        # Initialize monitoring
        self.health_monitor = StreamingHealthMonitor()

        # Prepare data 
        self._prepare_session_data()

        # ✅ FIXED: Build lightweight viewer index (NOT pre-grouped profiles)
        self.viewer_index = self._build_viewer_index()

        logger.info(f"AutoTicketMVP initialized: {len(self.df_sessions)} sessions, "
                   f"{len(self.viewer_index)} viewers indexed")

    def _build_viewer_index(self) -> Dict[str, List[int]]:
        """
        Build lightweight viewer lookup index

        This replaces the heavy _build_all_viewer_profiles() method.
        Instead of pre-building complete profiles, we just create an index
        mapping viewer_id → [row_indices] for on-demand lookup.

        Returns:
            Dict mapping viewer_id to list of session row indices
            Example: {'viewer_123': [0, 5, 12], 'viewer_456': [1, 3, 8]}
        """
        if self.df_sessions.empty:
            logger.info("Empty sessions DataFrame - no viewer index created")
            return {}

        logger.info("=" * 70)
        logger.info("📇 BUILDING VIEWER INDEX (Lightweight)")
        logger.info("=" * 70)

        # Find viewer_id column using flexible matching
        viewer_col = None

        # First, check if already in standard format
        if 'viewer_id' in self.df_sessions.columns:
            viewer_col = 'viewer_id'
            logger.info("✅ Using standard column: 'viewer_id'")
        else:
            # Use helper to find column variations
            viewer_col = _find_column_helper(self.df_sessions.columns, [
                'Viewer ID', 'ViewerID', 'user_id', 'viewer_id'
            ])
            if viewer_col:
                logger.info(f"✅ Found viewer column: '{viewer_col}'")

        if not viewer_col:
            logger.warning("⚠️ No viewer_id column found - viewer analytics will be limited")
            logger.warning(f"Available columns: {list(self.df_sessions.columns)[:10]}")
            return {}

        # Build index: {viewer_id: [row_indices]}
        viewer_index = {}
        null_count = 0

        for idx, row in self.df_sessions.iterrows():
            viewer_id = row.get(viewer_col)

            # Handle null/NaN viewer IDs
            if pd.isna(viewer_id):
                null_count += 1
                continue
            
            viewer_id_str = str(viewer_id).strip()

            # Skip empty strings
            if not viewer_id_str or viewer_id_str.lower() in ['nan', 'none', 'null']:
                null_count += 1
                continue
            
            # Add to index
            if viewer_id_str not in viewer_index:
                viewer_index[viewer_id_str] = []
            viewer_index[viewer_id_str].append(idx)

        # Log statistics
        total_viewers = len(viewer_index)
        total_sessions = sum(len(indices) for indices in viewer_index.values())
        avg_sessions_per_viewer = total_sessions / total_viewers if total_viewers > 0 else 0

        logger.info(f"✅ Indexed {total_viewers} unique viewers")
        logger.info(f"📊 Total sessions indexed: {total_sessions}")
        logger.info(f"📊 Average sessions per viewer: {avg_sessions_per_viewer:.1f}")
        logger.info(f"⚠️ Null viewer_ids skipped: {null_count}")
        logger.info(f"📋 Sample viewer_ids: {list(viewer_index.keys())[:5]}")

        # Show distribution
        session_counts = [len(indices) for indices in viewer_index.values()]
        if session_counts:
            logger.info(f"📊 Session distribution - Min: {min(session_counts)}, "
                       f"Max: {max(session_counts)}, "
                       f"Median: {sorted(session_counts)[len(session_counts)//2]}")

        logger.info("=" * 70)

        return viewer_index

    def _get_viewer_context_for_session(self, session: pd.Series, viewer_id: str) -> Dict[str, Any]:
        """
        NEW METHOD: Get viewer context on-demand for a single session
        This replaces the need for pre-built viewer profiles
        
        Args:
            session: Current session data
            viewer_id: Viewer identifier (can be 'unknown')
        
        Returns:
            Dict containing viewer context with same structure as old profiles
        """
        # Handle missing/invalid viewer_id
        if not viewer_id or viewer_id == 'unknown' or pd.isna(viewer_id):
            return {
                'viewer_id': 'unknown',
                'type': 'new_user',
                'session_count': 0,
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'recent_sessions': pd.DataFrame(),
                'is_heavy_user': False,
                'avg_playing_time': 0,
                'last_10_sessions': pd.DataFrame(),
                'all_sessions': pd.DataFrame()
            }
        
        # Lookup viewer sessions using index
        if viewer_id not in self.viewer_index:
            # First-time viewer
            return {
                'viewer_id': viewer_id,
                'type': 'new_user',
                'session_count': 1,
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'recent_sessions': pd.DataFrame([session]),
                'is_heavy_user': False,
                'avg_playing_time': 0,
                'last_10_sessions': pd.DataFrame([session]),
                'all_sessions': pd.DataFrame([session])
            }
        
        # Get viewer's session indices from index
        session_indices = self.viewer_index[viewer_id]
        viewer_sessions = self.df_sessions.loc[session_indices]
        
        # Build viewer profile on-demand using existing method
        return self._build_single_viewer_profile(viewer_sessions)

    def _prepare_session_data(self):
        """Prepare session data with datetime parsing"""
        if self.df_sessions.empty:
            return
        
        # ✅ FIXED: Use shared module-level helper
        time_col = _find_column_helper(self.df_sessions.columns, [
            'Session Start Time', 'session_start_time', 'timestamp'
        ])
        
        if time_col:
            self.df_sessions[time_col] = pd.to_datetime(
                self.df_sessions[time_col], errors='coerce'
            )
            self.df_sessions = self.df_sessions.sort_values(time_col)

    def _find_column(self, columns: List[str], possible_names: List[str]) -> Optional[str]:
        """
        Wrapper method for backward compatibility
        Delegates to shared module-level helper
        """
        return _find_column_helper(columns, possible_names)

    # ============================================================================
    # VIEWER PROFILE BUILDING
    # ============================================================================

    
    def _build_single_viewer_profile(self, viewer_sessions: pd.DataFrame) -> Dict[str, Any]:
        """Build comprehensive single viewer profile"""
        if viewer_sessions.empty:
            return {
                'type': 'new_user',
                'session_count': 0,
                'success_rate': 0.0,
                'failure_rate': 0.0
            }
        
        total = len(viewer_sessions)
        
        # ✅ Use shared helper
        status_col = _find_column_helper(viewer_sessions.columns, [
            'Status', 'status', 'Ended Status'
        ])
        
        if status_col:
            successes = len(viewer_sessions[viewer_sessions[status_col] == 'Played'])
            failures = len(viewer_sessions[viewer_sessions[status_col].isin(['VSF-T', 'VSF-B', 'EBVS'])])
        else:
            successes = 0
            failures = 0
        
        # ✅ Use shared helper
        playing_col = _find_column_helper(viewer_sessions.columns, [
            'Playing Time', 'playing_time', 'Duration'
        ])
        
        avg_playing_time = 0
        if playing_col:
            playing_times = pd.to_numeric(viewer_sessions[playing_col], errors='coerce').dropna()
            avg_playing_time = playing_times.mean() if not playing_times.empty else 0
        
        return {
            'type': 'returning_user',
            'session_count': total,
            'success_rate': successes / total if total > 0 else 0,
            'failure_rate': failures / total if total > 0 else 0,
            'recent_sessions': viewer_sessions.tail(10),
            'is_heavy_user': total > 50,
            'avg_playing_time': avg_playing_time,
            'last_10_sessions': viewer_sessions.tail(10),
            'all_sessions': viewer_sessions
        }

    # ============================================================================
    # MULTI-LAYER DIAGNOSTIC ENGINE
    # ============================================================================

    def _apply_mvp_diagnosis_rules(self, session_row: pd.Series, viewer_id: str,
                               session_id: str, viewer_context: Dict[str, Any]) -> Diagnosis:
        """
        FIXED: Multi-layer diagnostic engine with viewer context as parameter

        Args:
            session_row: Current session data
            viewer_id: Viewer identifier (can be 'unknown')
            session_id: Session identifier
            viewer_context: Pre-fetched viewer context (from _get_viewer_context_for_session)
        """

        # Layer 1: Basic rules (session-only)
        basic_diagnosis = self._apply_basic_rules(session_row, session_id, viewer_id)

        # Layer 2: Temporal analysis (session + viewer history)
        temporal_analysis = self._temporal_correlation(
            session_row, viewer_id, viewer_context.get('recent_sessions', pd.DataFrame())
        )

        # Layer 3: Geographic correlation (session + advancetags)
        geographic_analysis = self._geographic_correlation(session_row)

        # Layer 4: User behavior analysis (viewer context)
        user_pattern = self._user_behavior_analysis(
            viewer_id, session_row, viewer_context
        )

        # Build evidence factors for confidence calculation
        evidence_factors = {
            'concurrent_failures': geographic_analysis.get('concurrent_count', 0),
            'user_pattern': user_pattern,
            'temporal_trend': temporal_analysis.get('trend', 'stable'),
            'widespread': geographic_analysis.get('is_widespread', False),
            'geographic_analysis': geographic_analysis,
            'temporal_analysis': temporal_analysis
        }

        # Calculate dynamic confidence with session context
        session_context = {
            'recent_deployment': False,  # You can implement _check_recent_deployment() later
            'is_peak_hours': self._is_peak_hours(
                self._get_field_value(session_row, ['Session Start Time', 'session_start_time'])
            ) if self._get_field_value(session_row, ['Session Start Time', 'session_start_time']) else False
        }

        confidence = self._calculate_dynamic_confidence(
            session_row, viewer_id, basic_diagnosis.root_cause, evidence_factors, session_context
        )

        # Calculate dynamic severity
        severity = self._calculate_dynamic_severity(session_row=session_row, viewer_id=viewer_id, root_cause=basic_diagnosis.root_cause, user_pattern=user_pattern, temporal_analysis=temporal_analysis)
        

        # Build enhanced evidence string
        enhanced_evidence = self._build_enhanced_evidence(
            basic_diagnosis.evidence,
            temporal_analysis,
            geographic_analysis,
            user_pattern
        )

        return Diagnosis(
            root_cause=basic_diagnosis.root_cause,
            confidence=confidence,
            evidence=enhanced_evidence,
            assign_team=basic_diagnosis.assign_team,
            severity_score=severity,
            user_behavior=user_pattern,
            temporal_analysis=temporal_analysis,
            geographic_analysis=geographic_analysis
        )

    def _apply_basic_rules(self, session_row: pd.Series, session_id: str,
                          viewer_id: str) -> Diagnosis:
        """Apply original 4-rule diagnosis system"""
        
        status = self._get_field_value(session_row, ['Status', 'status', 'Ended Status']) or ""
        starting_bitrate = self._get_field_value(session_row, ['Starting Bitrate', 'starting_bitrate']) or ""
        
        try:
            bitrate_numeric = float(str(starting_bitrate).replace(' bps', '').replace(',', '')) if starting_bitrate else 0
        except:
            bitrate_numeric = 0
        
        if status == 'VSF-B':
            return Diagnosis(
                root_cause="Potential Entitlement/Auth Issue",
                confidence=0.5,
                evidence="Session status VSF-B",
                assign_team="technical",
                severity_score=5
            )
        
        if status == 'EBVS':
            return Diagnosis(
                root_cause="Potential Entitlement/Auth Issue",
                confidence=0.5,
                evidence="User exited before video start",
                assign_team="technical",
                severity_score=5
            )
        
        if status == 'VSF-T' and bitrate_numeric == 0:
            return Diagnosis(
                root_cause="Potential CDN/Manifest Issue",
                confidence=0.5,
                evidence="Starting bitrate was 0 bps",
                assign_team="network",
                severity_score=5
            )
        
        if status == 'VSF-T':
            if self._check_recent_user_success(session_id, session_row, viewer_id):
                return Diagnosis(
                    root_cause="Potential Transient Network Issue",
                    confidence=0.5,
                    evidence="User had recent successful sessions",
                    assign_team="technical",
                    severity_score=5
                )
        
        return Diagnosis(
            root_cause="Technical Investigation Needed",
            confidence=0.5,
            evidence=f"Status: {status}",
            assign_team="technical",
            severity_score=5
        )

    def _temporal_correlation(self, session_row: pd.Series, viewer_id: str, 
                          viewer_sessions: pd.DataFrame) -> Dict[str, Any]:
        """
        FIXED: Analyze temporal patterns with viewer_sessions as parameter

        Args:
            session_row: Current session
            viewer_id: Viewer identifier
            viewer_sessions: Pre-fetched viewer session history (can be empty)
        """

        session_time = self._get_field_value(session_row, [
            'Session Start Time', 'session_start_time'
        ])

        if not session_time:
            return {'trend': 'unknown', 'pattern': 'insufficient_data'}

        # ✅ FIXED: Use passed viewer_sessions instead of looking up profiles
        if viewer_sessions.empty or len(viewer_sessions) < 3:
            return {'trend': 'insufficient_data', 'pattern': 'new_user'}

        # Find status column
        status_col = _find_column_helper(viewer_sessions.columns, ['Status', 'status'])
        if not status_col:
            return {'trend': 'unknown', 'pattern': 'no_status_data'}

        # Analyze last 10 sessions
        last_10 = viewer_sessions.tail(10)
        failures = last_10[last_10[status_col].isin(['VSF-T', 'VSF-B', 'EBVS'])]

        if len(failures) >= 5:
            # Count consecutive failures
            consecutive = self._count_consecutive_failures(viewer_id, viewer_sessions)
            if consecutive >= 3:
                return {
                    'trend': 'deteriorating',
                    'pattern': 'service_degradation',
                    'consecutive_failures': consecutive,
                    'confidence_boost': 0.15
                }
            else:
                return {
                    'trend': 'intermittent',
                    'pattern': 'intermittent_issues',
                    'confidence_boost': 0.05
                }

        # Check for stable/successful pattern
        successes = last_10[last_10[status_col] == 'Played']
        if len(successes) >= 8:
            return {
                'trend': 'stable',
                'pattern': 'isolated_incident',
                'confidence_boost': 0.10
            }

        return {
            'trend': 'normal',
            'pattern': 'mixed_performance',
            'confidence_boost': 0.0
        }

    def _geographic_correlation(self, session_row: pd.Series) -> Dict[str, Any]:
        """
        FIXED: Analyze geographic patterns for widespread issues
        Now uses flexible column lookup instead of hard-coded column names
        """

        if self.df_advancetags.empty:
            return {'is_widespread': False, 'concurrent_count': 0}

        # ✅ FIXED: Use flexible column lookup for session_id
        session_id = None
        for col_name in ['session_id', 'Session ID', 'Session Id', 'SessionID', 'sessionid']:
            if col_name in session_row.index and pd.notna(session_row[col_name]):
                session_id = session_row[col_name]
                break
            
        if not session_id:
            return {'is_widespread': False, 'concurrent_count': 0}

        # ✅ FIXED: Use flexible column lookup for session_start_time
        session_time = None
        for col_name in ['Session Start Time', 'session_start_time', 'timestamp', 'Session_Start_Time']:
            if col_name in session_row.index and pd.notna(session_row[col_name]):
                session_time = session_row[col_name]
                break
            
        if not session_time:
            return {'is_widespread': False, 'concurrent_count': 0}

        # ✅ FIXED: Find session_id column in advancetags DataFrame
        advancetags_session_col = None
        for col_name in ['session_id', 'Session ID', 'Session Id', 'SessionID', 'sessionid']:
            if col_name in self.df_advancetags.columns:
                advancetags_session_col = col_name
                break
            
        if not advancetags_session_col:
            logger.debug("No session_id column found in advancetags")
            return {'is_widespread': False, 'concurrent_count': 0}

        # Get metadata for current session
        session_meta = self.df_advancetags[
            self.df_advancetags[advancetags_session_col] == session_id
        ]

        if session_meta.empty:
            return {'is_widespread': False, 'concurrent_count': 0}

        # ✅ FIXED: Flexible column lookup for ISP and City
        isp = None
        for col_name in ['ispName', 'isp_name', 'ISP', 'isp']:
            if col_name in session_meta.columns:
                isp = session_meta.iloc[0].get(col_name, '')
                if pd.notna(isp):
                    break
                
        city = None
        for col_name in ['city', 'City', 'city_name']:
            if col_name in session_meta.columns:
                city = session_meta.iloc[0].get(col_name, '')
                if pd.notna(city):
                    break
                
        if not isp or not city:
            return {'is_widespread': False, 'concurrent_count': 0}

        # ✅ FIXED: Find session_start_time column in main sessions DataFrame
        sessions_time_col = None
        for col_name in ['Session Start Time', 'session_start_time', 'timestamp']:
            if col_name in self.df_sessions.columns:
                sessions_time_col = col_name
                break
            
        if not sessions_time_col:
            return {'is_widespread': False, 'concurrent_count': 0}

        # Find sessions in time window
        time_window = timedelta(minutes=5)
        nearby_sessions = self.df_sessions[
            (self.df_sessions[sessions_time_col] >= (session_time - time_window)) &
            (self.df_sessions[sessions_time_col] <= (session_time + time_window)) &
            (self.df_sessions[advancetags_session_col if advancetags_session_col in self.df_sessions.columns 
              else sessions_time_col] != session_id)
        ]

        # Count concurrent failures in same ISP/City
        concurrent_count = 0
        for _, other_session in nearby_sessions.iterrows():
            if self._is_video_start_failure(other_session):
                # Get other session's metadata
                other_session_id = None
                for col_name in ['session_id', 'Session ID', 'Session Id', 'SessionID']:
                    if col_name in other_session.index and pd.notna(other_session[col_name]):
                        other_session_id = other_session[col_name]
                        break
                    
                if not other_session_id:
                    continue
                
                other_meta = self.df_advancetags[
                    self.df_advancetags[advancetags_session_col] == other_session_id
                ]

                if not other_meta.empty:
                    other_isp = None
                    for col_name in ['ispName', 'isp_name', 'ISP', 'isp']:
                        if col_name in other_meta.columns:
                            other_isp = other_meta.iloc[0].get(col_name)
                            if pd.notna(other_isp):
                                break
                            
                    other_city = None
                    for col_name in ['city', 'City', 'city_name']:
                        if col_name in other_meta.columns:
                            other_city = other_meta.iloc[0].get(col_name)
                            if pd.notna(other_city):
                                break
                            
                    if other_isp == isp and other_city == city:
                        concurrent_count += 1

        return {
            'is_widespread': concurrent_count >= 3,
            'concurrent_count': concurrent_count,
            'affected_isp': isp,
            'affected_city': city,
            'confidence_boost': 0.20 if concurrent_count >= 5 else 0.10 if concurrent_count >= 3 else 0.0
        }

    def _user_behavior_analysis(self, viewer_id: str, current_session: pd.Series,
                            viewer_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Deep user behavior pattern analysis with viewer_context as parameter

        Args:
            viewer_id: Viewer identifier
            current_session: Current session data
            viewer_context: Pre-fetched viewer context (includes recent_sessions)
        """

        # ✅ FIXED: Use passed viewer_context instead of looking up profiles
        recent_sessions = viewer_context.get('recent_sessions', pd.DataFrame())

        if recent_sessions.empty or len(recent_sessions) < 3:
            return {
                'pattern': 'insufficient_data',
                'confidence_modifier': 0.0,
                'behavior_trend': 'unknown',
                'user_value_tier': 'unknown'
            }

        # Find status column
        status_col = _find_column_helper(recent_sessions.columns, ['Status', 'status'])
        if not status_col:
            return {
                'pattern': 'insufficient_data',
                'confidence_modifier': 0.0
            }

        # Analyze recent 10 sessions
        recent_10 = recent_sessions.tail(10)
        successes = len(recent_10[recent_10[status_col] == 'Played'])
        failures = len(recent_10[recent_10[status_col].isin(['VSF-T', 'VSF-B', 'EBVS'])])

        # Calculate user value tier
        user_value_tier = self._calculate_user_value_tier(viewer_context)

        # Pattern: Reliable user hit issue (mostly successful, rare failure)
        if successes >= 8 and failures <= 2:
            return {
                'pattern': 'reliable_user_hit_issue',
                'confidence_modifier': 0.15,
                'severity_boost': True,
                'behavior_trend': 'excellent',
                'user_value_tier': user_value_tier,
                'suggested_action': 'immediate_investigation'
            }

        # Pattern: Deteriorating experience (many recent failures)
        elif failures >= 5:
            consecutive = self._count_consecutive_failures(viewer_id, recent_sessions)
            if consecutive >= 3:
                return {
                    'pattern': 'deteriorating_experience',
                    'confidence_modifier': 0.10,
                    'severity_boost': True,
                    'behavior_trend': 'deteriorating',
                    'user_value_tier': user_value_tier,
                    'consecutive_failures': consecutive,
                    'suggested_action': 'proactive_support_outreach'
                }
            else:
                return {
                    'pattern': 'intermittent_issues',
                    'confidence_modifier': 0.0,
                    'behavior_trend': 'unstable',
                    'user_value_tier': user_value_tier,
                    'suggested_action': 'monitor_and_correlate'
                }

        # Default: Normal usage pattern
        return {
            'pattern': 'normal_usage',
            'confidence_modifier': 0.0,
            'behavior_trend': 'normal',
            'user_value_tier': user_value_tier,
            'suggested_action': 'standard_triage'
        }
    
    def _calculate_signal_strength(self, session_row: pd.Series, diagnosis_root_cause: str) -> Dict[str, Any]:
        """
        Calculate confidence boost from failure signal strength
        
        Strong signals (VSF-B, 0 bps) → high confidence boost
        Weak signals (transient) → low/negative boost
        """
        
        confidence_boost = 0.0
        evidence_pieces = 0
        
        # Get status and bitrate
        status = self._get_field_value(session_row, ['Status', 'status', 'Ended Status']) or ""
        starting_bitrate = self._get_field_value(session_row, ['Starting Bitrate', 'starting_bitrate']) or ""
        
        try:
            bitrate_numeric = float(str(starting_bitrate).replace(' bps', '').replace(',', '')) if starting_bitrate else 0
        except:
            bitrate_numeric = 0
        
        # LAYER 1A: VSF-B (Very strong signal)
        if status == 'VSF-B':
            confidence_boost += 0.30
            evidence_pieces += 3
            
        # LAYER 1B: EBVS (Strong signal, but user-initiated)
        elif status == 'EBVS':
            # Check exit time to determine if it was quick frustration exit
            exit_time = self._get_field_value(session_row, ['Playing Time', 'playing_time'])
            try:
                exit_seconds = float(exit_time) if exit_time else 0
            except:
                exit_seconds = 0
                
            if exit_seconds < 5:
                # Very quick exit = likely frustrated by failure
                confidence_boost += 0.20
                evidence_pieces += 2
            else:
                # Longer exit = less clear signal
                confidence_boost += 0.12
                evidence_pieces += 1
        
        # LAYER 1C: VSF-T with 0 bps (Very strong signal)
        elif status == 'VSF-T' and bitrate_numeric == 0:
            confidence_boost += 0.30
            evidence_pieces += 3
        
        # LAYER 1D: VSF-T with non-zero bitrate (Moderate signal)
        elif status == 'VSF-T':
            confidence_boost += 0.15
            evidence_pieces += 2
        
        # LAYER 1E: Unknown status (No signal)
        else:
            confidence_boost += 0.0
            evidence_pieces += 0
        
        return {
            'confidence_boost': confidence_boost,
            'evidence_pieces': evidence_pieces
        }


    def _calculate_technical_confidence(self, session_row: pd.Series, diagnosis_root_cause: str) -> Dict[str, Any]:
        """
        Calculate confidence from technical metrics alignment

        When technical metrics support the diagnosis → boost confidence
        When they contradict → reduce confidence
        """

        confidence_boost = 0.0
        evidence_pieces = 0

        # Get technical fields
        starting_bitrate = self._get_field_value(session_row, ['Starting Bitrate', 'starting_bitrate']) or ""
        buffer_count = self._get_field_value(session_row, ['Rebuffer Count', 'rebuffer_count'])
        playing_time = self._get_field_value(session_row, ['Playing Time', 'playing_time'])

        try:
            bitrate_numeric = float(str(starting_bitrate).replace(' bps', '').replace(',', '')) if starting_bitrate else 0
        except:
            bitrate_numeric = 0

        # CDN/Manifest Issue Verification
        if 'CDN' in diagnosis_root_cause or 'Manifest' in diagnosis_root_cause:
            if bitrate_numeric == 0:
                # Strong technical evidence: 0 bps confirms CDN/manifest issue
                confidence_boost += 0.15
                evidence_pieces += 2
            elif bitrate_numeric > 0:
                # Weak evidence: non-zero bitrate suggests CDN worked partially
                confidence_boost += 0.05
                evidence_pieces += 1

        # Entitlement/Auth Issue Verification
        elif 'Entitlement' in diagnosis_root_cause or 'Auth' in diagnosis_root_cause:
            # Auth issues typically fail before bitrate negotiation
            if bitrate_numeric == 0:
                confidence_boost += 0.10
                evidence_pieces += 1
            else:
                # Non-zero bitrate weakly contradicts auth issue
                confidence_boost -= 0.05

        # Transient Network Issue Verification
        elif 'Transient' in diagnosis_root_cause or 'Network' in diagnosis_root_cause:
            # Transient issues may show partial success signals
            try:
                buffer_int = int(buffer_count) if buffer_count else 0
                playing_float = float(playing_time) if playing_time else 0
            except:
                buffer_int = 0
                playing_float = 0

            if buffer_int > 0 or playing_float > 0:
                # Evidence of partial playback supports transient theory
                confidence_boost += 0.08
                evidence_pieces += 1

        return {
            'confidence_boost': confidence_boost,
            'evidence_pieces': evidence_pieces
        }


    def _calculate_contradiction_penalty(self, session_row: pd.Series, diagnosis_root_cause: str, evidence_factors: Dict[str, Any]) -> float:
        """
        Calculate penalty for contradictory evidence
        
        Reduces confidence when evidence conflicts with diagnosis
        """
        
        penalty = 0.0
        
        # USER PATTERN CONTRADICTIONS
        user_pattern = evidence_factors.get('user_pattern', {})
        pattern_type = user_pattern.get('pattern', 'unknown')
        
        # If diagnosis is "transient" but user has deteriorating pattern
        if 'Transient' in diagnosis_root_cause:
            if pattern_type == 'deteriorating_experience':
                # NOT transient if user consistently failing
                penalty += 0.15
            elif pattern_type == 'reliable_user_hit_issue':
                # NOT transient if reliable user suddenly fails
                penalty += 0.10
        
        # If diagnosis is "CDN issue" but it's isolated (no concurrent failures)
        if 'CDN' in diagnosis_root_cause or 'Manifest' in diagnosis_root_cause:
            concurrent = evidence_factors.get('concurrent_failures', 0)
            if concurrent == 0:
                # Isolated CDN failure is suspicious
                penalty += 0.12
            elif concurrent == 1:
                penalty += 0.08
        
        # GEOGRAPHIC CONTRADICTIONS
        geographic_analysis = evidence_factors.get('geographic_analysis', {})
        if geographic_analysis.get('is_widespread', False):
            # If widespread issue but diagnosis suggests user-specific problem
            if 'Entitlement' in diagnosis_root_cause or 'Auth' in diagnosis_root_cause:
                # Auth issues are rarely widespread
                penalty += 0.10
        
        # TECHNICAL INDICATOR CONTRADICTIONS
        starting_bitrate = self._get_field_value(session_row, ['Starting Bitrate', 'starting_bitrate']) or ""
        try:
            bitrate_numeric = float(str(starting_bitrate).replace(' bps', '').replace(',', '')) if starting_bitrate else 0
        except:
            bitrate_numeric = 0
        
        # If we diagnose CDN issue but bitrate was negotiated successfully
        if 'CDN' in diagnosis_root_cause or 'Manifest' in diagnosis_root_cause:
            if bitrate_numeric > 1000000:  # > 1 Mbps
                # Strong bitrate contradicts CDN failure
                penalty += 0.15
        
        return penalty

    def _calculate_user_value_tier(self, profile: Dict[str, Any]) -> str:
        """Calculate user value tier for prioritization"""
        session_count = profile.get('session_count', 0)
        success_rate = profile.get('success_rate', 0)
        avg_playing_time = profile.get('avg_playing_time', 0)
        
        if session_count > 100 and success_rate > 0.9 and avg_playing_time > 3600:
            return 'premium'
        elif session_count > 50 and success_rate > 0.8:
            return 'high_value'
        elif session_count > 20:
            return 'regular'
        else:
            return 'casual'


    def _calculate_dynamic_confidence(self, session_row: pd.Series, viewer_id: str, diagnosis_root_cause: str, evidence_factors: Dict[str, Any], session_context: Dict[str, Any] = None):
        """
        Fully dynamic confidence calculation based on available data evidence.

        Starting Point: 0.5 (neutral - "we don't know yet")

        Philosophy: 
        - Start neutral (0.5 = uninformed prior)
        - Evidence pushes confidence UP (towards 1.0) or DOWN (towards 0.0)
        - Strong evidence → high confidence (0.8+)
        - Weak evidence → slight confidence (0.5-0.6)
        - Contradictory evidence → low confidence (0.2-0.4)

        Args:
            session_row: Current session data
            viewer_id: Viewer identifier
            diagnosis_root_cause: Diagnosed root cause
            evidence_factors: Context from analysis layers
            session_context: Additional session metadata (optional)

        Returns:
            Confidence score (0.0 - 1.0) based purely on data evidence
        """

        # Initialize session_context if not provided
        if session_context is None:
            session_context = {}

        # ================================================================
        # START AT 0.5: NEUTRAL BASELINE
        # ================================================================

        confidence_score = 0.5  # ✅ Uninformed prior
        evidence_count = 0

        # ================================================================
        # LAYER 1: FAILURE SIGNAL STRENGTH (Core Diagnostic Evidence)
        # ================================================================

        signal_strength = self._calculate_signal_strength(
            session_row, diagnosis_root_cause
        )
        confidence_score += signal_strength['confidence_boost']
        evidence_count += signal_strength['evidence_pieces']

        # ================================================================
        # LAYER 2: GEOGRAPHIC CORRELATION (Widespread Pattern Evidence)
        # ================================================================

        concurrent = evidence_factors.get('concurrent_failures', 0)

        if concurrent >= 10:
            # Very strong evidence: widespread outage
            confidence_score += 0.25
            evidence_count += 3
        elif concurrent >= 5:
            # Strong evidence: significant pattern
            confidence_score += 0.18
            evidence_count += 2
        elif concurrent >= 3:
            # Moderate evidence: emerging pattern
            confidence_score += 0.10
            evidence_count += 1
        elif concurrent >= 2:
            # Weak evidence: possible pattern
            confidence_score += 0.05
            evidence_count += 1

        # Geographic specificity adds confidence
        geo_analysis = evidence_factors.get('geographic_analysis', {})
        if geo_analysis.get('is_widespread', False):
            confidence_score += 0.07
            evidence_count += 1

        # ================================================================
        # LAYER 3: USER BEHAVIOR PATTERN (Historical Context Evidence)
        # ================================================================

        user_pattern = evidence_factors.get('user_pattern', {})
        pattern_type = user_pattern.get('pattern', 'unknown')

        if pattern_type == 'reliable_user_hit_issue':
            # Strong evidence: user never fails, suddenly failed
            confidence_score += 0.15
            evidence_count += 2
        elif pattern_type == 'deteriorating_experience':
            # Strong evidence: clear degradation trend
            confidence_score += 0.12
            evidence_count += 2
        elif pattern_type == 'intermittent_issues':
            # Moderate evidence: recurring problem
            confidence_score += 0.07
            evidence_count += 1
        elif pattern_type == 'user_troubleshooting':
            # Weak evidence: user trying different approaches
            confidence_score += 0.04
            evidence_count += 1
        elif pattern_type == 'new_user':
            # Neutral: no historical context (no change from 0.5)
            confidence_score += 0.0
            evidence_count += 0
        elif pattern_type == 'insufficient_data':
            # Negative weak evidence: can't verify with history
            confidence_score -= 0.03
            evidence_count += 0

        # User value tier consideration
        user_value = user_pattern.get('user_value_tier', 'unknown')
        if user_value in ['premium', 'high_value']:
            # High-value users typically have better setup
            # If they fail, more likely our issue
            confidence_score += 0.05
            evidence_count += 1
        elif user_value == 'casual':
            # Casual users might have setup issues
            # Slight negative adjustment
            confidence_score -= 0.02

        # ================================================================
        # LAYER 4: TEMPORAL CORRELATION (Time-Based Pattern Evidence)
        # ================================================================

        temporal_analysis = evidence_factors.get('temporal_analysis', {})
        temporal_trend = temporal_analysis.get('trend', 'unknown')
        temporal_pattern = temporal_analysis.get('pattern', 'unknown')

        if temporal_pattern == 'service_degradation':
            # Very strong evidence: clear degradation over time
            confidence_score += 0.15
            evidence_count += 2
        elif temporal_trend == 'deteriorating':
            # Strong evidence: getting worse
            confidence_score += 0.10
            evidence_count += 1
        elif temporal_trend == 'stable':
            # Moderate evidence: isolated incident in stable stream
            confidence_score += 0.07
            evidence_count += 1
        elif temporal_trend == 'intermittent':
            # Weak evidence: on-and-off pattern
            confidence_score += 0.03
            evidence_count += 1

        # Consecutive failures add evidence
        consecutive = temporal_analysis.get('consecutive_failures', 0)
        if consecutive >= 5:
            confidence_score += 0.12
            evidence_count += 2
        elif consecutive >= 3:
            confidence_score += 0.08
            evidence_count += 1

        # ================================================================
        # LAYER 5: TECHNICAL INDICATORS (Measurable Failure Evidence)
        # ================================================================

        tech_confidence = self._calculate_technical_confidence(
            session_row, diagnosis_root_cause
        )
        confidence_score += tech_confidence['confidence_boost']
        evidence_count += tech_confidence['evidence_pieces']

        # ================================================================
        # LAYER 6: CONTEXTUAL EVIDENCE (Environmental Factors)
        # ================================================================

        # Peak hours + CDN issue = more likely systemic
        time_col = _find_column_helper([c for c in session_row.index], [
            'Session Start Time', 'session_start_time'
        ])

        if time_col and self._is_peak_hours(session_row.get(time_col)):
            if 'CDN' in diagnosis_root_cause or 'Manifest' in diagnosis_root_cause:
                # Peak hours + CDN issue = higher confidence
                confidence_score += 0.08
                evidence_count += 1
            elif 'Network' in diagnosis_root_cause:
                # Peak hours + network issue = moderate confidence
                confidence_score += 0.04
                evidence_count += 1

        # Recent deployment or system changes
        if session_context.get('recent_deployment', False):
            confidence_score += 0.07
            evidence_count += 1

        # ================================================================
        # LAYER 7: EVIDENCE DIVERSITY BONUS
        # ================================================================

        # Multiple independent evidence sources increase confidence
        if evidence_count >= 10:
            confidence_score += 0.08
        elif evidence_count >= 7:
            confidence_score += 0.06
        elif evidence_count >= 5:
            confidence_score += 0.04
        elif evidence_count >= 3:
            confidence_score += 0.02
        # Less than 3 pieces of evidence → slight penalty
        elif evidence_count <= 1:
            confidence_score -= 0.05

        # ================================================================
        # LAYER 8: CONTRADICTORY EVIDENCE PENALTY
        # ================================================================

        contradiction_penalty = self._calculate_contradiction_penalty(
            session_row, diagnosis_root_cause, evidence_factors
        )
        confidence_score -= contradiction_penalty

        # ================================================================
        # FINAL ADJUSTMENT: ENSURE VALID RANGE
        # ================================================================

        # Cap between 0.1 and 0.95 (avoid absolute certainty)
        confidence_score = max(0.1, min(0.95, confidence_score))

        # Log for debugging
        logger.debug(
            f"Confidence: {confidence_score:.2f} | "
            f"Evidence: {evidence_count} pieces | "
            f"Root: {diagnosis_root_cause}"
        )

        return confidence_score

    # ============================================================================
    # DYNAMIC SEVERITY CALCULATION ENGINE
    # ============================================================================

    def _calculate_dynamic_severity(self, session_row: pd.Series, viewer_id: str, 
                               root_cause: str, user_pattern: Dict[str, Any] = None,
                               temporal_analysis: Dict[str, Any] = None) -> int:
        """
        Calculate severity dynamically based on multiple factors.
        Severity represents impact on user experience and system health.

        Scale: 0-10 where:
        - 0-2: Minimal impact (isolated, one-time issue)
        - 3-4: Low impact (occasional issues for user)
        - 5-6: Medium impact (recurring issues or single high-value user affected)
        - 7-8: High impact (many users affected or critical user blocked)
        - 9-10: Critical (widespread outage, premium users affected)
        """
        # Initialize defaults
        if user_pattern is None:
            user_pattern = {}
        if temporal_analysis is None:
            temporal_analysis = {}

        severity_score = 5  # Neutral baseline

        # ================================================================
        # LAYER 1: FAILURE TYPE SEVERITY
        # ================================================================

        status = self._get_field_value(session_row, ['Status', 'status', 'Ended Status']) or ""

        if status == 'VSF-B':
            # Auth failures = immediate blocker
            severity_score += 2
        elif status == 'EBVS':
            # User exit before start = moderate blocker
            severity_score += 1.5
        elif status == 'VSF-T':
            # Technical failure = medium issue
            starting_bitrate = self._get_field_value(session_row, 
                ['Starting Bitrate', 'starting_bitrate']) or ""
            try:
                bitrate_numeric = float(str(starting_bitrate).replace(' bps', '').replace(',', '')) \
                    if starting_bitrate else 0
            except:
                bitrate_numeric = 0

            if bitrate_numeric == 0:
                # No bitrate negotiation = critical technical issue
                severity_score += 2.5
            else:
                # Partial bitrate = moderate technical issue
                severity_score += 1

        # ================================================================
        # LAYER 2: USER IMPACT SEVERITY
        # ================================================================

        # Get viewer context to assess user importance
        if viewer_id and viewer_id != 'unknown' and viewer_id in self.viewer_index:
            session_indices = self.viewer_index[viewer_id]
            viewer_sessions = self.df_sessions.loc[session_indices]
            viewer_profile = self._build_single_viewer_profile(viewer_sessions)

            user_value_tier = self._calculate_user_value_tier(viewer_profile)

            # Premium/high-value users = higher severity
            if user_value_tier == 'premium':
                severity_score += 2.5
            elif user_value_tier == 'high_value':
                severity_score += 1.5
            elif user_value_tier == 'regular':
                severity_score += 0.5

            # User success rate history
            success_rate = viewer_profile.get('success_rate', 0.5)

            # Reliable users suddenly failing = higher severity
            if success_rate >= 0.9:
                severity_score += 1.5
            # Users with poor history = lower concern
            elif success_rate <= 0.5:
                severity_score -= 1.0

        # ================================================================
        # LAYER 3: TEMPORAL/PATTERN SEVERITY
        # ================================================================

        if viewer_id and viewer_id != 'unknown' and viewer_id in self.viewer_index:
            session_indices = self.viewer_index[viewer_id]
            viewer_sessions = self.df_sessions.loc[session_indices]

            if not viewer_sessions.empty:
                status_col = _find_column_helper(viewer_sessions.columns, 
                    ['Status', 'status'])

                if status_col:
                    # Check for consecutive failures
                    consecutive = self._count_consecutive_failures(viewer_id, 
                        viewer_sessions)

                    if consecutive >= 5:
                        # Multiple consecutive failures = high severity
                        severity_score += 2.5
                    elif consecutive >= 3:
                        # Some consecutive failures = moderate boost
                        severity_score += 1.5

                    # Check for deteriorating trend
                    last_10 = viewer_sessions.tail(10)
                    failures = last_10[last_10[status_col].isin(['VSF-T', 'VSF-B', 'EBVS'])]

                    if len(failures) >= 7:
                        # Most recent sessions failing = high severity
                        severity_score += 2.0
                    elif len(failures) >= 5:
                        # Many recent failures = moderate boost
                        severity_score += 1.0

        # ================================================================
        # LAYER 4: GEOGRAPHIC/SYSTEM-WIDE SEVERITY
        # ================================================================

        if not self.df_advancetags.empty:
            session_id = self._get_field_value(session_row, 
                ['session_id', 'Session ID', 'Session Id', 'SessionID'])
            session_time = self._get_field_value(session_row, 
                ['Session Start Time', 'session_start_time', 'timestamp'])

            if session_id and session_time:
                geographic = self._geographic_correlation(session_row)
                concurrent_count = geographic.get('concurrent_count', 0)

                # Widespread outage detection
                if concurrent_count >= 20:
                    # Major outage = critical
                    severity_score += 3.0
                elif concurrent_count >= 10:
                    # Significant outage = high
                    severity_score += 2.5
                elif concurrent_count >= 5:
                    # Emerging outage = moderate
                    severity_score += 1.5
                elif concurrent_count >= 2:
                    # Regional pattern = slight boost
                    severity_score += 0.5

        # ================================================================
        # LAYER 5: TIME OF DAY IMPACT
        # ================================================================

        session_time = self._get_field_value(session_row, 
            ['Session Start Time', 'session_start_time'])

        if session_time and self._is_peak_hours(session_time):
            # Issues during peak hours affect more users
            severity_score += 1.5

        # ================================================================
        # LAYER 6: TECHNICAL INDICATOR SEVERITY
        # ================================================================

        # Rebuffer count
        rebuffer_count = self._get_field_value(session_row, 
            ['Rebuffer Count', 'rebuffer_count'])
        try:
            rebuf_int = int(rebuffer_count) if rebuffer_count else 0
        except:
            rebuf_int = 0

        if rebuf_int > 10:
            # Many rebuffers = poor QoE
            severity_score += 1.0
        elif rebuf_int > 5:
            severity_score += 0.5

        # Playing time before failure
        playing_time = self._get_field_value(session_row, 
            ['Playing Time', 'playing_time'])
        try:
            play_float = float(playing_time) if playing_time else 0
        except:
            play_float = 0

        # Failed immediately = worse UX than failed after watching
        if status == 'VSF-T' and play_float < 5:
            severity_score += 1.0
        elif status == 'VSF-T' and play_float < 30:
            severity_score += 0.5

        # ================================================================
        # LAYER 7: RECOVERY POTENTIAL
        # ================================================================

        # If user had recent success, this looks like transient issue
        if viewer_id and viewer_id != 'unknown':
            if self._check_recent_user_success(session_id or "", session_row, viewer_id):
                # Transient issue = lower severity
                severity_score -= 1.5

        # ================================================================
        # ✅ NEW: LAYER 8 - USER PATTERN ADJUSTMENTS (from old wrapper)
        # ================================================================

        if user_pattern.get('severity_boost'):
            if user_pattern.get('pattern') == 'deteriorating_experience':
                severity_score += 2
            elif user_pattern.get('pattern') == 'reliable_user_hit_issue':
                severity_score += 1

        # ================================================================
        # ✅ NEW: LAYER 9 - TEMPORAL PATTERN ADJUSTMENTS (from old wrapper)
        # ================================================================

        if temporal_analysis.get('pattern') == 'service_degradation':
            severity_score += 1


        # ================================================================
        # FINAL NORMALIZATION
        # ================================================================

        # Ensure integer in valid range
        final_severity = max(0, min(10, int(round(severity_score))))

        logger.debug(f"Severity calculation: base=5 → adjustments={severity_score-5:.1f} → "
                    f"final={final_severity} | status={status} | concurrent={concurrent_count if 'concurrent_count' in locals() else 'N/A'}")

        return final_severity


    # ============================================================================
    # UPDATED: _calculate_comprehensive_severity (now calls dynamic calculation)
    # ============================================================================

    def _build_enhanced_evidence(self, base_evidence: str,
                                 temporal: Dict[str, Any],
                                 geographic: Dict[str, Any],
                                 user_pattern: Dict[str, Any]) -> str:
        """Build comprehensive evidence string"""
        
        evidence_parts = [base_evidence]
        
        if geographic.get('is_widespread'):
            evidence_parts.append(
                f"{geographic['concurrent_count']} concurrent failures in "
                f"{geographic['affected_city']} ({geographic['affected_isp']})"
            )
        
        if temporal.get('pattern') != 'insufficient_data':
            evidence_parts.append(f"User pattern: {temporal['pattern']}")
        
        if user_pattern.get('pattern') != 'insufficient_data':
            evidence_parts.append(f"Behavior: {user_pattern['pattern']}")
        
        return ". ".join(evidence_parts)

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def _count_consecutive_failures(self, viewer_id: str, 
                                viewer_sessions: pd.DataFrame) -> int:
        """
        FIXED: Count consecutive failures with viewer_sessions as parameter
        
        Args:
            viewer_id: Viewer identifier
            viewer_sessions: Pre-fetched viewer session history
        """
        
        # ✅ FIXED: Use passed viewer_sessions instead of looking up profiles
        if viewer_sessions.empty:
            return 0
        
        # Find status column
        status_col = _find_column_helper(viewer_sessions.columns, ['Status', 'status'])
        if not status_col:
            return 0
        
        # Count consecutive failures from most recent sessions
        count = 0
        for _, session in viewer_sessions.iloc[::-1].iterrows():  # Reverse order (newest first)
            if session[status_col] in ['VSF-T', 'VSF-B', 'EBVS']:
                count += 1
            else:
                break  # Stop at first success
        
        return count
    
    def _is_peak_hours(self, timestamp) -> bool:
        """Check peak hours (6 PM - 11 PM)"""
        if not isinstance(timestamp, datetime):
            try:
                timestamp = pd.to_datetime(timestamp)
            except:
                return False
        return 18 <= timestamp.hour <= 23

    def _check_recent_user_success(self, session_id: str, current_session: pd.Series,
                              viewer_id: str) -> bool:
        """
        FIXED: Check if viewer has recent successful sessions
        Now fetches viewer sessions on-demand instead of using profiles

        Args:
            session_id: Current session ID
            current_session: Current session data
            viewer_id: Viewer identifier
        """

        # ✅ FIXED: Get viewer sessions on-demand
        if not viewer_id or viewer_id == 'unknown' or viewer_id not in self.viewer_index:
            return False

        # Get viewer's sessions using index
        session_indices = self.viewer_index[viewer_id]
        viewer_sessions = self.df_sessions.loc[session_indices]

        if viewer_sessions.empty:
            return False

        # Get current session time
        time_col = _find_column_helper([c for c in current_session.index], [
            'Session Start Time', 'session_start_time'
        ])

        if not time_col:
            return False

        current_time = current_session.get(time_col)

        # Find required columns
        status_col = _find_column_helper(viewer_sessions.columns, ['Status', 'status'])
        session_id_col = _find_column_helper(viewer_sessions.columns, ['Session ID', 'session_id'])

        if not all([status_col, session_id_col]):
            return False

        # Get earlier sessions (before current session)
        earlier = viewer_sessions[
            (viewer_sessions[time_col] < current_time) &
            (viewer_sessions[session_id_col] != session_id)
        ].tail(100)

        # Check for successful sessions
        successful = earlier[earlier[status_col] == 'Played']
        return not successful.empty

    def _is_video_start_failure(self, session_row: pd.Series) -> bool:
        """Check if session is a video start failure"""
        status = self._get_field_value(session_row, ['Status', 'status', 'Ended Status'])
        return status in ['VSF-T', 'VSF-B', 'EBVS']
    
    def _get_field_value(self, row: pd.Series, field_names: List[str]) -> Any:
        """
        FIXED: Get field value from possible column names with BETTER fallback
        """
        if row is None or row.empty:
            return None
        available_columns = list(row.index)
        # Step 1: Try exact matches first (case-sensitive)
        for field_name in field_names:
            if field_name in available_columns:
                value = row[field_name]
                if pd.notna(value):
                    return value
        # Step 2: Try case-insensitive matches
        for field_name in field_names:
            field_lower = str(field_name).lower().strip()
            for col in available_columns:
                col_lower = str(col).lower().strip()
                if col_lower == field_lower:
                    value = row[col]
                    if pd.notna(value):
                        logger.debug(f"✅ Found '{field_name}' as '{col}'")
                        return value
        # Step 3: Try partial matches (contains) - FIXED LOGIC
        for field_name in field_names:
            field_lower = str(field_name).lower().strip()
            # Extract key terms (e.g., "session_id" → ["session", "id"])
            terms = field_lower.replace('_', ' ').replace('-', ' ').split()
            for col in available_columns:
                col_lower = str(col).lower().strip()
                # Check if ALL terms present in column name
                if all(term in col_lower for term in terms):
                    value = row[col]
                    if pd.notna(value):
                        logger.debug(f"✅ Found '{field_name}' via partial match: '{col}'")
                        return value
        # Step 4: Return None if nothing found (NO KeyError)
        logger.debug(f"⚠️ Field not found. Looking for: {field_names}, Available: {available_columns}")
        return None
    
    def debug_dataframe_structure(self):
        """Debug method to inspect DataFrame structure"""
        logger.info("=== DATAFRAME DEBUG INFO ===")
        logger.info(f"Sessions DataFrame shape: {self.df_sessions.shape}")
        logger.info(f"Sessions columns: {list(self.df_sessions.columns)}")

        if not self.df_sessions.empty:
            logger.info("Sample session data (first row):")
            first_row = self.df_sessions.iloc[0]
            for col in first_row.index:
                logger.info(f"  {col}: {first_row[col]}")

        logger.info("=== END DEBUG INFO ===")

    # ============================================================================
    # MAIN PROCESSING FLOW
    # ============================================================================

    def process(self) -> List[Dict[str, Any]]:
        """
        FIXED: Session-first ticket processing with VERIFIED column mapping
        """
        try:
            logger.info("=" * 70)
            logger.info("🎫 TICKET ENGINE: Starting Session-First Processing")
            logger.info("=" * 70)

            if self.df_sessions.empty:
                logger.warning("❌ No session data available")
                return []

            # ✅ NEW: VERIFY AND LOG COLUMN NAMES
            logger.info(f"📊 Processing {len(self.df_sessions)} sessions")
            logger.info(f"📋 Column names: {list(self.df_sessions.columns)[:15]}")

            # ✅ NEW: Check for required columns
            has_session_id = any(col for col in self.df_sessions.columns 
                                if 'session' in str(col).lower() and 'id' in str(col).lower())

            if not has_session_id:
                logger.error("❌ CRITICAL: No session ID column found!")
                logger.error(f"Available columns: {list(self.df_sessions.columns)}")
                return []

            tickets = []
            processed_count = 0
            failed_count = 0
            skipped_no_session_id = 0

            # ✅ Iterate sessions directly
            for idx, session_row in self.df_sessions.iterrows():
                try:
                    # ✅ FIXED: More flexible session_id extraction
                    session_id = None

                    # Try standard column names
                    for col_name in ['session_id', 'Session ID', 'Session Id', 'SessionID', 'sessionid']:
                        if col_name in session_row.index and pd.notna(session_row[col_name]):
                            session_id = str(session_row[col_name]).strip()
                            break
                        
                    # Fallback: Search for any column with 'session' and 'id'
                    if not session_id:
                        for col in session_row.index:
                            col_lower = str(col).lower()
                            if 'session' in col_lower and 'id' in col_lower:
                                if pd.notna(session_row[col]):
                                    session_id = str(session_row[col]).strip()
                                    logger.debug(f"Found session ID in column: '{col}'")
                                    break

                    if not session_id or session_id.lower() in ['nan', 'none', '']:
                        skipped_no_session_id += 1
                        if skipped_no_session_id <= 5:  # Log first 5 only
                            logger.debug(f"Skipping row {idx}: No valid session_id")
                        continue

                    # Extract channel name
                    channel_name = None
                    for col_name in ['asset_name', 'Asset Name', 'channel', 'Channel']:
                        if col_name in session_row.index and pd.notna(session_row[col_name]):
                            channel_name = str(session_row[col_name])
                            break
                        
                    if not channel_name:
                        channel_name = "Unknown"

                    # Filter by target channels if specified
                    if self.target_channels and channel_name not in self.target_channels:
                        continue
                    
                    # Check if this is a video start failure
                    if not self._is_video_start_failure(session_row):
                        continue
                    
                    failed_count += 1

                    # Extract viewer_id (OPTIONAL)
                    viewer_id = None
                    for col_name in ['viewer_id', 'Viewer ID', 'ViewerID', 'user_id']:
                        if col_name in session_row.index and pd.notna(session_row[col_name]):
                            viewer_id = str(session_row[col_name]).strip()
                            break
                        
                    if not viewer_id or viewer_id.lower() in ['nan', 'none', '']:
                        viewer_id = 'unknown'

                    # Get viewer context on-demand
                    viewer_context = self._get_viewer_context_for_session(session_row, viewer_id)

                    # Apply enhanced diagnosis
                    diagnosis = self._apply_mvp_diagnosis_rules(
                        session_row, viewer_id, session_id, viewer_context
                    )

                    # Generate alerts
                    alerts = []

                    # Build ticket
                    ticket = self._build_mvp_ticket(
                        session_id, viewer_id, channel_name, diagnosis, session_row, alerts
                    )
                    tickets.append(ticket)
                    processed_count += 1

                    if processed_count % 100 == 0:
                        logger.info(f"   ✅ Processed {processed_count} failures...")

                except Exception as e:
                    logger.error(f"   ❌ Error processing session {idx}: {e}", exc_info=True)
                    continue
                
            # Summary
            logger.info("=" * 70)
            logger.info("📊 TICKET GENERATION SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Total sessions processed: {len(self.df_sessions)}")
            logger.info(f"Skipped (no session_id): {skipped_no_session_id}")
            logger.info(f"Video start failures found: {failed_count}")
            logger.info(f"Tickets generated: {len(tickets)}")
            if failed_count > 0:
                logger.info(f"Success rate: {(processed_count/failed_count*100):.1f}%")
            logger.info("=" * 70)

            return tickets

        except Exception as e:
            logger.error(f"❌ Ticket processing failed: {e}", exc_info=True)
            return []
    
    def _build_mvp_ticket(self, session_id: str, viewer_id: str, channel: str,
                          diagnosis: Diagnosis, session_row: pd.Series,
                          alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build MVP ticket with all diagnostic layers"""

        failure_time = self._get_field_value(session_row, [
            'Session Start Time', 'session_start_time', 'timestamp'
        ])
        
        if failure_time and pd.notna(failure_time):
            try:
                if isinstance(failure_time, str):
                    failure_time = pd.to_datetime(failure_time)
                ts_str = failure_time.strftime('%b %d %Y, %H:%M:%S')
            except:
                ts_str = str(failure_time)
        else:
            ts_str = "Unknown"
        
        ticket_id = f"TKT_{viewer_id}_{session_id[-8:]}"
        confidence_pct = int(diagnosis.confidence * 100)
        severity_label = ("CRITICAL" if diagnosis.severity_score >= 8 else 
                         "HIGH" if diagnosis.severity_score >= 6 else 
                         "MEDIUM")
        
        alert_prefix = ""
        if alerts:
            critical_alerts = [a for a in alerts if a.get('severity') == 'CRITICAL']
            if critical_alerts:
                alert_prefix = "🚨 OUTAGE DETECTED - "
        
        title = f"{alert_prefix}[VSF] [{severity_label}] [{diagnosis.root_cause}] for Viewer {viewer_id} on {channel}"
        
        description_parts = [
            "=== ENHANCED FAILURE TICKET ===",
            "",
            f"TITLE: {title}",
            "",
            "BODY:",
            f"- Viewer ID: {viewer_id}",
            f"- Impacted Channel: {channel}",
            f"- Time of Failure: {ts_str}",
            "",
            "DIAGNOSIS:",
            f"- Root Cause: {diagnosis.root_cause}",
            f"- Confidence: {confidence_pct}% ({diagnosis.confidence:.2f})",
            f"- Severity Score: {diagnosis.severity_score}/10 ({severity_label})",
            f"- Evidence: {diagnosis.evidence}",
            "",
            "USER BEHAVIOR ANALYSIS:",
            f"- Pattern: {diagnosis.user_behavior.get('pattern', 'unknown')}",
            f"- Behavior Trend: {diagnosis.user_behavior.get('behavior_trend', 'unknown')}",
            f"- User Value Tier: {diagnosis.user_behavior.get('user_value_tier', 'unknown')}",
            f"- Suggested Action: {diagnosis.user_behavior.get('suggested_action', 'standard_triage')}"
        ]
        
        if diagnosis.temporal_analysis.get('pattern') != 'insufficient_data':
            description_parts.extend([
                "",
                "TEMPORAL ANALYSIS:",
                f"- Trend: {diagnosis.temporal_analysis.get('trend', 'unknown')}",
                f"- Pattern: {diagnosis.temporal_analysis.get('pattern', 'unknown')}"
            ])
            if 'consecutive_failures' in diagnosis.temporal_analysis:
                description_parts.append(
                    f"- Consecutive Failures: {diagnosis.temporal_analysis['consecutive_failures']}"
                )
        
        if diagnosis.geographic_analysis.get('is_widespread'):
            description_parts.extend([
                "",
                "GEOGRAPHIC ANALYSIS:",
                f"- Widespread Issue: YES",
                f"- Concurrent Failures: {diagnosis.geographic_analysis['concurrent_count']}",
                f"- Affected ISP: {diagnosis.geographic_analysis.get('affected_isp', 'Unknown')}",
                f"- Affected City: {diagnosis.geographic_analysis.get('affected_city', 'Unknown')}"
            ])
        
        if alerts:
            description_parts.extend(["", "REAL-TIME ALERTS:"])
            for alert in alerts:
                description_parts.append(f"- [{alert['type']}] {alert.get('severity', 'INFO')}")
        
        description_parts.extend([
            "",
            f"- Deep Link: https://example.com/session/{session_id}",
            f"- Assign to: {diagnosis.assign_team}"
        ])
        
        description = "\n".join(description_parts)
        
        return {
            'ticket_id': ticket_id,
            'viewer_id': viewer_id,
            'session_id': session_id,
            'title': title,
            'priority': 'critical' if diagnosis.severity_score >= 8 else 
                       'high' if diagnosis.severity_score >= 6 else 'medium',
            'status': 'new',
            'assign_team': diagnosis.assign_team,
            'issue_type': 'video_start_failure',
            'description': description,
            'failure_details': {
                'root_cause': diagnosis.root_cause,
                'confidence': diagnosis.confidence,
                'confidence_percentage': confidence_pct,
                'evidence': diagnosis.evidence,
                'failure_type': self._get_field_value(session_row, ['Status', 'status']) or 'VSF-T',
                'severity_score': diagnosis.severity_score,
                'severity_label': severity_label,
                'user_behavior': diagnosis.user_behavior,
                'temporal_analysis': diagnosis.temporal_analysis,
                'geographic_analysis': diagnosis.geographic_analysis
            },
            'context_data': {
                'asset_name': channel,
                'viewer_id': viewer_id,
                'failure_time': str(failure_time) if pd.notna(failure_time) else None,
                'deep_link': f"https://example.com/session/{session_id}"
            },
            'confidence_score': diagnosis.confidence,
            'severity_score': diagnosis.severity_score,
            'alerts': alerts,
            'suggested_actions': self._get_mvp_suggested_actions(
                diagnosis.root_cause, diagnosis.user_behavior, alerts
            )
        }

    def _get_mvp_suggested_actions(self, root_cause: str, 
                                       user_behavior: Dict[str, Any],
                                       alerts: List[Dict[str, Any]]) -> List[str]:
        """Get MVP suggested actions"""

        action_mapping = {
            'Potential Entitlement/Auth Issue': [
                'Check user subscription status for the channel',
                'Verify authentication tokens and session validity',
                'Review entitlement service logs for errors',
                'Test user access from different device/location'
            ],
            'Potential CDN/Manifest Issue': [
                'Check CDN health and video manifest availability', 
                'Verify content delivery network routing',
                'Test video playback from same CDN edge server',
                'Review CDN logs for 5xx errors or timeouts'
            ],
            'Technical Investigation Needed': [
                'Review video start failure logs in detail',
                'Check for recent system changes or deployments',
                'Analyze session flow and error patterns', 
                'Escalate to video platform engineering team'
            ],
            'Potential Transient Network Issue': [
                'Monitor if pattern continues',
                'Check user network stability',
                'Review connection quality metrics',
                'No immediate action needed if isolated incident'
            ],
            'Persistent Technical Issue': [
                'Immediate investigation required',
                'Check for user-specific routing issues',
                'Review CDN performance for this user segment',
                'Consider proactive user support outreach'
            ]
        }
        
        actions = action_mapping.get(root_cause, [
            'Investigate failure using session details',
            'Check system health and recent changes',
            'Review logs for error patterns'
        ])
        
        pattern = user_behavior.get('pattern', '')
        
        if pattern == 'reliable_user_hit_issue':
            actions.insert(0, '🔴 HIGH PRIORITY: Reliable user affected - immediate investigation')
        elif pattern == 'deteriorating_experience':
            actions.insert(0, '🔴 URGENT: User experiencing service degradation - proactive outreach recommended')
        elif pattern == 'user_troubleshooting':
            actions.append('Note: User trying multiple devices - may indicate device-specific issue')
        
        if alerts:
            for alert in alerts:
                if alert['type'] == 'OUTAGE_ALERT':
                    actions.insert(0, '🚨 CRITICAL: Widespread outage detected - initiate incident response')
                elif alert['type'] == 'USER_EXPERIENCE_DEGRADATION':
                    actions.insert(0, '⚠️ Proactive support outreach recommended for this user')
        
        return actions

# ============================================================================
# CONVENIENCE FUNCTIONS - ORIGINAL NAMES MAINTAINED
# ============================================================================

def create_mvp_ticket_engine(sessions_df: pd.DataFrame, target_channels: List[str] = None, 
                           kpi_df: pd.DataFrame = None, advancetags_df: pd.DataFrame = None) -> AutoTicketMVP:
    """Factory function - original name maintained"""
    return AutoTicketMVP(
        df_sessions=sessions_df,
        df_kpi=kpi_df,
        df_advancetags=advancetags_df,
        target_channels=target_channels
    )

def generate_tickets_for_failures(sessions_df: pd.DataFrame, target_channels: List[str] = None,
                                 advancetags_df: pd.DataFrame = None) -> List[Dict[str, Any]]:
    """Convenience function - original name maintained"""
    engine = create_mvp_ticket_engine(sessions_df, target_channels, advancetags_df=advancetags_df)
    return engine.process()

def analyze_session_for_failure(session_row: pd.Series) -> bool:
    """Utility function - original name maintained"""
    engine = AutoTicketMVP(pd.DataFrame([session_row]))
    return engine._is_video_start_failure(session_row)

def diagnose_session_failure(session_row: pd.Series, session_id: str, viewer_id: str = 'unknown_viewer') -> Diagnosis:
    """Utility function - original name, enhanced signature"""
    engine = AutoTicketMVP(pd.DataFrame([session_row]))
    return engine._apply_mvp_diagnosis_rules(session_row, viewer_id, session_id)

# ============================================================================
# BATCH PROCESSING FOR MULTIPLE VIEWERS
# ============================================================================

def process_multiple_viewers(viewers_data: Dict[str, Dict[str, pd.DataFrame]], 
                            target_channels: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Process multiple viewers at once"""
    all_tickets = {}
    
    for viewer_id, data in viewers_data.items():
        sessions_df = data.get('sessions', pd.DataFrame())
        advancetags_df = data.get('advancetags', pd.DataFrame())
        
        if sessions_df.empty:
            logger.warning(f"No session data for viewer {viewer_id}")
            all_tickets[viewer_id] = []
            continue
        
        try:
            engine = AutoTicketMVP(
                df_sessions=sessions_df,
                df_advancetags=advancetags_df,
                target_channels=target_channels
            )
            
            tickets = engine.process()
            all_tickets[viewer_id] = tickets
            logger.info(f"Generated {len(tickets)} tickets for viewer {viewer_id}")
            
        except Exception as e:
            logger.error(f"Error processing viewer {viewer_id}: {e}")
            all_tickets[viewer_id] = []
    
    return all_tickets

# ============================================================================
# SUMMARY AND REPORTING
# ============================================================================

def get_ticket_summary(tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get comprehensive ticket summary statistics"""
    if not tickets:
        return {
            'total_tickets': 0,
            'by_root_cause': {},
            'by_severity': {},
            'by_confidence_range': {},
            'by_user_pattern': {},
            'avg_confidence': 0.0,
            'avg_severity': 0.0
        }
    
    root_causes = [t['failure_details']['root_cause'] for t in tickets]
    severities = [t['severity_score'] for t in tickets]
    confidences = [t['confidence_score'] for t in tickets]
    user_patterns = [t['failure_details']['user_behavior'].get('pattern', 'unknown') for t in tickets]
    
    confidence_ranges = []
    for c in confidences:
        if c >= 0.8:
            confidence_ranges.append('High (80%+)')
        elif c >= 0.6:
            confidence_ranges.append('Medium (60-80%)')
        else:
            confidence_ranges.append('Low (<60%)')
    
    return {
        'total_tickets': len(tickets),
        'by_root_cause': dict(Counter(root_causes)),
        'by_severity': dict(Counter(severities)),
        'by_confidence_range': dict(Counter(confidence_ranges)),
        'by_user_pattern': dict(Counter(user_patterns)),
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
        'avg_severity': sum(severities) / len(severities) if severities else 0,
        'critical_tickets': len([s for s in severities if s >= 8]),
        'high_confidence_tickets': len([c for c in confidences if c >= 0.8]),
        'deteriorating_users': len([p for p in user_patterns if p == 'deteriorating_experience']),
        'reliable_users_affected': len([p for p in user_patterns if p == 'reliable_user_hit_issue'])
    }

def print_ticket_summary(tickets: List[Dict[str, Any]]):
    """Print human-readable ticket summary"""
    summary = get_ticket_summary(tickets)
    
    print(f"\n{'='*70}")
    print(f"ENHANCED TICKET GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tickets Generated: {summary['total_tickets']}")
    print(f"Average Confidence: {summary['avg_confidence']:.1%}")
    print(f"Average Severity: {summary['avg_severity']:.1f}/10")
    print(f"Critical Tickets (≥8): {summary['critical_tickets']}")
    print(f"High Confidence (≥80%): {summary['high_confidence_tickets']}")
    
    print(f"\nUser Impact Analysis:")
    print(f"  - Deteriorating Users: {summary['deteriorating_users']}")
    print(f"  - Reliable Users Affected: {summary['reliable_users_affected']}")
    
    print(f"\nBy Root Cause:")
    for cause, count in summary['by_root_cause'].items():
        print(f"  - {cause}: {count}")
    
    print(f"\nBy User Pattern:")
    for pattern, count in summary['by_user_pattern'].items():
        print(f"  - {pattern}: {count}")
    
    print(f"\nBy Confidence Range:")
    for range_name, count in summary['by_confidence_range'].items():
        print(f"  - {range_name}: {count}")
    print(f"{'='*70}\n")
