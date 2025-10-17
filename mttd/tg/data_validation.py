# data_validation.py - COMPLETE ENHANCED VERSION

"""
Enhanced Data Validation Pipeline
==================================

Features:
- Pure validation (NO cleaning - delegated to data_processing.py)
- Enhanced ticket structure validation
- Multi-layer diagnostics validation
- Session-only ticket support
- MongoDB compatibility
- Confidence/severity validation (0.0-1.0, 0-10)
- User behavior pattern validation
- Temporal and geographic analysis validation
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
import re
import ipaddress

logger = logging.getLogger(__name__)

# ============================================================================
# VALIDATION RULES AND CONFIGURATION
# ============================================================================

@dataclass
class EnhancedValidationRules:
    """Validation rules matching enhanced ticket engine"""
    
    # Confidence score validation (0.0-1.0)
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    
    # Severity score validation (0-10)
    severity_min: int = 0
    severity_max: int = 10
    
    # Required fields for session-only tickets
    session_required_fields: List[str] = None
    
    # Optional viewer_id (not required for session-only)
    viewer_id_optional: bool = True
    
    # User behavior patterns (from ticket_engine.py)
    valid_user_patterns: List[str] = None
    
    # User value tiers
    valid_user_value_tiers: List[str] = None
    
    # Temporal patterns
    valid_temporal_trends: List[str] = None
    valid_temporal_patterns: List[str] = None
    
    # Geographic validation
    validate_concurrent_failures: bool = True
    max_concurrent_failures: int = 100
    
    # Valid status values
    valid_session_statuses: List[str] = None
    
    # Valid severity labels
    valid_severity_labels: List[str] = None
    
    def __post_init__(self):
        if self.session_required_fields is None:
            self.session_required_fields = ['session_id']
        
        if self.valid_user_patterns is None:
            self.valid_user_patterns = [
                'reliable_user_hit_issue',
                'deteriorating_experience',
                'intermittent_issues',
                'normal_usage',
                'insufficient_data',
                'user_troubleshooting'
            ]
        
        if self.valid_user_value_tiers is None:
            self.valid_user_value_tiers = [
                'premium',
                'high_value',
                'regular',
                'casual',
                'unknown'
            ]
        
        if self.valid_temporal_trends is None:
            self.valid_temporal_trends = [
                'deteriorating',
                'stable',
                'normal',
                'intermittent',
                'insufficient_data',
                'unknown'
            ]
        
        if self.valid_temporal_patterns is None:
            self.valid_temporal_patterns = [
                'service_degradation',
                'isolated_incident',
                'intermittent_issues',
                'mixed_performance',
                'new_user',
                'insufficient_data'
            ]
        
        if self.valid_session_statuses is None:
            self.valid_session_statuses = [
                'Played', 'VSF-T', 'VSF-B', 'EBVS'
            ]
        
        if self.valid_severity_labels is None:
            self.valid_severity_labels = [
                'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
            ]

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processed_rows: int = 0
    failed_rows: int = 0
    quality_score: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# ENHANCED TICKET VALIDATOR
# ============================================================================

class EnhancedTicketValidator:
    """Validates enhanced ticket data structure from ticket_engine.py"""
    
    def __init__(self, rules: EnhancedValidationRules = None):
        self.rules = rules or EnhancedValidationRules()
        self.validation_stats = {
            'total_validated': 0,
            'valid_tickets': 0,
            'invalid_tickets': 0,
            'errors_by_type': {}
        }
    
    def validate_ticket_data(self, ticket: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate single ticket with enhanced fields"""
        errors = []
        warnings = []
        
        # 1. Session ID validation (REQUIRED for session-only tickets)
        if not ticket.get('session_id'):
            errors.append("Missing required field: session_id")
        else:
            session_id = str(ticket['session_id']).strip()
            if not session_id or session_id.lower() in ['nan', 'none', 'null']:
                errors.append(f"Invalid session_id: '{session_id}'")
            elif len(session_id) < 3:
                errors.append(f"session_id too short: '{session_id}'")
        
        # 2. Viewer ID validation (OPTIONAL for session-only)
        if 'viewer_id' in ticket and not self.rules.viewer_id_optional:
            if not ticket['viewer_id']:
                warnings.append("viewer_id is empty (session-only mode)")
        
        # 3. Confidence score validation
        confidence = ticket.get('confidence_score')
        if confidence is not None:
            try:
                confidence_float = float(confidence)
                if not (self.rules.confidence_min <= confidence_float <= self.rules.confidence_max):
                    errors.append(
                        f"confidence_score {confidence_float} outside valid range "
                        f"[{self.rules.confidence_min}, {self.rules.confidence_max}]"
                    )
            except (ValueError, TypeError):
                errors.append(f"confidence_score must be numeric, got {type(confidence).__name__}")
        else:
            warnings.append("confidence_score is missing")
        
        # 4. Severity score validation
        severity = ticket.get('severity_score')
        if severity is not None:
            try:
                severity_int = int(severity)
                if not (self.rules.severity_min <= severity_int <= self.rules.severity_max):
                    errors.append(
                        f"severity_score {severity_int} outside valid range "
                        f"[{self.rules.severity_min}, {self.rules.severity_max}]"
                    )
            except (ValueError, TypeError):
                errors.append(f"severity_score must be integer, got {type(severity).__name__}")
        else:
            warnings.append("severity_score is missing")
        
        # 5. Failure details validation
        failure_details = ticket.get('failure_details', {})
        if not isinstance(failure_details, dict):
            errors.append(f"failure_details must be dict, got {type(failure_details).__name__}")
        else:
            fd_errors, fd_warnings = self._validate_failure_details(failure_details)
            errors.extend(fd_errors)
            warnings.extend(fd_warnings)
        
        # 6. Context data validation
        if 'context_data' in ticket:
            if not isinstance(ticket['context_data'], dict):
                errors.append(f"context_data must be dict, got {type(ticket['context_data']).__name__}")
        
        # 7. Suggested actions validation
        if 'suggested_actions' in ticket:
            if not isinstance(ticket['suggested_actions'], list):
                errors.append(f"suggested_actions must be list, got {type(ticket['suggested_actions']).__name__}")
        
        # 8. Status validation
        if 'status' in ticket:
            valid_statuses = ['new', 'InProgress', 'resolved', 'closed']
            if ticket['status'] not in valid_statuses:
                warnings.append(f"Unknown status: {ticket['status']}")
        
        # 9. Priority validation
        if 'priority' in ticket:
            valid_priorities = ['low', 'medium', 'high', 'critical']
            if ticket['priority'] not in valid_priorities:
                warnings.append(f"Unknown priority: {ticket['priority']}")
        
        self.validation_stats['total_validated'] += 1
        if len(errors) == 0:
            self.validation_stats['valid_tickets'] += 1
        else:
            self.validation_stats['invalid_tickets'] += 1
            for error in errors:
                error_type = error.split(':')[0]
                self.validation_stats['errors_by_type'][error_type] = \
                    self.validation_stats['errors_by_type'].get(error_type, 0) + 1
        
        return len(errors) == 0, errors, warnings
    
    def _validate_failure_details(self, failure_details: dict) -> Tuple[List[str], List[str]]:
        """Validate nested failure_details structure"""
        errors = []
        warnings = []
        
        # Required fields in failure_details
        if 'root_cause' not in failure_details:
            warnings.append("failure_details.root_cause missing")
        
        if 'evidence' not in failure_details:
            warnings.append("failure_details.evidence missing")
        
        # Validate nested confidence (if present)
        if 'confidence' in failure_details:
            try:
                conf = float(failure_details['confidence'])
                if not (0.0 <= conf <= 1.0):
                    errors.append(f"failure_details.confidence {conf} outside [0.0, 1.0]")
            except (ValueError, TypeError):
                errors.append("failure_details.confidence must be numeric")
        
        # Validate nested severity_score (if present)
        if 'severity_score' in failure_details:
            try:
                sev = int(failure_details['severity_score'])
                if not (0 <= sev <= 10):
                    errors.append(f"failure_details.severity_score {sev} outside [0, 10]")
            except (ValueError, TypeError):
                errors.append("failure_details.severity_score must be integer")
        
        # Validate severity_label (if present)
        if 'severity_label' in failure_details:
            label = failure_details['severity_label']
            if label not in self.rules.valid_severity_labels:
                warnings.append(f"Unknown severity_label: {label}")
        
        # Validate user_behavior
        if 'user_behavior' in failure_details:
            ub_errors, ub_warnings = self._validate_user_behavior(
                failure_details['user_behavior']
            )
            errors.extend(ub_errors)
            warnings.extend(ub_warnings)
        
        # Validate temporal_analysis
        if 'temporal_analysis' in failure_details:
            ta_errors, ta_warnings = self._validate_temporal_analysis(
                failure_details['temporal_analysis']
            )
            errors.extend(ta_errors)
            warnings.extend(ta_warnings)
        
        # Validate geographic_analysis
        if 'geographic_analysis' in failure_details:
            ga_errors, ga_warnings = self._validate_geographic_analysis(
                failure_details['geographic_analysis']
            )
            errors.extend(ga_errors)
            warnings.extend(ga_warnings)
        
        return errors, warnings
    
    def _validate_user_behavior(self, user_behavior: Any) -> Tuple[List[str], List[str]]:
        """Validate user_behavior structure"""
        errors = []
        warnings = []
        
        if not isinstance(user_behavior, dict):
            errors.append(f"user_behavior must be dict, got {type(user_behavior).__name__}")
            return errors, warnings
        
        # Validate pattern
        if 'pattern' in user_behavior:
            pattern = user_behavior['pattern']
            if pattern not in self.rules.valid_user_patterns:
                warnings.append(f"Unknown user_behavior.pattern: {pattern}")
        
        # Validate user_value_tier
        if 'user_value_tier' in user_behavior:
            tier = user_behavior['user_value_tier']
            if tier not in self.rules.valid_user_value_tiers:
                warnings.append(f"Unknown user_value_tier: {tier}")
        
        # Validate confidence_modifier
        if 'confidence_modifier' in user_behavior:
            try:
                modifier = float(user_behavior['confidence_modifier'])
                if not (-1.0 <= modifier <= 1.0):
                    warnings.append(f"confidence_modifier {modifier} outside typical range [-1.0, 1.0]")
            except (ValueError, TypeError):
                errors.append("confidence_modifier must be numeric")
        
        # Validate severity_boost
        if 'severity_boost' in user_behavior:
            if not isinstance(user_behavior['severity_boost'], bool):
                warnings.append("severity_boost should be boolean")
        
        return errors, warnings
    
    def _validate_temporal_analysis(self, temporal_analysis: Any) -> Tuple[List[str], List[str]]:
        """Validate temporal_analysis structure"""
        errors = []
        warnings = []
        
        if not isinstance(temporal_analysis, dict):
            errors.append(f"temporal_analysis must be dict, got {type(temporal_analysis).__name__}")
            return errors, warnings
        
        # Validate trend
        if 'trend' in temporal_analysis:
            trend = temporal_analysis['trend']
            if trend not in self.rules.valid_temporal_trends:
                warnings.append(f"Unknown temporal_analysis.trend: {trend}")
        
        # Validate pattern
        if 'pattern' in temporal_analysis:
            pattern = temporal_analysis['pattern']
            if pattern not in self.rules.valid_temporal_patterns:
                warnings.append(f"Unknown temporal_analysis.pattern: {pattern}")
        
        # Validate consecutive_failures
        if 'consecutive_failures' in temporal_analysis:
            try:
                consecutive = int(temporal_analysis['consecutive_failures'])
                if consecutive < 0:
                    errors.append("consecutive_failures cannot be negative")
            except (ValueError, TypeError):
                errors.append("consecutive_failures must be integer")
        
        return errors, warnings
    
    def _validate_geographic_analysis(self, geographic_analysis: Any) -> Tuple[List[str], List[str]]:
        """Validate geographic_analysis structure"""
        errors = []
        warnings = []
        
        if not isinstance(geographic_analysis, dict):
            errors.append(f"geographic_analysis must be dict, got {type(geographic_analysis).__name__}")
            return errors, warnings
        
        # Validate is_widespread
        if 'is_widespread' in geographic_analysis:
            if not isinstance(geographic_analysis['is_widespread'], bool):
                warnings.append("is_widespread should be boolean")
        
        # Validate concurrent_count
        if 'concurrent_count' in geographic_analysis:
            try:
                count = int(geographic_analysis['concurrent_count'])
                if count < 0:
                    errors.append("concurrent_count cannot be negative")
                elif count > self.rules.max_concurrent_failures:
                    warnings.append(f"Unusually high concurrent_count: {count}")
            except (ValueError, TypeError):
                errors.append("concurrent_count must be integer")
        
        return errors, warnings
    
    def validate_ticket_batch(self, tickets: List[Dict[str, Any]]) -> ValidationResult:
        """Validate batch of tickets"""
        start_time = datetime.now()
        
        all_errors = []
        all_warnings = []
        valid_count = 0
        invalid_count = 0
        
        for i, ticket in enumerate(tickets):
            is_valid, errors, warnings = self.validate_ticket_data(ticket)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                all_errors.append(f"Ticket {i} ({ticket.get('ticket_id', 'unknown')}): {errors}")
            
            if warnings:
                all_warnings.append(f"Ticket {i}: {warnings}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        quality_score = (valid_count / len(tickets)) * 100 if tickets else 0
        
        return ValidationResult(
            is_valid=(invalid_count == 0),
            errors=all_errors,
            warnings=all_warnings,
            processed_rows=len(tickets),
            failed_rows=invalid_count,
            quality_score=quality_score,
            processing_time=processing_time,
            metadata={
                'valid_tickets': valid_count,
                'invalid_tickets': invalid_count,
                'validation_stats': self.validation_stats
            }
        )

# ============================================================================
# SESSION DATA VALIDATOR
# ============================================================================

class SessionDataValidator:
    """Validates session data matching models.py structure"""
    
    def __init__(self):
        self.valid_statuses = ['Played', 'VSF-T', 'VSF-B', 'EBVS']
        self.failure_statuses = ['VSF-T', 'VSF-B', 'EBVS']
    
    def validate_sessions_df(self, df: pd.DataFrame) -> ValidationResult:
        """Validate sessions dataframe"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        if df.empty:
            errors.append("Sessions dataframe is empty")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                processed_rows=0
            )
        
        # 1. Check for session_id column
        session_id_col = self._find_session_id_column(df)
        if not session_id_col:
            errors.append("No session_id column found. Expected one of: session_id, Session ID, Session Id")
        else:
            # Validate session_id values
            null_count = df[session_id_col].isnull().sum()
            if null_count > 0:
                warnings.append(f"{null_count} rows have null session_id")
            
            # Check for duplicates
            duplicates = df[session_id_col].duplicated().sum()
            if duplicates > 0:
                warnings.append(f"{duplicates} duplicate session_ids found")
        
        # 2. Validate Status/Ended Status values
        status_col = self._find_status_column(df)
        if status_col:
            invalid_statuses = df[~df[status_col].isin(self.valid_statuses + [np.nan])][status_col].unique()
            if len(invalid_statuses) > 0:
                warnings.append(f"Unknown status values: {list(invalid_statuses)[:5]}")
        else:
            warnings.append("No status column found")
        
        # 3. Check for failure indicators
        failure_indicators = self._check_failure_indicators(df)
        if not failure_indicators['has_any']:
            warnings.append("No failure indicator columns found")
        
        # 4. Validate timestamps
        timestamp_warnings = self._validate_timestamps(df)
        warnings.extend(timestamp_warnings)
        
        # 5. Validate numeric fields
        numeric_warnings = self._validate_numeric_fields(df)
        warnings.extend(numeric_warnings)
        
        # 6. Check for required fields
        required_fields = ['session_id', 'Session ID']
        has_required = any(field in df.columns for field in required_fields)
        if not has_required:
            errors.append("Missing required session identifier column")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        quality_score = self._calculate_quality_score(df, errors, warnings)
        
        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            processed_rows=len(df),
            failed_rows=len(df[df[session_id_col].isnull()]) if session_id_col else 0,
            quality_score=quality_score,
            processing_time=processing_time,
            metadata=failure_indicators
        )
    
    def _find_session_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find session_id column with flexible naming"""
        possible_names = ['session_id', 'Session ID', 'Session Id', 'sessionid', 'SessionID']
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _find_status_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find status column"""
        possible_names = ['Status', 'status', 'Ended Status', 'ended_status']
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _check_failure_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for failure indicator columns"""
        indicators = {
            'has_video_start_failure': False,
            'has_exit_before_video_starts': False,
            'has_status': False,
            'has_any': False
        }
        
        vsf_cols = ['Video Start Failure', 'video_start_failure']
        ebvs_cols = ['Exit Before Video Starts', 'exit_before_video_starts']
        
        indicators['has_video_start_failure'] = any(col in df.columns for col in vsf_cols)
        indicators['has_exit_before_video_starts'] = any(col in df.columns for col in ebvs_cols)
        indicators['has_status'] = self._find_status_column(df) is not None
        indicators['has_any'] = any([
            indicators['has_video_start_failure'],
            indicators['has_exit_before_video_starts'],
            indicators['has_status']
        ])
        
        return indicators
    
    def _validate_timestamps(self, df: pd.DataFrame) -> List[str]:
        """Validate timestamp columns"""
        warnings = []
        
        time_cols = ['Session Start Time', 'session_start_time', 
                    'Session End Time', 'session_end_time']
        
        for col in time_cols:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    warnings.append(f"Could not parse {col} as datetime: {str(e)}")
        
        return warnings
    
    def _validate_numeric_fields(self, df: pd.DataFrame) -> List[str]:
        """Validate numeric fields"""
        warnings = []
        
        numeric_fields = [
            'Playing Time', 'playing_time',
            'Video Start Time', 'video_start_time',
            'Rebuffering Ratio', 'rebuffering_ratio',
            'Starting Bitrate', 'starting_bitrate'
        ]
        
        for field in numeric_fields:
            if field in df.columns:
                non_numeric = pd.to_numeric(df[field], errors='coerce').isnull().sum()
                total = df[field].notna().sum()
                if non_numeric > 0 and total > 0:
                    ratio = non_numeric / total
                    if ratio > 0.1:  # More than 10% non-numeric
                        warnings.append(f"{field} has {non_numeric} non-numeric values ({ratio:.1%})")
        
        return warnings
    
    def _calculate_quality_score(self, df: pd.DataFrame, errors: List[str], warnings: List[str]) -> float:
        """Calculate data quality score"""
        score = 100.0
        
        # Deduct for errors
        score -= len(errors) * 20
        
        # Deduct for warnings
        score -= len(warnings) * 5
        
        # Bonus for completeness
        completeness = df.notna().mean().mean() * 100
        score = (score + completeness) / 2
        
        return max(0.0, min(100.0, score))

# ============================================================================
# KPI DATA VALIDATOR
# ============================================================================

class KPIDataValidator:
    """Validates KPI data structure"""
    
    def validate_kpi_df(self, df: pd.DataFrame) -> ValidationResult:
        """Validate KPI dataframe"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        if df.empty:
            errors.append("KPI dataframe is empty")
            return ValidationResult(is_valid=False, errors=errors, processed_rows=0)
        
        # Check for timestamp
        timestamp_cols = ['timestamp', 'Timestamp', 'time', 'Time']
        has_timestamp = any(col in df.columns for col in timestamp_cols)
        if not has_timestamp:
            errors.append("No timestamp column found")
        
        # Check for plays
        plays_cols = ['plays', 'Plays', 'play_count']
        has_plays = any(col in df.columns for col in plays_cols)
        if not has_plays:
            warnings.append("No plays column found")
        
        # Validate numeric fields
        numeric_fields = [
            'Video Start Failures Technical',
            'Video Start Failures Business',
            'Exit Before Video Starts'
        ]
        
        for field in numeric_fields:
            if field in df.columns:
                non_numeric = pd.to_numeric(df[field], errors='coerce').isnull().sum()
                if non_numeric > 0:
                    warnings.append(f"{field} has {non_numeric} non-numeric values")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            processed_rows=len(df),
            quality_score=self._calculate_kpi_quality(df, errors, warnings),
            processing_time=processing_time
        )
    
    def _calculate_kpi_quality(self, df: pd.DataFrame, errors: List[str], warnings: List[str]) -> float:
        """Calculate KPI data quality score"""
        score = 100.0
        score -= len(errors) * 20
        score -= len(warnings) * 5
        
        completeness = df.notna().mean().mean() * 100
        score = (score + completeness) / 2
        
        return max(0.0, min(100.0, score))

# ============================================================================
# ADVANCETAGS VALIDATOR
# ============================================================================

class AdvancetagsValidator:
    """Validates advancetags/metadata structure"""
    
    def validate_advancetags_df(self, df: pd.DataFrame) -> ValidationResult:
        """Validate advancetags dataframe"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        if df.empty:
            errors.append("Advancetags dataframe is empty")
            return ValidationResult(is_valid=False, errors=errors, processed_rows=0)
        
        # Check for session_id
        session_id_cols = ['session_id', 'Session ID', 'Session Id']
        has_session_id = any(col in df.columns for col in session_id_cols)
        if not has_session_id:
            errors.append("No session_id column found")
        
        # Validate IP addresses if present
        ip_cols = ['IP', 'ip', 'IPv6', 'ipv6']
        for ip_col in ip_cols:
            if ip_col in df.columns:
                invalid_ips = self._validate_ip_column(df[ip_col])
                if invalid_ips > 0:
                    warnings.append(f"{ip_col} has {invalid_ips} invalid IP addresses")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            processed_rows=len(df),
            quality_score=self._calculate_advancetags_quality(df, errors, warnings),
            processing_time=processing_time
        )
    
    def _validate_ip_column(self, ip_series: pd.Series) -> int:
        """Validate IP addresses in a column"""
        invalid_count = 0
        for ip_val in ip_series.dropna():
            try:
                ipaddress.ip_address(str(ip_val).strip())
            except ValueError:
                invalid_count += 1
        return invalid_count
    
    def _calculate_advancetags_quality(self, df: pd.DataFrame, errors: List[str], warnings: List[str]) -> float:
        """Calculate advancetags quality score"""
        score = 100.0
        score -= len(errors) * 20
        score -= len(warnings) * 5
        
        completeness = df.notna().mean().mean() * 100
        score = (score + completeness) / 2
        
        return max(0.0, min(100.0, score))

# ============================================================================
# UNIFIED VALIDATOR
# ============================================================================

class ComprehensiveDataValidator:
    """Main validator orchestrating all validation operations"""
    
    def __init__(self, rules: EnhancedValidationRules = None):
        self.rules = rules or EnhancedValidationRules()
        self.ticket_validator = EnhancedTicketValidator(self.rules)
        self.session_validator = SessionDataValidator()
        self.kpi_validator = KPIDataValidator()
        self.advancetags_validator = AdvancetagsValidator()
        self.validation_history = []
    
    def validate_all_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, ValidationResult]:
        """Validate all data types"""
        results = {}
        
        if 'sessions' in data and not data['sessions'].empty:
            results['sessions'] = self.session_validator.validate_sessions_df(data['sessions'])
        
        if 'kpi_data' in data and not data['kpi_data'].empty:
            results['kpi_data'] = self.kpi_validator.validate_kpi_df(data['kpi_data'])
        
        if 'advancetags' in data and not data['advancetags'].empty:
            results['advancetags'] = self.advancetags_validator.validate_advancetags_df(data['advancetags'])
        
        return results
    
    def validate_tickets(self, tickets: List[Dict[str, Any]]) -> ValidationResult:
        """Validate enhanced tickets"""
        return self.ticket_validator.validate_ticket_batch(tickets)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations"""
        if not self.validation_history:
            return {'message': 'No validations performed yet'}
        
        total_validations = len(self.validation_history)
        successful = sum(1 for h in self.validation_history if h['result'].is_valid)
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful,
            'success_rate': (successful / total_validations) * 100,
            'latest_validation': self.validation_history[-1]['timestamp'],
            'total_rows_validated': sum(h['result'].processed_rows for h in self.validation_history)
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_enhanced_tickets(tickets: List[Dict[str, Any]]) -> ValidationResult:
    """Validate list of enhanced tickets"""
    validator = EnhancedTicketValidator()
    return validator.validate_ticket_batch(tickets)

def validate_session_data(df: pd.DataFrame) -> ValidationResult:
    """Quick session validation"""
    validator = SessionDataValidator()
    return validator.validate_sessions_df(df)

def validate_kpi_data(df: pd.DataFrame) -> ValidationResult:
    """Quick KPI validation"""
    validator = KPIDataValidator()
    return validator.validate_kpi_df(df)

def validate_metadata(df: pd.DataFrame) -> ValidationResult:
    """Quick advancetags validation"""
    validator = AdvancetagsValidator()
    return validator.validate_advancetags_df(df)

def create_data_validator(rules: EnhancedValidationRules = None) -> ComprehensiveDataValidator:
    """Factory function to create comprehensive validator"""
    return ComprehensiveDataValidator(rules)

def validate_all_pipeline_data(data: Dict[str, pd.DataFrame]) -> Dict[str, ValidationResult]:
    """Validate all data in pipeline"""
    validator = create_data_validator()
    return validator.validate_all_data(data)
