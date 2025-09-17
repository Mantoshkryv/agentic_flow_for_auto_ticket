# data_validation.py - COMPLETE AND FINAL VERSION
"""
Comprehensive data validation and preprocessing module for ticket management system.
Handles validation, cleaning, and quality monitoring for both uploaded files and database data.
COMPLETE VERSION - All methods implemented, no truncations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
import re
from pathlib import Path
import psutil
import os
import ipaddress

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """Define validation rules for data columns."""
    column_name: str
    data_type: str
    required: bool = True
    allowed_values: Optional[List] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    max_length: Optional[int] = None
    custom_validator: Optional[callable] = None

@dataclass
class DataSchema:
    """Define expected schema for datasets."""
    name: str
    rules: List[ValidationRule]
    min_rows: int = 1
    max_rows: Optional[int] = None
    allow_extra_columns: bool = True

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processed_rows: int = 0
    failed_rows: int = 0
    quality_score: float = 0.0
    processing_time: float = 0.0

class DataQualityMonitor:
    """Monitor and report data quality metrics."""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def analyze_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Comprehensive data quality analysis."""
        start_time = datetime.now()
        
        metrics = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'completeness': self._calculate_completeness(df),
            'uniqueness': self._calculate_uniqueness(df),
            'validity': self._calculate_validity(df),
            'consistency': self._calculate_consistency(df),
            'anomalies': self._detect_anomalies(df),
            'data_types': self._analyze_data_types(df),
            'analysis_time': (datetime.now() - start_time).total_seconds()
        }
        
        metrics['quality_score'] = self._calculate_quality_score(metrics)
        self.quality_metrics[dataset_name] = metrics
        return metrics
    
    def _calculate_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate completeness score for each column."""
        completeness = {}
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            completeness[col] = (non_null_count / total_count) * 100 if total_count > 0 else 0
        
        completeness['overall'] = np.mean(list(completeness.values())) if completeness else 0
        return completeness
    
    def _calculate_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate uniqueness metrics."""
        uniqueness = {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0,
            'column_uniqueness': {}
        }
        
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = df[col].notna().sum()
            uniqueness['column_uniqueness'][col] = {
                'unique_values': unique_count,
                'uniqueness_percentage': (unique_count / total_count) * 100 if total_count > 0 else 0
            }
        
        return uniqueness
    
    def _calculate_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate validity metrics based on data types and patterns."""
        validity = {
            'invalid_emails': 0,
            'invalid_dates': 0,
            'invalid_numbers': 0,
            'empty_strings': 0
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            if 'email' in col_lower:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_emails = df[col].notna() & ~df[col].str.match(email_pattern, na=False)
                validity['invalid_emails'] += invalid_emails.sum()
            
            datetime_fields = ['session_start_time', 'session_end_time', 'video_start_time', 'timestamp', 'created_at']
            if any(field in col_lower for field in datetime_fields):
                try:
                    pd.to_datetime(df[col], errors='raise')
                except:
                    validity['invalid_dates'] += df[col].notna().sum()
            
            if df[col].dtype == 'object':
                empty_strings = df[col].str.strip().eq('').sum()
                validity['empty_strings'] += empty_strings
        
        return validity
    
    def _calculate_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate consistency metrics."""
        consistency = {
            'data_type_consistency': {},
            'format_consistency': {},
            'value_consistency': {}
        }
        
        for col in df.columns:
            if df[col].dtype == 'object':
                non_null_values = df[col].dropna().astype(str)
                if len(non_null_values) > 0:
                    patterns = non_null_values.apply(lambda x: re.sub(r'\w', 'X', re.sub(r'\d', '9', x)))
                    pattern_variety = patterns.nunique() / len(patterns) if len(patterns) > 0 else 0
                    consistency['format_consistency'][col] = {
                        'pattern_variety': pattern_variety,
                        'consistent': pattern_variety < 0.1
                    }
        
        return consistency
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect various types of data anomalies."""
        anomalies = {
            'outliers': {},
            'suspicious_patterns': {},
            'statistical_anomalies': {}
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                    anomalies['outliers'][col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(df)) * 100,
                        'values': outliers[col].tolist()[:10]
                    }
        
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                value_counts = non_null_values.value_counts()
                if len(value_counts) > 0:
                    most_common_value = value_counts.index[0]
                    most_common_count = value_counts.iloc[0]
                    
                    if most_common_count / len(non_null_values) > 0.5:
                        anomalies['suspicious_patterns'][col] = {
                            'dominant_value': most_common_value,
                            'frequency': most_common_count,
                            'percentage': (most_common_count / len(non_null_values)) * 100
                        }
        
        return anomalies
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze data types of each column."""
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        scores = []
        
        completeness_score = metrics['completeness'].get('overall', 0)
        scores.append(completeness_score * 0.3)
        
        duplicate_penalty = min(metrics['uniqueness']['duplicate_percentage'], 20)
        uniqueness_score = max(0, 100 - duplicate_penalty)
        scores.append(uniqueness_score * 0.2)
        
        validity_issues = sum(metrics['validity'].values())
        total_cells = metrics['total_rows'] * metrics['total_columns']
        validity_score = max(0, 100 - (validity_issues / total_cells * 100)) if total_cells > 0 else 100
        scores.append(validity_score * 0.25)
        
        total_outliers = sum(
            anomaly.get('count', 0) 
            for anomaly in metrics['anomalies'].get('outliers', {}).values()
        )
        anomaly_penalty = min((total_outliers / metrics['total_rows']) * 100, 25) if metrics['total_rows'] > 0 else 0
        anomaly_score = max(0, 100 - anomaly_penalty)
        scores.append(anomaly_score * 0.25)
        
        return sum(scores)

class MemoryManager:
    """Manage memory usage during data processing."""
    
    def __init__(self, max_memory_mb: int = 1000, chunk_size: int = 10000):
        self.max_memory_mb = max_memory_mb
        self.chunk_size = chunk_size
        self.current_memory_mb = 0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        current_usage = self.get_memory_usage()
        return current_usage < self.max_memory_mb
    
    def process_in_chunks(self, df: pd.DataFrame, processor_func) -> List[Any]:
        """Process DataFrame in chunks to manage memory."""
        results = []
        total_rows = len(df)
        processed = 0
        
        logger.info(f"Processing {total_rows:,} rows in chunks of {self.chunk_size:,}")
        
        while processed < total_rows:
            if not self.check_memory_limit():
                logger.warning(f"Memory limit exceeded: {self.get_memory_usage():.1f}MB")
                break
            
            end_idx = min(processed + self.chunk_size, total_rows)
            chunk = df.iloc[processed:end_idx]
            
            try:
                chunk_results = processor_func(chunk)
                results.extend(chunk_results)
                processed = end_idx
                
                progress = (processed / total_rows) * 100
                logger.info(f"Processed {processed:,}/{total_rows:,} rows ({progress:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error processing chunk {processed}-{end_idx}: {e}")
                processed = end_idx
        
        return results

class DataValidator:
    """Comprehensive data validator for ticket management system."""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.quality_monitor = DataQualityMonitor()
        self.schemas = self._define_schemas()
    
    def _validate_ip_address(self, ip_value: Any) -> bool:
        """Custom validator for IP addresses."""
        if pd.isna(ip_value):
            return True
        
        ip_str = str(ip_value).strip()
        if not ip_str:
            return True
        
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
    
    def _validate_ipv6_address(self, ipv6_value: Any) -> bool:
        """Custom validator for IPv6 addresses."""
        if pd.isna(ipv6_value):
            return True
        
        ipv6_str = str(ipv6_value).strip()
        if not ipv6_str:
            return True
        
        try:
            ipaddress.IPv6Address(ipv6_str)
            return True
        except ValueError:
            return False
    
    def _define_schemas(self) -> Dict[str, DataSchema]:
        """Define validation schemas PERFECTLY ALIGNED with Django models."""
        
        session_schema = DataSchema(
            name="sessions",
            rules=[
                ValidationRule("viewer_id", "string", required=False, max_length=200),
                ValidationRule("session_id", "string", required=True, max_length=200, 
                             pattern=r"^[A-Za-z0-9-_]+$"),
                ValidationRule("session_start_time", "datetime", required=True),
                ValidationRule("status", "string", required=True, max_length=50),
                ValidationRule("ended_status", "string", required=False, max_length=200,
                             allowed_values=["VSF-T", "VSF-B", "EBVS", "SUCCESS"]),
                ValidationRule("video_start_failure", "boolean", required=False),
                ValidationRule("asset_name", "string", required=False, max_length=300),
                ValidationRule("channel", "string", required=False, max_length=200),
                ValidationRule("starting_bitrate", "float", required=False, min_value=0),
                ValidationRule("playing_time", "float", required=False, min_value=0, max_value=86400),
                ValidationRule("rebuffering_ratio", "float", required=False, min_value=0, max_value=1),
                ValidationRule("avg_peak_bitrate", "float", required=False, min_value=0),
                ValidationRule("avg_average_bitrate", "float", required=False, min_value=0),
                ValidationRule("average_framerate", "float", required=False, min_value=0),
                ValidationRule("connection_induced_rebuffering_ratio", "float", required=False, min_value=0, max_value=1),
                ValidationRule("total_video_restart_time", "float", required=False, min_value=0),
                ValidationRule("bitrate_switches", "int", required=False, min_value=0),
                ValidationRule("ended_session", "boolean", required=False),
                ValidationRule("impacted_session", "boolean", required=False),
                ValidationRule("exit_before_video_starts", "boolean", required=False),
                ValidationRule("session_end_time", "datetime", required=False),
                ValidationRule("video_start_time", "datetime", required=False),
                ValidationRule("created_at", "datetime", required=False),
                ValidationRule("updated_at", "datetime", required=False),
            ],
            min_rows=1,
            allow_extra_columns=True
        )
        
        kpi_schema = DataSchema(
            name="kpi_data",
            rules=[
                ValidationRule("timestamp", "datetime", required=True),
                ValidationRule("plays", "int", required=False, min_value=0),
                ValidationRule("video_start_failures_technical", "int", required=False, min_value=0),
                ValidationRule("video_start_failures_business", "int", required=False, min_value=0),
                ValidationRule("exit_before_video_starts", "int", required=False, min_value=0),
                ValidationRule("video_playback_failures_technical", "int", required=False, min_value=0),
                ValidationRule("video_playback_failures_business", "int", required=False, min_value=0),
                ValidationRule("playing_time_ended_mins", "float", required=False, min_value=0),
                ValidationRule("streaming_performance_index", "float", required=False, 
                             min_value=0, max_value=100),
                ValidationRule("video_start_time_sec", "float", required=False, min_value=0),
                ValidationRule("rebuffering_ratio_pct", "float", required=False, min_value=0, max_value=100),
                ValidationRule("connection_induced_rebuffering_ratio_pct", "float", required=False, min_value=0, max_value=100),
                ValidationRule("video_restart_time_sec", "float", required=False, min_value=0),
                ValidationRule("avg_peak_bitrate_mbps", "float", required=False, min_value=0),
                ValidationRule("created_at", "datetime", required=False),
            ],
            min_rows=1,
            allow_extra_columns=True
        )
        
        advancetags_schema = DataSchema(
            name="advancetags",
            rules=[
                ValidationRule("session_id", "string", required=True, max_length=200,
                             pattern=r"^[A-Za-z0-9-_]+$"),
                ValidationRule("asset_name", "string", required=False, max_length=300),
                ValidationRule("content_category", "string", required=False, max_length=200),
                ValidationRule("browser_name", "string", required=False, max_length=100),
                ValidationRule("browser_version", "string", required=False, max_length=100),
                ValidationRule("device_hardware_type", "string", required=False, max_length=100),
                ValidationRule("device_manufacturer", "string", required=False, max_length=100),
                ValidationRule("device_marketing_name", "string", required=False, max_length=200),
                ValidationRule("device_model", "string", required=False, max_length=100),
                ValidationRule("device_name", "string", required=False, max_length=200),
                ValidationRule("device_operating_system", "string", required=False, max_length=100),
                ValidationRule("device_operating_system_family", "string", required=False, max_length=100),
                ValidationRule("device_operating_system_version", "string", required=False, max_length=100),
                ValidationRule("app_name", "string", required=False, max_length=100),
                ValidationRule("app_version", "string", required=False, max_length=100),
                ValidationRule("player_framework_name", "string", required=False, max_length=100),
                ValidationRule("player_framework_version", "string", required=False, max_length=100),
                ValidationRule("last_cdn", "string", required=False, max_length=200),
                ValidationRule("cdn", "string", required=False, max_length=200),
                ValidationRule("channel", "string", required=False, max_length=200),
                ValidationRule("city", "string", required=False, max_length=100),
                ValidationRule("state", "string", required=False, max_length=100),
                ValidationRule("country", "string", required=False, max_length=100),
                ValidationRule("address", "string", required=False, max_length=300),
                ValidationRule("isp_name", "string", required=False, max_length=200),
                ValidationRule("asn_name", "string", required=False, max_length=200),
                ValidationRule("ip_address", "string", required=False, custom_validator=self._validate_ip_address),
                ValidationRule("ipv6_address", "string", required=False, custom_validator=self._validate_ipv6_address),
                ValidationRule("stream_url", "string", required=False),
                ValidationRule("created_at", "datetime", required=False),
            ],
            min_rows=1,
            allow_extra_columns=True
        )
        
        return {
            "session": session_schema,
            "kpi_data": kpi_schema,
            "advancetags": advancetags_schema,

        }
    
    def validate_dataframe(self, df: pd.DataFrame, schema_name: str) -> ValidationResult:
        """Validate DataFrame against specified schema."""
        start_time = datetime.now()
        
        if schema_name not in self.schemas:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unknown schema: {schema_name}. Available: {list(self.schemas.keys())}"]
            )
        
        schema = self.schemas[schema_name]
        result = ValidationResult()
        
        logger.info(f"Validating {schema_name} data: {len(df)} rows, {len(df.columns)} columns")
        
        self._validate_structure(df, schema, result)
        self._validate_row_count(df, schema, result)
        
        if result.is_valid:
            self._validate_columns(df, schema, result)
        
        quality_metrics = self.quality_monitor.analyze_data_quality(df, schema_name)
        result.quality_score = quality_metrics['quality_score']
        
        if result.quality_score < 70:
            result.warnings.append(f"Low data quality score: {result.quality_score:.1f}/100")
        
        if schema_name in ['session', 'meta', 'advancetags']:
            self._validate_critical_fields(df, schema_name, result)
        
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Validation completed for {schema_name}: "
                   f"Valid={result.is_valid}, Quality={result.quality_score:.1f}/100, "
                   f"Errors={len(result.errors)}, Warnings={len(result.warnings)}")
        
        return result
    
    def _validate_critical_fields(self, df: pd.DataFrame, schema_name: str, result: ValidationResult):
        """Validate fields critical for ticket generation."""
        if schema_name == 'session':
            status_fields = ['ended_status', 'status', 'video_start_failure']
            has_status_field = any(field in df.columns for field in status_fields)
            if not has_status_field:
                result.warnings.append(
                    f"No failure detection fields found ({status_fields}). "
                    "Ticket generation may not work properly."
                )
            
            if 'session_id' not in df.columns:
                result.errors.append("Missing required field: session_id")
        
        elif schema_name in ['meta', 'advancetags']:
            correlation_fields = ['isp_name', 'cdn', 'city']
            missing_correlation = [field for field in correlation_fields if field not in df.columns]
            if missing_correlation:
                result.warnings.append(
                    f"Missing correlation fields {missing_correlation}. "
                    "Advanced ticket diagnosis may be limited."
                )
            
            if 'session_id' not in df.columns:
                result.errors.append("Missing required field: session_id (needed for merging with session data)")
    
    def _validate_structure(self, df: pd.DataFrame, schema: DataSchema, result: ValidationResult):
        """Validate DataFrame structure."""
        required_columns = [rule.column_name for rule in schema.rules if rule.required]
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            result.is_valid = False
            result.errors.append(f"Missing required columns: {sorted(missing_columns)}")
        
        if not schema.allow_extra_columns:
            schema_columns = {rule.column_name for rule in schema.rules}
            extra_columns = set(df.columns) - schema_columns
            if extra_columns:
                result.warnings.append(f"Extra columns found: {sorted(extra_columns)}")
    
    def _validate_row_count(self, df: pd.DataFrame, schema: DataSchema, result: ValidationResult):
        """Validate row count requirements."""
        row_count = len(df)
        
        if row_count < schema.min_rows:
            result.is_valid = False
            result.errors.append(f"Insufficient rows: {row_count} < {schema.min_rows}")
        
        if schema.max_rows and row_count > schema.max_rows:
            result.warnings.append(f"Row count exceeds recommended maximum: {row_count} > {schema.max_rows}")
    
    def _validate_columns(self, df: pd.DataFrame, schema: DataSchema, result: ValidationResult):
        """Validate individual columns."""
        for rule in schema.rules:
            if rule.column_name not in df.columns:
                continue
            
            column_data = df[rule.column_name]
            self._validate_column(column_data, rule, result)
    
    def _validate_column(self, column_data: pd.Series, rule: ValidationRule, result: ValidationResult):
        """Validate a single column."""
        column_name = rule.column_name
        
        if rule.data_type == "datetime":
            try:
                pd.to_datetime(column_data, errors='raise')
            except:
                invalid_count = column_data.notna().sum() - pd.to_datetime(column_data, errors='coerce').notna().sum()
                if invalid_count > 0:
                    result.errors.append(f"{column_name}: {invalid_count} invalid datetime values")
        
        elif rule.data_type == "boolean":
            valid_boolean_values = [True, False, 1, 0, '1', '0', 'true', 'false', 'True', 'False', 
                                  'yes', 'no', 'Yes', 'No', 'VSF-T', 'VSF-B', 'EBVS', 'SUCCESS']
            non_null_data = column_data.dropna()
            if len(non_null_data) > 0:
                invalid_booleans = ~non_null_data.isin(valid_boolean_values)
                if invalid_booleans.sum() > 0:
                    result.warnings.append(f"{column_name}: {invalid_booleans.sum()} values may need boolean conversion")
        
        elif rule.data_type in ["int", "float"]:
            numeric_data = pd.to_numeric(column_data, errors='coerce')
            invalid_count = column_data.notna().sum() - numeric_data.notna().sum()
            if invalid_count > 0:
                result.errors.append(f"{column_name}: {invalid_count} non-numeric values")
            
            valid_numeric = numeric_data.dropna()
            if rule.min_value is not None:
                below_min = (valid_numeric < rule.min_value).sum()
                if below_min > 0:
                    result.errors.append(f"{column_name}: {below_min} values below minimum {rule.min_value}")
            
            if rule.max_value is not None:
                above_max = (valid_numeric > rule.max_value).sum()
                if above_max > 0:
                    result.errors.append(f"{column_name}: {above_max} values above maximum {rule.max_value}")
        
        if rule.allowed_values:
            invalid_values = column_data[~column_data.isin(rule.allowed_values + [None, np.nan])]
            if len(invalid_values) > 0:
                unique_invalid = invalid_values.unique()[:5]
                result.errors.append(f"{column_name}: Invalid values found: {list(unique_invalid)}")
        
        if rule.pattern and column_data.dtype == 'object':
            non_null_data = column_data.dropna()
            if len(non_null_data) > 0:
                pattern_matches = non_null_data.str.match(rule.pattern, na=False)
                invalid_pattern_count = (~pattern_matches).sum()
                if invalid_pattern_count > 0:
                    result.errors.append(f"{column_name}: {invalid_pattern_count} values don't match pattern")
        
        if rule.max_length and column_data.dtype == 'object':
            non_null_data = column_data.dropna().astype(str)
            if len(non_null_data) > 0:
                too_long = non_null_data.str.len() > rule.max_length
                if too_long.sum() > 0:
                    result.warnings.append(f"{column_name}: {too_long.sum()} values exceed max length {rule.max_length}")
        
        if rule.custom_validator:
            try:
                non_null_data = column_data.dropna()
                if len(non_null_data) > 0:
                    invalid_custom = ~non_null_data.apply(rule.custom_validator)
                    if invalid_custom.sum() > 0:
                        result.errors.append(f"{column_name}: {invalid_custom.sum()} values failed custom validation")
            except Exception as e:
                result.warnings.append(f"{column_name}: Custom validation error: {str(e)}")

# Factory function for easy access
def create_validator() -> DataValidator:
    """Create a new DataValidator instance."""
    return DataValidator()

# Convenience functions for views.py compatibility
def validate_session_data(df: pd.DataFrame) -> ValidationResult:
    """Quick validation for session data."""
    validator = create_validator()
    return validator.validate_dataframe(df, "session")

def validate_kpi_data(df: pd.DataFrame) -> ValidationResult:
    """Quick validation for KPI data."""
    validator = create_validator()
    return validator.validate_dataframe(df, "kpi")

def validate_metadata(df: pd.DataFrame) -> ValidationResult:
    """Quick validation for metadata."""
    validator = create_validator()
    return validator.validate_dataframe(df, "meta")
