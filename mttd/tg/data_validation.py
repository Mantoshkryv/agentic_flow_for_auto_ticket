# data_validation.py - ENHANCED WITH COMPREHENSIVE CLEANING AND VALIDATION

"""
Enhanced Data Validation and Preprocessing
=========================================

Features:
- Removes blank columns and rows intelligently
- Eliminates instruction headers and unnecessary content
- Advanced data quality monitoring
- Memory-efficient processing
- Session-only validation support
- Variable channel validation
- Prevents code duplicacy through imports
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
import gc

# Import existing functions to prevent duplicacy
try:
    from .models import Session, KPI, Advancetags, Ticket
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND RESULT CLASSES
# ============================================================================

@dataclass
class ValidationRule:
    """Define validation rules for data columns"""
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
    """Define expected schema for datasets"""
    name: str
    rules: List[ValidationRule]
    min_rows: int = 1
    max_rows: Optional[int] = None
    allow_extra_columns: bool = True

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
    cleaned_columns: int = 0
    removed_rows: int = 0

@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations"""
    remove_blank_columns_threshold: float = 0.95  # Remove if 95%+ blank
    remove_blank_rows_threshold: float = 0.90     # Remove if 90%+ blank
    remove_instruction_rows: bool = True
    clean_whitespace: bool = True
    standardize_column_names: bool = True
    remove_duplicate_rows: bool = True
    handle_mixed_data_types: bool = True

# ============================================================================
# ENHANCED DATA CLEANER WITH INSTRUCTION REMOVAL
# ============================================================================

class EnhancedDataCleaner:
    """Advanced data cleaner with intelligent preprocessing"""
    
    def __init__(self, config: CleaningConfig = None):
        self.config = config or CleaningConfig()
        self.cleaning_stats = {
            'original_shape': None,
            'final_shape': None,
            'removed_columns': 0,
            'removed_rows': 0,
            'cleaned_instructions': 0,
            'processing_time': 0.0
        }
        
        # Patterns for identifying instruction rows and headers
        self.instruction_patterns = [
            r'instructions?:.*',
            r'note:.*', 
            r'please.*',
            r'format:.*',
            r'example:.*',
            r'how\s+to.*',
            r'step\s+\d+.*',
            r'follow.*steps.*',
            r'click.*here.*',
            r'download.*',
            r'upload.*',
            r'select.*file.*',
            r'description:.*',
            r'help:.*',
            r'warning:.*',
            r'important:.*',
            r'tips?:.*',
            r'guide.*',
            r'tutorial.*',
            r'^\s*[-=*]+\s*$',  # Separator lines
            r'^\s*\d+\.\s*',     # Numbered lists
        ]
        
        # Patterns for unnecessary data
        self.unnecessary_patterns = [
            r'^\s*$',           # Empty rows
            r'^null$',          # Null values
            r'^n/?a$',          # N/A values
            r'^none$',          # None values
            r'^undefined$',     # Undefined values
            r'^#.*',           # Comments
            r'^//.*',          # Comments
        ]

    def comprehensive_clean(self, df: pd.DataFrame, data_type: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive data cleaning pipeline
        
        Args:
            df: Input dataframe
            data_type: Type of data (sessions, kpi_data, advancetags)
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_statistics)
        """
        start_time = datetime.now()
        self.cleaning_stats['original_shape'] = df.shape
        
        logger.info(f"Starting comprehensive cleaning for {df.shape[0]} rows, {df.shape[1]} columns")
        
        # 1. Remove instruction headers and unnecessary rows
        df_clean = self._remove_instruction_rows(df)
        
        # 2. Remove unnecessary data and comments
        df_clean = self._remove_unnecessary_data(df_clean)
        
        # 3. Remove blank columns based on threshold
        df_clean = self._remove_blank_columns(df_clean)
        
        # 4. Remove blank rows based on threshold
        df_clean = self._remove_blank_rows(df_clean)
        
        # 5. Clean and standardize column names
        df_clean = self._clean_column_names(df_clean)
        
        # 6. Handle whitespace and empty values
        df_clean = self._clean_whitespace(df_clean)
        
        # 7. Remove duplicate rows
        if self.config.remove_duplicate_rows:
            df_clean = self._remove_duplicates(df_clean)
        
        # 8. Handle mixed data types intelligently
        if self.config.handle_mixed_data_types:
            df_clean = self._handle_mixed_types(df_clean, data_type)
        
        # 9. Validate and clean specific data types
        if data_type:
            df_clean = self._clean_by_data_type(df_clean, data_type)
        
        # 10. Final validation and optimization
        df_clean = self._optimize_dataframe(df_clean)
        
        # Update statistics
        self.cleaning_stats['final_shape'] = df_clean.shape
        self.cleaning_stats['removed_columns'] = df.shape[1] - df_clean.shape[1]
        self.cleaning_stats['removed_rows'] = df.shape[0] - df_clean.shape[0]
        self.cleaning_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Cleaning complete: {df.shape} -> {df_clean.shape}")
        logger.info(f"Removed {self.cleaning_stats['removed_columns']} columns, {self.cleaning_stats['removed_rows']} rows")
        
        return df_clean, self.cleaning_stats

    def _remove_instruction_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove instruction headers and guide text"""
        if not self.config.remove_instruction_rows:
            return df
            
        rows_to_remove = []
        instruction_count = 0
        
        for idx, row in df.iterrows():
            # Convert row to string for pattern matching
            row_text = ' '.join([str(val).lower().strip() for val in row.values if pd.notna(val) and str(val).strip()])
            
            # Skip empty rows in this step
            if not row_text:
                continue
            
            # Check for instruction patterns
            for pattern in self.instruction_patterns:
                if re.search(pattern, row_text, re.IGNORECASE):
                    rows_to_remove.append(idx)
                    instruction_count += 1
                    break
            
            # Check if row looks like header/instruction based on structure
            non_null_count = row.count()
            if non_null_count == 1:  # Single cell rows often instructions
                single_value = str([val for val in row.values if pd.notna(val)][0]).strip()
                if len(single_value) > 50 and any(word in single_value.lower() for word in 
                    ['instruction', 'note', 'please', 'step', 'click', 'select', 'download', 'upload']):
                    rows_to_remove.append(idx)
                    instruction_count += 1
        
        if rows_to_remove:
            df_cleaned = df.drop(index=rows_to_remove).reset_index(drop=True)
            self.cleaning_stats['cleaned_instructions'] = instruction_count
            logger.info(f"Removed {len(rows_to_remove)} instruction/header rows")
            return df_cleaned
        
        return df

    def _remove_unnecessary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unnecessary data like comments and separators"""
        rows_to_remove = []
        
        for idx, row in df.iterrows():
            row_values = [str(val).strip() for val in row.values if pd.notna(val)]
            
            # Check if entire row matches unnecessary patterns
            if len(row_values) == 1:
                value = row_values[0].lower()
                for pattern in self.unnecessary_patterns:
                    if re.match(pattern, value, re.IGNORECASE):
                        rows_to_remove.append(idx)
                        break
        
        if rows_to_remove:
            df_cleaned = df.drop(index=rows_to_remove).reset_index(drop=True)
            logger.info(f"Removed {len(rows_to_remove)} unnecessary data rows")
            return df_cleaned
        
        return df

    def _remove_blank_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that are mostly blank based on threshold"""
        columns_to_remove = []
        
        for col in df.columns:
            # Calculate blank percentage
            total_cells = len(df[col])
            if total_cells == 0:
                columns_to_remove.append(col)
                continue
                
            # Count blanks (NaN, empty strings, whitespace)
            blank_count = 0
            for val in df[col]:
                if pd.isna(val) or str(val).strip() == '' or str(val).lower() in ['null', 'none', 'n/a']:
                    blank_count += 1
            
            blank_percentage = blank_count / total_cells
            
            if blank_percentage >= self.config.remove_blank_columns_threshold:
                columns_to_remove.append(col)
        
        if columns_to_remove:
            df_cleaned = df.drop(columns=columns_to_remove)
            logger.info(f"Removed {len(columns_to_remove)} mostly blank columns: {columns_to_remove[:5]}")
            return df_cleaned
        
        return df

    def _remove_blank_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that are mostly blank based on threshold"""
        if df.empty:
            return df
            
        rows_to_remove = []
        
        for idx, row in df.iterrows():
            # Calculate blank percentage for this row
            total_cells = len(row)
            if total_cells == 0:
                rows_to_remove.append(idx)
                continue
            
            # Count blanks
            blank_count = 0
            for val in row.values:
                if pd.isna(val) or str(val).strip() == '' or str(val).lower() in ['null', 'none', 'n/a']:
                    blank_count += 1
            
            blank_percentage = blank_count / total_cells
            
            if blank_percentage >= self.config.remove_blank_rows_threshold:
                rows_to_remove.append(idx)
        
        if rows_to_remove:
            df_cleaned = df.drop(index=rows_to_remove).reset_index(drop=True)
            logger.info(f"Removed {len(rows_to_remove)} mostly blank rows")
            return df_cleaned
        
        return df

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        if not self.config.standardize_column_names:
            return df
            
        original_columns = df.columns.tolist()
        new_columns = []
        
        for col in df.columns:
            # Convert to string and clean
            clean_col = str(col).strip()
            
            # Remove special characters but preserve important ones
            clean_col = re.sub(r'[^\w\s\(\)\%\.\-]', ' ', clean_col)
            
            # Normalize whitespace
            clean_col = re.sub(r'\s+', ' ', clean_col).strip()
            
            # Handle empty column names
            if not clean_col or clean_col.lower() in ['unnamed', 'null', 'none']:
                clean_col = f'Column_{len(new_columns) + 1}'
            
            new_columns.append(clean_col)
        
        # Handle duplicate column names
        seen = {}
        final_columns = []
        for col in new_columns:
            if col in seen:
                seen[col] += 1
                final_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_columns.append(col)
        
        df.columns = final_columns
        
        # Log significant changes
        changes = [(orig, new) for orig, new in zip(original_columns, final_columns) if str(orig) != new]
        if changes:
            logger.info(f"Cleaned {len(changes)} column names")
        
        return df

    def _clean_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean whitespace and standardize empty values"""
        if not self.config.clean_whitespace:
            return df
            
        # Clean string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Strip whitespace and standardize empty values
                df[col] = df[col].astype(str).str.strip()
                
                # Replace various empty representations with NaN
                empty_values = ['', 'null', 'None', 'none', 'N/A', 'n/a', 'NA', 'NULL', '#N/A', '#NULL!', 'undefined']
                df[col] = df[col].replace(empty_values, np.nan)
                
                # Convert back to object type to handle NaN properly
                df[col] = df[col].where(df[col] != 'nan', np.nan)
        
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows intelligently"""
        if df.empty:
            return df
            
        initial_count = len(df)
        
        # Remove exact duplicates
        df_dedup = df.drop_duplicates()
        
        # For session data, also check for duplicates based on session_id
        session_id_columns = ['Session ID', 'session_id', 'Session Id', 'sessionid']
        session_id_col = None
        
        for col in session_id_columns:
            if col in df_dedup.columns:
                session_id_col = col
                break
        
        if session_id_col:
            # Keep the first occurrence of each session_id
            df_dedup = df_dedup.drop_duplicates(subset=[session_id_col], keep='first')
        
        removed_count = initial_count - len(df_dedup)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        
        return df_dedup.reset_index(drop=True)

    def _handle_mixed_types(self, df: pd.DataFrame, data_type: str = None) -> pd.DataFrame:
        """Handle mixed data types intelligently"""
        for col in df.columns:
            # Skip if already proper type
            if df[col].dtype in ['int64', 'float64', 'datetime64[ns]', 'bool']:
                continue
            
            # Try numeric conversion
            if self._should_be_numeric(df[col]):
                df[col] = self._convert_to_numeric(df[col])
            
            # Try datetime conversion
            elif self._should_be_datetime(df[col]):
                df[col] = self._convert_to_datetime(df[col])
            
            # Try boolean conversion
            elif self._should_be_boolean(df[col]):
                df[col] = self._convert_to_boolean(df[col])
        
        return df

    def _should_be_numeric(self, series: pd.Series) -> bool:
        """Check if series should be converted to numeric"""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        numeric_count = 0
        for val in sample:
            try:
                # Try to convert, handling common numeric formats
                val_str = str(val).replace(',', '').replace('$', '').replace('%', '').strip()
                float(val_str)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        return numeric_count / len(sample) > 0.8

    def _should_be_datetime(self, series: pd.Series) -> bool:
        """Check if series should be converted to datetime"""
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        
        datetime_count = 0
        for val in sample:
            val_str = str(val).strip()
            # Check for common datetime patterns
            if (re.search(r'\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}', val_str) or 
                re.search(r'\\d{1,2}[-/]\\d{1,2}[-/]\\d{4}', val_str) or
                'T' in val_str and ':' in val_str):
                datetime_count += 1
        
        return datetime_count / len(sample) > 0.5

    def _should_be_boolean(self, series: pd.Series) -> bool:
        """Check if series should be converted to boolean"""
        unique_vals = series.dropna().astype(str).str.lower().unique()
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        
        return len(set(unique_vals) - boolean_values) == 0 and len(unique_vals) <= 4

    def _convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert series to numeric with intelligent handling"""
        def clean_numeric(val):
            if pd.isna(val):
                return val
            val_str = str(val).replace(',', '').replace('$', '').replace('%', '').strip()
            try:
                return float(val_str)
            except:
                return np.nan
        
        return series.apply(clean_numeric)

    def _convert_to_datetime(self, series: pd.Series) -> pd.Series:
        """Convert series to datetime with error handling"""
        return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)

    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert series to boolean with intelligent mapping"""
        def map_boolean(val):
            if pd.isna(val):
                return val
            val_str = str(val).lower().strip()
            if val_str in ['true', '1', 'yes', 'y']:
                return True
            elif val_str in ['false', '0', 'no', 'n']:
                return False
            return np.nan
        
        return series.apply(map_boolean)

    def _clean_by_data_type(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Apply data type specific cleaning"""
        if data_type == 'sessions':
            # Clean session-specific fields
            failure_columns = ['Video Start Failure', 'video_start_failure', 'Exit Before Video Starts', 'exit_before_video_starts']
            for col in failure_columns:
                if col in df.columns:
                    df[col] = self._convert_to_boolean(df[col])
        
        elif data_type == 'kpi_data':
            # Clean KPI numeric fields
            numeric_fields = ['Plays', 'plays', 'Video Start Failures Technical', 'video_start_failures_technical']
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
        
        elif data_type == 'advancetags':
            # Clean metadata fields
            ip_fields = ['IP', 'ip', 'IPv6', 'ipv6']
            for field in ip_fields:
                if field in df.columns:
                    df[field] = df[field].apply(self._validate_ip_address)
        
        return df

    def _validate_ip_address(self, ip_str):
        """Validate and clean IP addresses"""
        if pd.isna(ip_str):
            return ip_str
        try:
            ipaddress.ip_address(str(ip_str).strip())
            return str(ip_str).strip()
        except ValueError:
            return np.nan

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Optimize object columns
            if col_type == 'object':
                unique_count = df[col].nunique()
                total_count = len(df[col])
                # Convert to category if low cardinality
                if unique_count / total_count < 0.5 and unique_count < 100:
                    df[col] = df[col].astype('category')
            
            # Downcast numeric columns
            elif col_type in ['int64', 'int32']:
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min >= 0:  # Unsigned integers
                    if c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                else:  # Signed integers
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
            
            # Downcast floats
            elif col_type == 'float64':
                c_min = df[col].min()
                c_max = df[col].max()
                if not pd.isna(c_min) and not pd.isna(c_max):
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        if initial_memory > 0:
            reduction = (initial_memory - final_memory) / initial_memory * 100
            logger.info(f"Memory optimized: {initial_memory:.1f}MB -> {final_memory:.1f}MB ({reduction:.1f}% reduction)")
        
        return df

# ============================================================================
# DATA QUALITY MONITOR WITH ENHANCED METRICS
# ============================================================================

class DataQualityMonitor:
    """Monitor and report data quality metrics"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def comprehensive_quality_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Comprehensive data quality analysis with enhanced metrics"""
        start_time = datetime.now()
        
        metrics = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'completeness': self._calculate_completeness(df),
            'uniqueness': self._calculate_uniqueness(df), 
            'validity': self._calculate_validity(df),
            'consistency': self._calculate_consistency(df),
            'anomalies': self._detect_anomalies(df),
            'data_types': self._analyze_data_types(df),
            'column_stats': self._calculate_column_statistics(df),
            'analysis_time': (datetime.now() - start_time).total_seconds()
        }
        
        # Calculate overall quality score
        metrics['quality_score'] = self._calculate_quality_score(metrics)
        
        # Store metrics
        self.quality_metrics[dataset_name] = metrics
        
        logger.info(f"Quality analysis complete for {dataset_name}: Score = {metrics['quality_score']:.1f}/100")
        
        return metrics

    def _calculate_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate completeness score for each column"""
        completeness = {}
        
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            completeness[col] = (non_null_count / total_count) * 100 if total_count > 0 else 0
        
        completeness['overall'] = np.mean(list(completeness.values())) if completeness else 0
        return completeness

    def _calculate_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate uniqueness metrics"""
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
        """Calculate validity metrics based on data types and patterns"""
        validity = {
            'invalid_emails': 0,
            'invalid_dates': 0,  
            'invalid_numbers': 0,
            'invalid_ips': 0,
            'empty_strings': 0,
            'format_violations': {}
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Email validation
            if 'email' in col_lower:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
                invalid_emails = df[col].notna() & ~df[col].astype(str).str.match(email_pattern, na=False)
                validity['invalid_emails'] += invalid_emails.sum()
            
            # Date validation
            datetime_fields = ['time', 'date', 'timestamp']
            if any(field in col_lower for field in datetime_fields):
                try:
                    pd.to_datetime(df[col], format='mixed', errors='raise')
                except:
                    invalid_count = df[col].notna().sum()
                    validity['invalid_dates'] += invalid_count
            
            # IP address validation
            if 'ip' in col_lower:
                invalid_ips = 0
                for val in df[col].dropna():
                    try:
                        ipaddress.ip_address(str(val))
                    except ValueError:
                        invalid_ips += 1
                validity['invalid_ips'] += invalid_ips
            
            # Empty string count
            if df[col].dtype == 'object':
                empty_strings = df[col].astype(str).str.strip().eq('').sum()
                validity['empty_strings'] += empty_strings
        
        return validity

    def _calculate_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate consistency metrics"""
        consistency = {
            'format_consistency': {},
            'value_consistency': {},
            'type_consistency': {}
        }
        
        for col in df.columns:
            if df[col].dtype == 'object':
                non_null_values = df[col].dropna().astype(str)
                if len(non_null_values) > 0:
                    # Format consistency (pattern analysis)
                    patterns = non_null_values.apply(lambda x: re.sub(r'\\w', 'X', re.sub(r'\\d', '9', str(x))))
                    pattern_variety = patterns.nunique() / len(patterns) if len(patterns) > 0 else 0
                    
                    consistency['format_consistency'][col] = {
                        'pattern_variety': pattern_variety,
                        'consistent': pattern_variety < 0.1,
                        'most_common_pattern': patterns.value_counts().index[0] if len(patterns) > 0 else None
                    }
        
        return consistency

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect various types of data anomalies"""
        anomalies = {
            'outliers': {},
            'suspicious_patterns': {},
            'statistical_anomalies': {}
        }
        
        # Outlier detection for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].notna().sum() > 10:  # Need at least 10 values
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                    anomalies['outliers'][col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(df)) * 100,
                        'values': outliers[col].tolist()[:10]  # Sample outliers
                    }
        
        # Suspicious pattern detection for text columns
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                value_counts = non_null_values.value_counts()
                if len(value_counts) > 0:
                    most_common_value = value_counts.index[0]
                    most_common_count = value_counts.iloc[0]
                    dominance = most_common_count / len(non_null_values)
                    
                    # Flag if a single value dominates more than 80%
                    if dominance > 0.8:
                        anomalies['suspicious_patterns'][col] = {
                            'dominant_value': most_common_value,
                            'frequency': most_common_count,
                            'percentage': dominance * 100
                        }
        
        return anomalies

    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze data types and their suitability"""
        type_analysis = {}
        
        for col in df.columns:
            type_analysis[col] = {
                'current_type': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_values': df[col].nunique(),
                'memory_usage': df[col].memory_usage(deep=True)
            }
            
            # Suggest better type if applicable
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    # Check if could be numeric
                    try:
                        pd.to_numeric(sample)
                        type_analysis[col]['suggested_type'] = 'numeric'
                    except:
                        # Check if could be datetime
                        try:
                            pd.to_datetime(sample)
                            type_analysis[col]['suggested_type'] = 'datetime'
                        except:
                            # Check if could be category
                            if sample.nunique() / len(sample) < 0.5:
                                type_analysis[col]['suggested_type'] = 'category'
        
        return type_analysis

    def _calculate_column_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate detailed statistics for each column"""
        column_stats = {}
        
        for col in df.columns:
            stats = {
                'count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Numeric statistics
                stats.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q50': df[col].quantile(0.50),
                    'q75': df[col].quantile(0.75)
                })
            
            elif df[col].dtype == 'object':
                # Text statistics
                non_null = df[col].dropna().astype(str)
                if len(non_null) > 0:
                    stats.update({
                        'avg_length': non_null.str.len().mean(),
                        'max_length': non_null.str.len().max(),
                        'min_length': non_null.str.len().min(),
                        'most_common': non_null.value_counts().head(3).to_dict()
                    })
            
            column_stats[col] = stats
        
        return column_stats

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        scores = []
        weights = []
        
        # Completeness score (30% weight)
        completeness_score = metrics['completeness'].get('overall', 0)
        scores.append(completeness_score)
        weights.append(0.30)
        
        # Uniqueness score (20% weight)
        duplicate_penalty = min(metrics['uniqueness']['duplicate_percentage'], 20)
        uniqueness_score = max(0, 100 - duplicate_penalty)
        scores.append(uniqueness_score)
        weights.append(0.20)
        
        # Validity score (25% weight) 
        validity_issues = sum([
            metrics['validity']['invalid_emails'],
            metrics['validity']['invalid_dates'], 
            metrics['validity']['invalid_numbers'],
            metrics['validity']['invalid_ips']
        ])
        total_cells = metrics['total_rows'] * metrics['total_columns']
        validity_score = max(0, 100 - (validity_issues / total_cells * 100)) if total_cells > 0 else 100
        scores.append(validity_score)
        weights.append(0.25)
        
        # Anomaly score (25% weight)
        total_outliers = sum(
            anomaly.get('count', 0)
            for anomaly in metrics['anomalies'].get('outliers', {}).values()
        )
        anomaly_penalty = min((total_outliers / metrics['total_rows']) * 100, 25) if metrics['total_rows'] > 0 else 0
        anomaly_score = max(0, 100 - anomaly_penalty)
        scores.append(anomaly_score)
        weights.append(0.25)
        
        # Calculate weighted average
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        return round(weighted_score, 2)

# ============================================================================
# COMPREHENSIVE DATA VALIDATOR
# ============================================================================

class ComprehensiveDataValidator:
    """Main validator class that orchestrates cleaning and validation"""
    
    def __init__(self, cleaning_config: CleaningConfig = None):
        self.cleaner = EnhancedDataCleaner(cleaning_config)
        self.quality_monitor = DataQualityMonitor()
        self.validation_history = []
    
    def validate_and_clean(self, df: pd.DataFrame, data_type: str = None, 
                          dataset_name: str = None) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Main validation and cleaning method
        
        Args:
            df: Input dataframe
            data_type: Type of data (sessions, kpi_data, advancetags)
            dataset_name: Name for logging and tracking
            
        Returns:
            Tuple of (cleaned_dataframe, validation_result)
        """
        start_time = datetime.now()
        dataset_name = dataset_name or f"dataset_{len(self.validation_history) + 1}"
        
        logger.info(f"Starting validation and cleaning for {dataset_name}")
        
        # 1. Comprehensive cleaning
        df_clean, cleaning_stats = self.cleaner.comprehensive_clean(df, data_type)
        
        # 2. Quality analysis
        quality_metrics = self.quality_monitor.comprehensive_quality_analysis(df_clean, dataset_name)
        
        # 3. Validation checks
        validation_errors = []
        validation_warnings = []
        
        # Check if dataframe is empty after cleaning
        if df_clean.empty:
            validation_errors.append("Dataframe is empty after cleaning")
        
        # Check if essential columns exist for data type
        if data_type:
            missing_essential = self._check_essential_columns(df_clean, data_type)
            if missing_essential:
                validation_warnings.extend([f"Missing recommended column: {col}" for col in missing_essential])
        
        # Check data quality thresholds
        if quality_metrics['quality_score'] < 50:
            validation_warnings.append(f"Data quality score is low: {quality_metrics['quality_score']:.1f}/100")
        
        # Create validation result
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = ValidationResult(
            is_valid=len(validation_errors) == 0,
            errors=validation_errors,
            warnings=validation_warnings,
            processed_rows=len(df_clean),
            failed_rows=len(df) - len(df_clean),
            quality_score=quality_metrics['quality_score'],
            processing_time=processing_time,
            cleaned_columns=cleaning_stats['removed_columns'],
            removed_rows=cleaning_stats['removed_rows']
        )
        
        # Store validation history
        self.validation_history.append({
            'dataset_name': dataset_name,
            'timestamp': datetime.now(),
            'result': result,
            'quality_metrics': quality_metrics,
            'cleaning_stats': cleaning_stats
        })
        
        logger.info(f"Validation complete for {dataset_name}: {'PASSED' if result.is_valid else 'FAILED'}")
        logger.info(f"Quality score: {result.quality_score:.1f}/100, Processing time: {result.processing_time:.2f}s")
        
        return df_clean, result
    
    def _check_essential_columns(self, df: pd.DataFrame, data_type: str) -> List[str]:
        """Check for essential columns based on data type"""
        essential_columns = {
            'sessions': ['session_id', 'Session ID'],
            'kpi_data': ['timestamp', 'Timestamp', 'plays', 'Plays'], 
            'advancetags': ['session_id', 'Session ID', 'Session Id']
        }
        
        if data_type not in essential_columns:
            return []
        
        missing = []
        required = essential_columns[data_type]
        
        # Check if at least one variant exists
        for req in required[::2]:  # Check every other (base names)
            variants = [req, req.replace('_', ' ').title(), req.title()]
            if not any(variant in df.columns for variant in variants):
                missing.append(req)
        
        return missing
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation operations"""
        if not self.validation_history:
            return {'message': 'No validations performed yet'}
        
        total_validations = len(self.validation_history)
        successful_validations = sum(1 for h in self.validation_history if h['result'].is_valid)
        
        avg_quality_score = np.mean([h['quality_metrics']['quality_score'] for h in self.validation_history])
        avg_processing_time = np.mean([h['result'].processing_time for h in self.validation_history])
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': (successful_validations / total_validations) * 100,
            'average_quality_score': round(avg_quality_score, 2),
            'average_processing_time': round(avg_processing_time, 2),
            'latest_validation': self.validation_history[-1]['timestamp'],
            'total_rows_processed': sum(h['result'].processed_rows for h in self.validation_history),
            'total_rows_cleaned': sum(h['result'].removed_rows for h in self.validation_history),
        }

# ============================================================================
# CONVENIENCE FUNCTIONS - PREVENT DUPLICACY
# ============================================================================

def create_data_validator(cleaning_config: CleaningConfig = None) -> ComprehensiveDataValidator:
    """Factory function to create data validator"""
    return ComprehensiveDataValidator(cleaning_config)

def quick_clean_dataframe(df: pd.DataFrame, data_type: str = None) -> pd.DataFrame:
    """Quick cleaning function for simple use cases"""
    cleaner = EnhancedDataCleaner()
    cleaned_df, _ = cleaner.comprehensive_clean(df, data_type)
    return cleaned_df

def analyze_data_quality(df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
    """Standalone data quality analysis"""
    monitor = DataQualityMonitor()
    return monitor.comprehensive_quality_analysis(df, dataset_name)

def validate_session_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
    """Specialized validation for session data"""
    validator = create_data_validator()
    return validator.validate_and_clean(df, 'sessions', 'session_data')

def validate_kpi_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
    """Specialized validation for KPI data"""
    validator = create_data_validator()
    return validator.validate_and_clean(df, 'kpi_data', 'kpi_data')

def validate_metadata(df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
    """Specialized validation for metadata/advancetags"""
    validator = create_data_validator()
    return validator.validate_and_clean(df, 'advancetags', 'metadata')
