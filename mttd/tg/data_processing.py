# data_processing.py - CLEAN VERSION WITH ORIGINAL NAMES

"""
Unified Data Processing Pipeline
===============================

Features:
- Manual Upload Data: Full preprocessing + ticket generation (no MongoDB save)
- MongoDB Data: Light processing (column mapping only) + ticket generation
- Session-focused processing with session_id as unique identifier
- Flexible column mapping maintained
- No batch_id concept
- Original function names preserved
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime, timedelta
import re
import json
import tempfile
import os
import gc
import psutil

# Django imports
from django.db import transaction
from django.utils import timezone
from django.core import serializers

from .mongo_service import test_mongodb_connection
from .operation.ticket_engine import AutoTicketMVP
from .mongo_service import test_mongodb_connection, fetch_collections, save_tickets
# Import existing functions to prevent duplicacy
try:
    from .models import Session, KPI, Advancetags, Ticket
    from .data_validation import ComprehensiveDataValidator, create_data_validator
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import some dependencies")

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for data processing operations"""
    chunk_size: int = 10000
    memory_threshold_mb: int = 500
    max_columns: int = 200
    enable_parallel: bool = True
    skip_validation: bool = False
    auto_detect_types: bool = True
    cleanup_temp_files: bool = True
    session_only_tickets: bool = True
    variable_channels: bool = True
    remove_blank_columns_threshold: float = 0.95
    remove_blank_rows_threshold: float = 0.90
    flexible_column_mapping: bool = True

# ============================================================================
# FLEXIBLE COLUMN MAPPER
# ============================================================================

class FlexibleColumnMapper:
    """Maps various column names to standardized model fields"""
    
    def __init__(self):
        # Session mappings
        self.session_mappings = {
            'session_id': {
                'exact': ['session_id', 'Session ID', 'sessionid', 'SessionID'],
                'contains': ['session', 'session_id'],
                'patterns': [r'session.*id', r'id.*session']
            },
            'viewer_id': {
                'exact': ['viewer_id', 'Viewer ID', 'viewerid', 'ViewerID'],
                'contains': ['viewer', 'viewer_id'],
                'patterns': [r'viewer.*id', r'id.*viewer']
            },
            'asset_name': {
                'exact': ['asset_name', 'Asset Name', 'assetname', 'channel', 'Channel'],
                'contains': ['asset', 'channel', 'content'],
                'patterns': [r'asset.*name', r'channel.*name', r'content.*name']
            },
            'session_start_time': {
                'exact': ['session_start_time', 'Session Start Time', 'start_time', 'timestamp'],
                'contains': ['start_time', 'session_start', 'timestamp'],
                'patterns': [r'session.*start.*time', r'start.*time', r'time.*start']
            },
            'session_end_time': {
                'exact': ['session_end_time', 'Session End Time', 'end_time'],
                'contains': ['end_time', 'session_end'],
                'patterns': [r'session.*end.*time', r'end.*time', r'time.*end']
            },
            'status': {
                'exact': ['status', 'Status', 'session_status'],
                'contains': ['status', 'state'],
                'patterns': [r'.*status.*', r'.*state.*']
            },
            'playing_time': {
                'exact': ['playing_time', 'Playing Time'],
                'contains': ['playing', 'duration'],
                'patterns': [r'playing.*time', r'time.*playing']
            },
            'rebuffering_ratio': {
                'exact': ['rebuffering_ratio', 'Rebuffering Ratio'],
                'contains': ['rebuffering', 'buffering'],
                'patterns': [r'rebuffer.*ratio', r'buffer.*ratio']
            },
            'avg_peak_bitrate': {
                'exact': ['avg_peak_bitrate', 'Avg. Peak Bitrate', 'Average Peak Bitrate'],
                'contains': ['peak_bitrate', 'peak'],
                'patterns': [r'avg.*peak.*bitrate', r'average.*peak']
            }
        }
        
        # KPI mappings
        self.kpi_mappings = {
            'timestamp': {
                'exact': ['timestamp', 'Timestamp', 'time', 'Time'],
                'contains': ['time', 'date'],
                'patterns': [r'.*time.*', r'.*date.*']
            },
            'plays': {
                'exact': ['plays', 'Plays', 'play_count'],
                'contains': ['play', 'count'],
                'patterns': [r'play.*count', r'.*play.*']
            },
            'streaming_performance_index': {
                'exact': ['streaming_performance_index', 'Streaming Performance Index'],
                'contains': ['streaming', 'performance'],
                'patterns': [r'streaming.*performance', r'performance.*index']
            }
        }
        
        # Advancetags mappings
        self.advancetags_mappings = {
            'session_id': {
                'exact': ['session_id', 'Session ID', 'sessionid', 'Session Id'],
                'contains': ['session'],
                'patterns': [r'session.*id', r'id.*session']
            },
            'asset_name': {
                'exact': ['asset_name', 'Asset Name', 'channel'],
                'contains': ['asset', 'channel'],
                'patterns': [r'asset.*name', r'channel.*name']
            },
            'browser_name': {
                'exact': ['browser_name', 'Browser Name'],
                'contains': ['browser'],
                'patterns': [r'browser.*name', r'.*browser.*']
            },
            'device_name': {
                'exact': ['device_name', 'Device Name'],
                'contains': ['device', 'name'],
                'patterns': [r'device.*name', r'name.*device']
            },
            'city': {
                'exact': ['city', 'City'],
                'contains': ['city'],
                'patterns': [r'.*city.*']
            },
            'country': {
                'exact': ['country', 'Country'],
                'contains': ['country'],
                'patterns': [r'.*country.*']
            },
            'ip': {
                'exact': ['ip', 'IP', 'ip_address'],
                'contains': ['ip'],
                'patterns': [r'.*ip.*', r'ip.*address']
            }
        }
    
    def flexible_map_columns(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Apply flexible column mapping based on data type"""
        if df is None or df.empty:
            logger.warning(f"Empty dataframe provided for {data_type} mapping")
            return df

        logger.info(f"=== MAPPING {data_type.upper()} ===")
        logger.info(f"Input columns: {list(df.columns)}")

        if data_type == 'sessions':
            mapping_dict = self.session_mappings
        elif data_type == 'kpi_data':
            mapping_dict = self.kpi_mappings
        elif data_type == 'advancetags':
            mapping_dict = self.advancetags_mappings
        else:
            logger.warning(f"Unknown data type: {data_type}")
            return df

        # Create column mapping
        column_map = {}
        available_columns = df.columns.tolist()

        for standard_name, match_config in mapping_dict.items():
            mapped_column = self._find_column_flexible(available_columns, match_config)
            if mapped_column:
                column_map[mapped_column] = standard_name
                logger.info(f"Mapped: '{mapped_column}' -> '{standard_name}'")

        # Apply mapping
        if column_map:
            df_mapped = df.rename(columns=column_map)
            logger.info(f"Successfully mapped {len(column_map)} columns for {data_type}")
            return df_mapped
        else:
            logger.warning(f"No columns mapped for {data_type}")
            # RETURN ORIGINAL DF INSTEAD OF FAILING
            return df

    def _find_column_flexible(self, available_columns: List[str], match_config: Dict[str, List]) -> Optional[str]:
        """Find matching column using flexible patterns"""
        # Try exact matches first
        for exact_name in match_config.get('exact', []):
            for col in available_columns:
                if str(col).lower() == str(exact_name).lower():
                    return col

        # Try contains matches
        for contains_text in match_config.get('contains', []):
            for col in available_columns:
                if contains_text.lower() in str(col).lower():
                    return col

        # Try pattern matches
        for pattern in match_config.get('patterns', []):
            for col in available_columns:
                if re.search(pattern, str(col).lower(), re.IGNORECASE):
                    return col

        return None

# ============================================================================
# SMART DATA CLEANER
# ============================================================================

class SmartDataCleaner:
    """Enhanced data cleaner for manual upload data"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
    
    def smart_clean_dataframe(self, df: pd.DataFrame, data_type: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Full data cleaning for manual upload data"""
        start_time = datetime.now()
        original_shape = df.shape
        
        logger.info(f"Smart cleaning started: {original_shape[0]} rows, {original_shape[1]} columns")
        
        # 1. Find actual data start position
        df_clean = self._find_data_start_position(df)
        
        # 2. Remove instruction rows and noise
        df_clean = self._remove_instructions_and_noise(df_clean)
        
        # 3. Remove blank columns intelligently
        df_clean = self._remove_blank_columns_smart(df_clean)
        
        # 4. Remove blank rows intelligently
        df_clean = self._remove_blank_rows_smart(df_clean)
        
        # 5. Clean column names
        df_clean = self._clean_column_names(df_clean)
        
        # 6. Standardize data types
        df_clean = self._standardize_data_types(df_clean, data_type)
        
        # 7. Handle mixed data in cells
        df_clean = self._handle_mixed_cell_data(df_clean)
        
        # 8. Final optimization
        df_clean = self._optimize_memory(df_clean)
        
        # Calculate statistics
        final_shape = df_clean.shape
        processing_time = (datetime.now() - start_time).total_seconds()
        
        cleaning_stats = {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'removed_rows': original_shape[0] - final_shape[0],
            'removed_columns': original_shape[1] - final_shape[1],
            'processing_time': processing_time,
        }
        
        logger.info(f"Smart cleaning complete: {original_shape} -> {final_shape}")
        return df_clean, cleaning_stats
    
    def _find_data_start_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find where actual data starts"""
        if df.empty:
            return df
        
        potential_headers = []
        
        for idx in range(min(20, len(df))):
            row = df.iloc[idx]
            non_null_count = row.count()
            
            if non_null_count >= 3:
                row_text = ' '.join([str(val) for val in row.values if pd.notna(val)])
                header_indicators = ['session', 'timestamp', 'plays', 'browser', 'device', 'ip', 'cdn', 'asset']
                matches = sum(1 for indicator in header_indicators 
                            if indicator.lower() in row_text.lower())
                
                if matches >= 2:
                    potential_headers.append((idx, matches, non_null_count))
        
        if potential_headers:
            best_header_idx = max(potential_headers, key=lambda x: (x[1], x[2]))[0]
            
            if best_header_idx > 0:
                logger.info(f"Data starts at row {best_header_idx}")
                new_columns = []
                header_row = df.iloc[best_header_idx]
                
                for val in header_row.values:
                    if pd.notna(val) and str(val).strip():
                        new_columns.append(str(val).strip())
                    else:
                        new_columns.append(f'Column_{len(new_columns) + 1}')
                
                df_data = df.iloc[best_header_idx + 1:].copy()
                df_data.columns = new_columns[:len(df_data.columns)]
                df_data = df_data.reset_index(drop=True)
                return df_data
        
        return df
    
    def _remove_instructions_and_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove instruction rows and noise data"""
        if df.empty:
            return df
        
        instruction_patterns = [
            r'instructions?:', r'note:', r'please', r'click', r'select',
            r'download', r'upload', r'step\s+\d+', r'guide', r'help'
        ]
        
        rows_to_remove = []
        
        for idx, row in df.iterrows():
            row_text = ' '.join([str(val).lower() for val in row.values if pd.notna(val)])
            
            if len(row_text.strip()) < 5:
                rows_to_remove.append(idx)
                continue
            
            for pattern in instruction_patterns:
                if re.search(pattern, row_text, re.IGNORECASE):
                    rows_to_remove.append(idx)
                    break
        
        if rows_to_remove:
            df_clean = df.drop(index=rows_to_remove).reset_index(drop=True)
            logger.info(f"Removed {len(rows_to_remove)} instruction/noise rows")
            return df_clean
        
        return df
    
    def _remove_blank_columns_smart(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove blank columns based on threshold"""
        if df.empty:
            return df
        
        columns_to_remove = []
        threshold = self.config.remove_blank_columns_threshold
        
        for col in df.columns:
            total_rows = len(df)
            blank_count = sum(1 for val in df[col] if pd.isna(val) or str(val).strip() == '')
            blank_ratio = blank_count / total_rows if total_rows > 0 else 1.0
            
            if blank_ratio >= threshold:
                columns_to_remove.append(col)
        
        if columns_to_remove:
            df_clean = df.drop(columns=columns_to_remove)
            logger.info(f"Removed {len(columns_to_remove)} blank columns")
            return df_clean
        
        return df
    
    def _remove_blank_rows_smart(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove blank rows based on threshold"""
        if df.empty:
            return df
        
        rows_to_remove = []
        threshold = self.config.remove_blank_rows_threshold
        
        for idx, row in df.iterrows():
            total_cols = len(row)
            blank_count = sum(1 for val in row.values if pd.isna(val) or str(val).strip() == '')
            blank_ratio = blank_count / total_cols if total_cols > 0 else 1.0
            
            if blank_ratio >= threshold:
                rows_to_remove.append(idx)
        
        if rows_to_remove:
            df_clean = df.drop(index=rows_to_remove).reset_index(drop=True)
            logger.info(f"Removed {len(rows_to_remove)} blank rows")
            return df_clean
        
        return df
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        new_columns = []
        
        for col in df.columns:
            clean_name = str(col).strip()
            clean_name = re.sub(r'[^\w\s\(\)\%\.\-]', ' ', clean_name)
            clean_name = re.sub(r'\s+', ' ', clean_name).strip()
            
            if not clean_name or clean_name.lower() in ['unnamed', 'null', 'none']:
                clean_name = f'Column_{len(new_columns) + 1}'
            
            new_columns.append(clean_name)
        
        # Handle duplicates
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
        return df
    
    def _standardize_data_types(self, df: pd.DataFrame, data_type: str = None) -> pd.DataFrame:
        """Standardize data types"""
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'datetime64[ns]']:
                continue
            
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
            
            if self._looks_like_number(sample):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif self._looks_like_datetime(sample):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _looks_like_number(self, series: pd.Series) -> bool:
        """Check if series looks numeric"""
        numeric_count = 0
        for val in series:
            try:
                float(str(val).strip().replace(',', ''))
                numeric_count += 1
            except ValueError:
                pass
        return numeric_count / len(series) > 0.8
    
    def _looks_like_datetime(self, series: pd.Series) -> bool:
        """Check if series looks like datetime"""
        datetime_count = 0
        for val in series:
            val_str = str(val).strip()
            if (re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', val_str) or
                'T' in val_str and ':' in val_str):
                datetime_count += 1
        return datetime_count / len(series) > 0.5
    
    def _handle_mixed_cell_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle mixed cell data"""
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                empty_vals = ['nan', 'None', 'null', 'NULL', 'N/A', 'n/a', 'NA']
                df[col] = df[col].replace(empty_vals, np.nan)
        return df
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage"""
        for col in df.columns:
            if df[col].dtype == 'object':
                nunique = df[col].nunique()
                if nunique / len(df) < 0.5 and nunique < 100:
                    df[col] = df[col].astype('category')
        return df

# ============================================================================
# MAIN PROCESSOR CLASS
# ============================================================================

class FlexibleUnifiedDataProcessor:
    """
    Main processor handling both manual uploads and MongoDB data
    - Manual Upload: Full preprocessing + ticket generation (no save to MongoDB)
    - MongoDB Data: Light processing (column mapping only) + ticket generation
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.mapper = FlexibleColumnMapper()
        self.cleaner = SmartDataCleaner(config)
        self.validator = None
        
        try:
            self.validator = create_data_validator()
        except:
            logger.warning("Data validator not available")
        
        self.processing_stats = {
            'files_processed': 0,
            'total_records': 0,
            'errors': [],
            'warnings': [],
            'session_ids_processed': []
        }
    
    def process_manual_upload_flexible(self, files_mapping: Dict, target_channels: List[str] = None) -> Dict[str, Any]:
        """Process manual upload data with FULL preprocessing + ticket generation (NO MongoDB save)"""
        logger.info(f"Processing manual upload data with {len(files_mapping)} files")
        logger.info(f"Target channels: {target_channels or 'All channels'}")
        
        try:
            # Initialize data containers
            all_data = {
                'sessions': pd.DataFrame(),
                'kpi_data': pd.DataFrame(),
                'advancetags': pd.DataFrame()
            }
            
            # Process each file with FULL preprocessing
            for file_obj, data_types in files_mapping.items():
                try:
                    filename = getattr(file_obj, 'name', 'uploaded_file')
                    logger.info(f"Full preprocessing: {filename}")
                    
                    # Read and process file
                    file_data = self._process_single_file_flexible(file_obj, filename, data_types)
                    
                    # Apply column mapping and merge
                    for data_type, df in file_data.items():
                        if not df.empty and data_type in all_data:
                            # Apply flexible column mapping
                            df_mapped = self.mapper.flexible_map_columns(df, data_type)
                            
                            # Filter by channels if specified
                            if target_channels and data_type in ['sessions', 'advancetags']:
                                df_mapped = self._filter_by_channels(df_mapped, target_channels)
                            
                            # Merge with existing data
                            all_data[data_type] = pd.concat([all_data[data_type], df_mapped], ignore_index=True)
                    
                    self.processing_stats['files_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing manual file {filename}: {e}")
                    self.processing_stats['errors'].append(f"File {filename}: {str(e)}")
            
            # Extract session IDs
            session_ids = self._extract_session_ids(all_data)
            self.processing_stats['session_ids_processed'] = session_ids
            
            # Generate tickets directly (NO MongoDB save)
            tickets_generated = self._generate_session_tickets_flexible(all_data, target_channels)
            
            self.processing_stats['total_records'] = sum(len(df) for df in all_data.values())
            
            return {
                'success': True,
                'processing_type': 'manual_upload',
                'session_ids_processed': len(session_ids),
                'tickets_generated': tickets_generated,
                'target_channels': target_channels or [],
                'data_counts': {k: len(v) for k, v in all_data.items()},
                'stats': self.processing_stats,
                'flexible_mapping_used': True
            }
            
        except Exception as e:
            logger.error(f"Manual upload processing failed: {e}")
            return {
                'success': False,
                'processing_type': 'manual_upload',
                'errors': [str(e)]
            }

# process_mongodb_ingestion_flexible method:

    def process_mongodb_ingestion_flexible(self, target_channels: List[str] = None) -> Dict[str, Any]:
        """FIXED: Process MongoDB data using Django ORM approach"""
        logger.info("🚀 Processing MongoDB data with Django ORM")
        logger.info(f"🎯 Target channels: {target_channels or 'All channels'}")

        try:
            # Step 1: Fetch data using Django ORM
            mongodb_data = fetch_collections(target_channels)

            # Check if we got any data
            total_records = sum(len(df) for df in mongodb_data.values())
            if total_records == 0:
                logger.warning("⚠️ No data found in Django MongoDB models")
                return {
                    "success": False, 
                    "error": "No data found in MongoDB collections. Make sure data is loaded into Django models.",
                    "processing_type": "mongodb_django_orm",
                    "datacounts": {"sessions": 0, "kpi_data": 0, "advancetags": 0}
                }

            # Step 2: LIGHT PROCESSING - only column mapping (as designed)
            processed_data = {}
            for datatype, df in mongodb_data.items():
                if not df.empty:
                    logger.info(f"📋 Processing {datatype}: {len(df)} records")

                    # Apply flexible column mapping
                    df_mapped = self.mapper.flexible_map_columns(df, datatype)

                    # Filter by channels if specified
                    if target_channels and datatype in ['sessions', 'advancetags']:
                        df_mapped = self.filter_by_channels(df_mapped, target_channels)

                    processed_data[datatype] = df_mapped
                    logger.info(f"✅ Processed {datatype}: {len(df_mapped)} records")
                else:
                    processed_data[datatype] = df
                    
            session_ids = self._extract_session_ids(processed_data)
            
            logger.info(f"📋 Extracted {len(session_ids)} unique session IDs")
            # HANDLE EMPTY RESULT GRACEFULLY
            if not session_ids:
                logger.warning("No session IDs extracted - database appears to be empty")
        
            # Step 4: Generate tickets
            tickets_generated = self._generate_session_tickets_flexible(processed_data, target_channels)

            # Step 5: Return success result
            return {
                "success": True,
                "processing_type": "mongodb_django_orm",
                "session_ids_processed": len(session_ids),
                "tickets_generated": tickets_generated,
                "target_channels": target_channels or [],
                "datacounts": {k: len(v) for k, v in processed_data.items()},
                "flexible_processing_used": True,
                "django_orm_used": True,  # Flag to indicate Django ORM approach
                "backend": "django_mongodb_backend"
            }

        except Exception as e:
            logger.error(f"❌ Django MongoDB processing failed: {e}", exc_info=True)
            return {
                "success": False, 
                "processing_type": "mongodb_django_orm", 
                "error": str(e),
                "django_orm_attempted": True
            }
    
    def _process_single_file_flexible(self, file_obj, filename: str, data_types: List[str]) -> Dict[str, pd.DataFrame]:
        """Process single file with flexible positioning"""
        try:
            # 1. Read raw file
            df_raw = self._read_file_flexible(file_obj, filename)

            # 2. Initialize result container
            result: Dict[str, pd.DataFrame] = {}

            # 3. If auto_detect requested, clean once and auto‐detect types
            if 'auto_detect' in data_types:
                df_clean, cleaning_stats = self.cleaner.smart_clean_dataframe(df_raw, None)
                detected = self._auto_detect_data_types_flexible(df_clean)
                for dt, df in detected.items():
                    if dt in ['sessions', 'kpi_data', 'advancetags']:
                        result[dt] = df.copy()
                logger.info(f"Processed file {filename} (auto_detect): {cleaning_stats}")
                return result

            # 4. Manual types: clean separately for each requested type
            for dt in data_types:
                if dt in ['sessions', 'kpi_data', 'advancetags']:
                    df_clean, cleaning_stats = self.cleaner.smart_clean_dataframe(df_raw, dt)
                    result[dt] = df_clean.copy()
                    logger.info(f"Processed file {filename} for {dt}: {cleaning_stats}")

            return result

        except Exception as e:
            logger.error(f"Error in flexible file processing {filename}: {e}")
            return {}   
 
    def _read_file_flexible(self, file_obj, filename: str) -> pd.DataFrame:
        """Read file with flexible format support"""
        file_extension = Path(filename).suffix.lower()
        
        if file_extension in ['.xlsx', '.xls']:
            for engine in ['openpyxl', 'xlrd']:
                try:
                    if hasattr(file_obj, 'seek'):
                        file_obj.seek(0)
                    return pd.read_excel(file_obj, engine=engine)
                except Exception:
                    continue
            raise ValueError("Could not read Excel file")
            
        elif file_extension == '.csv':
            encodings = ['utf-8', 'latin1', 'cp1252']
            separators = [',', ';', '\t']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        if hasattr(file_obj, 'seek'):
                            file_obj.seek(0)
                        df = pd.read_csv(file_obj, encoding=encoding, sep=sep)
                        if not df.empty and len(df.columns) > 1:
                            return df
                    except Exception:
                        continue
            raise ValueError("Could not read CSV file")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _auto_detect_data_types_flexible(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Auto-detect data types with flexible column analysis"""
        result = {}
        columns = [str(col).lower() for col in df.columns]
        
        # Session-focused pattern matching
        session_score = sum(1 for col in columns if any(p in col for p in ['session', 'asset', 'status', 'time']))
        kpi_score = sum(1 for col in columns if any(p in col for p in ['plays', 'performance', 'ratio']))
        meta_score = sum(1 for col in columns if any(p in col for p in ['browser', 'device', 'ip', 'city']))
        
        scores = {'sessions': session_score, 'kpi_data': kpi_score, 'advancetags': meta_score}
        
        # Assign to highest scoring type, prioritizing sessions
        max_score = max(scores.values())
        if max_score > 0:
            if scores['sessions'] == max_score:
                best_type = 'sessions'
            else:
                best_type = max(scores, key=scores.get)
            
            result[best_type] = df.copy()
        else:
            result['sessions'] = df.copy()
        
        logger.info(f"Auto-detected session types: {list(result.keys())} (scores: {scores})")
        return result
    
    def _filter_by_channels(self, df: pd.DataFrame, target_channels: List[str]) -> pd.DataFrame:
        """Filter data by target channels with flexible column detection"""
        if not target_channels or df.empty:
            return df
        
        # Find channel column
        channel_col = None
        for col in df.columns:
            if any(term in str(col).lower() for term in ['asset', 'channel', 'content', 'name']):
                channel_col = col
                break
        
        if channel_col:
            initial_count = len(df)
            mask = df[channel_col].astype(str).str.lower().isin([ch.lower() for ch in target_channels])
            filtered_df = df[mask].copy()
            logger.info(f"Channel filtering: {initial_count} -> {len(filtered_df)} records")
            return filtered_df
        
        return df
    # Debug Session ID Values - Add this to your _extract_session_ids function

    def _extract_session_ids(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """Extract unique session IDs from sessions data - WITH DEBUGGING"""
        logger.info("=== Extracting Session IDs ===")

        if 'sessions' not in data:
            logger.warning("No 'sessions' key found in data")
            return []

        df = data['sessions']
        if df.empty:
            logger.warning("Sessions DataFrame is empty")
            return []

        logger.info(f"Processing sessions DataFrame: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Available columns: {list(df.columns)}")

        session_id_columns = ['session_id', 'Session ID', 'sessionid', 'Session Id']

        for col in session_id_columns:
            if col in df.columns:
                logger.info(f"Found session ID column: '{col}'")

                # DEBUG: Check raw values before any processing
                raw_values = df[col].tolist()
                logger.info(f"Raw values (first 10): {raw_values[:10]}")
                logger.info(f"Total raw values: {len(raw_values)}")
                logger.info(f"Non-null raw values: {df[col].notna().sum()}")

                # Check data types
                logger.info(f"Column dtype: {df[col].dtype}")
                logger.info(f"Unique raw values: {df[col].nunique()}")

                # Process values step by step
                non_null_values = df[col].dropna()
                logger.info(f"After dropna: {len(non_null_values)} values")
                logger.info(f"Sample non-null values: {non_null_values.head(10).tolist()}")

                string_values = non_null_values.astype(str)
                logger.info(f"After astype(str): {len(string_values)} values")
                logger.info(f"Sample string values: {string_values.head(10).tolist()}")

                unique_values = string_values.unique().tolist()
                logger.info(f"After unique(): {len(unique_values)} values")
                logger.info(f"All unique values: {unique_values}")

                # DEBUG: Test each validation step
                step1_filtered = [sid for sid in unique_values if sid]
                logger.info(f"After removing falsy values: {len(step1_filtered)} values")
                logger.info(f"Step1 filtered: {step1_filtered}")

                step2_filtered = [sid for sid in step1_filtered if str(sid).strip()]
                logger.info(f"After removing empty strings: {len(step2_filtered)} values")
                logger.info(f"Step2 filtered: {step2_filtered}")

                excluded_values = ['nan', 'None', '', 'null', 'NULL']
                step3_filtered = [sid for sid in step2_filtered if str(sid).strip() not in excluded_values]
                logger.info(f"After removing excluded values {excluded_values}: {len(step3_filtered)} values")
                logger.info(f"Step3 filtered (final): {step3_filtered}")

                # DEBUG: Check what's being excluded
                excluded = [sid for sid in step2_filtered if str(sid).strip() in excluded_values]
                if excluded:
                    logger.warning(f"These values were excluded: {excluded}")

                # Check for other problematic values
                weird_values = []
                for val in unique_values:
                    str_val = str(val).strip()
                    if len(str_val) == 0:
                        weird_values.append(f"Empty: '{val}'")
                    elif str_val in excluded_values:
                        weird_values.append(f"Excluded: '{val}'")
                    elif len(str_val) < 3:
                        weird_values.append(f"Too short: '{val}'")

                if weird_values:
                    logger.warning(f"Problematic values found: {weird_values[:10]}")

                if step3_filtered:
                    logger.info(f"SUCCESS: Extracted {len(step3_filtered)} valid session IDs")
                    return step3_filtered
                else:
                    logger.error(f"FAILED: Column '{col}' exists but all values were filtered out")

                    # Let's try a more lenient validation
                    lenient_filtered = [
                        sid for sid in unique_values 
                        if sid and str(sid).strip() and str(sid).strip().lower() not in ['nan', 'none', 'null']
                    ]

                    if lenient_filtered:
                        logger.info(f"LENIENT validation found {len(lenient_filtered)} values: {lenient_filtered}")
                        return lenient_filtered

        logger.error("No valid session IDs found in any column")
        return []

    def _generate_session_tickets_flexible(self, session_data: Dict[str, pd.DataFrame], 
                                      target_channels: List[str] = None) -> int:
        """Generate tickets from processed data using common save function"""
        try:
            logger.info("=== Starting Session-Based Ticket Generation ===")

            if session_data['sessions'].empty:
                logger.warning("No session data available for ticket generation")
                return 0

            # Create ticket engine
            engine = AutoTicketMVP(
                df_sessions=session_data['sessions'],
                df_advancetags=session_data.get('advancetags', pd.DataFrame()),
                target_channels=target_channels
            )

            # Generate tickets
            tickets_data = engine.process()


            if not tickets_data:
                logger.warning("No tickets generated by ticket engine")
                return 0

            logger.info(f"Generated {len(tickets_data)} tickets from engine")

            # Prepare tickets for bulk save
            prepared_tickets = self._prepare_tickets_for_save(tickets_data, session_data['sessions'])

            if not prepared_tickets:
                logger.warning("No valid tickets prepared for saving")
                return 0

            # Use common save function from mongo_service
            tickets_saved = save_tickets(prepared_tickets)

            logger.info(f"Successfully saved {tickets_saved} tickets")
            return tickets_saved

        except Exception as e:
            logger.error(f"Error in ticket generation: {e}")
            return 0
    
    def _prepare_tickets_for_save(self, tickets_data: List[Dict[str, Any]], 
                             sessions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare tickets data with proper session_id extraction and validation"""
        prepared_tickets = []

        for ticket_info in tickets_data:
            try:
                # Extract session ID using helper function
                session_id = self._extract_session_id_for_ticket(ticket_info, sessions_df)

                if not session_id:
                    logger.warning("Skipping ticket - no valid session ID found")
                    continue
                
                # Prepare ticket data with all required fields
                prepared_ticket = {
                    'session_id': session_id,
                    'title': ticket_info.get('title', 'Auto-generated Ticket'),
                    'description': ticket_info.get('description', ''),
                    'priority': ticket_info.get('priority', 'medium'),
                    'status': ticket_info.get('status', 'new'),
                    'assign_team': ticket_info.get('assign_team', 'technical'),
                    'issue_type': ticket_info.get('issue_type', 'video_start_failure'),
                    'confidence_score': self._extract_confidence_score(ticket_info),
                    'context_data': ticket_info.get('context_data', {}),
                    'failure_details': ticket_info.get('failure_details', {}),
                    'suggested_actions': ticket_info.get('suggested_actions', []),
                    'data_source': 'auto'
                }

                prepared_tickets.append(prepared_ticket)

            except Exception as e:
                logger.error(f"Error preparing ticket: {e}")
                continue
            
        logger.info(f"Prepared {len(prepared_tickets)} valid tickets for saving")
        return prepared_tickets

    def _extract_session_id_for_ticket(self, ticket_info: Dict[str, Any],
                                      sessions_df: pd.DataFrame) -> Optional[str]:
        """Extract valid session ID from ticket info or session data"""

        # Try from ticket info first
        if 'session_id' in ticket_info:
            session_id = str(ticket_info['session_id']).strip()
            if session_id and session_id not in ['nan', 'None', '', 'null']:
                return session_id

        # Try from sessions DataFrame - get first available session ID
        if not sessions_df.empty:
            session_columns = ['session_id', 'Session ID', 'sessionid', 'Session Id']
            for col in session_columns:
                if col in sessions_df.columns and not sessions_df[col].empty:
                    first_val = sessions_df[col].iloc[0]
                    if pd.notna(first_val):
                        session_id = str(first_val).strip()
                        if session_id and session_id not in ['nan', 'None', '', 'null']:
                            return session_id

        logger.warning("Could not extract valid session ID from ticket or sessions data")
        return None

    def _extract_confidence_score(self, ticket_info: Dict[str, Any]) -> float:
        """Extract confidence score with proper defaults"""

        # Try direct confidence_score field
        if 'confidence_score' in ticket_info:
            try:
                score = float(ticket_info['confidence_score'])
                # Ensure score is between 0 and 1
                return max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                pass
            
        # Try from failure_details confidence mapping
        if 'failure_details' in ticket_info and isinstance(ticket_info['failure_details'], dict):
            confidence_text = ticket_info['failure_details'].get('confidence', 'Medium')
            confidence_map = {
                'High': 0.9,
                'high': 0.9,
                'Medium': 0.7,
                'medium': 0.7,
                'Low': 0.5,
                'low': 0.5
            }
            return confidence_map.get(confidence_text, 0.6)

        # Default confidence score
        return 0.6
    def _extract_session_id_from_ticket(self, ticket_info: Dict[str, Any],
                                       sessions_df: pd.DataFrame) -> Optional[str]:
        """Extract valid session ID from ticket info or session data"""
        
        # Try from ticket info first
        if 'session_id' in ticket_info:
            session_id = str(ticket_info['session_id']).strip()
            if session_id and session_id not in ['nan', 'None', '']:
                return session_id
        
        # Try from sessions DataFrame
        if not sessions_df.empty:
            session_columns = ['session_id', 'Session ID', 'sessionid', 'Session Id']
            for col in session_columns:
                if col in sessions_df.columns and not sessions_df[col].empty:
                    first_val = sessions_df[col].iloc[0]
                    if pd.notna(first_val):
                        session_id = str(first_val).strip()
                        if session_id and session_id not in ['nan', 'None', '']:
                            return session_id
        
        return None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.config.cleanup_temp_files:
            try:
                gc.collect()
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage after cleanup: {memory_mb:.1f} MB")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_flexible_processor(config: ProcessingConfig = None) -> FlexibleUnifiedDataProcessor:
    """Factory function to create flexible processor"""
    return FlexibleUnifiedDataProcessor(config)

def process_files_flexible(files_mapping: Dict, target_channels: List[str] = None) -> Dict[str, Any]:
    """Process manual upload files with FULL preprocessing + ticket generation (NO MongoDB save)"""
    processor = create_flexible_processor()
    try:
        return processor.process_manual_upload_flexible(files_mapping, target_channels)
    finally:
        processor.cleanup()

def process_mongodb_flexible(target_channels: List[str] = None) -> Dict[str, Any]:
    """Process MongoDB data with LIGHT processing (column mapping only) + ticket generation"""
    processor = create_flexible_processor()
    try:
        return processor.process_mongodb_ingestion_flexible(target_channels)
    finally:
        processor.cleanup()

def detect_file_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze and detect the structure of a dataframe"""
    mapper = FlexibleColumnMapper()
    cleaner = SmartDataCleaner()
    
    # Light cleaning for analysis
    df_clean = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Try to detect data types
    detected_types = {}
    for data_type in ['sessions', 'kpi_data', 'advancetags']:
        mapped_df = mapper.flexible_map_columns(df_clean.copy(), data_type)
        mapped_columns = [col for col in mapped_df.columns if col not in df_clean.columns]
        if len(mapped_columns) > 1:
            detected_types[data_type] = {
                'mapped_columns': mapped_columns,
                'confidence': len(mapped_columns) / len(df_clean.columns) if len(df_clean.columns) > 0 else 0
            }
    
    # Extract session IDs
    session_ids = []
    session_id_cols = ['Session Id', 'session_id', 'sessionid', 'Session ID']
    for col in session_id_cols:
        if col in df_clean.columns:
            ids = df_clean[col].dropna().astype(str).unique().tolist()
            session_ids.extend(ids)
            break
    
    unique_session_ids = list(set([sid for sid in session_ids 
                                 if sid and str(sid).strip() not in ['nan', 'None', '']]))
    
    return {
        'original_shape': df.shape,
        'cleaned_shape': df_clean.shape,
        'detected_types': detected_types,
        'session_analysis': {
            'total_session_ids': len(unique_session_ids),
            'sample_session_ids': unique_session_ids[:5],
            'has_session_data': len(unique_session_ids) > 0
        },
        'column_analysis': {
            'total_columns': len(df_clean.columns),
            'data_columns': [col for col in df_clean.columns if not df_clean[col].isna().all()],
            'session_id_columns': [col for col in df_clean.columns 
                                 if any(sid_col.lower() in col.lower() for sid_col in ['session', 'id'])]
        },
        'recommended_processing': {
            'manual_upload': 'Full preprocessing + column mapping + ticket generation',
            'mongodb_data': 'Column mapping only + ticket generation'
        }
    }
