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
from .operation.ticket_engine import AutoTicketMVP
from .mongo_service import test_mongodb_connection, fetch_collections, save_tickets
# Import existing functions to prevent duplicacy
try:
    from .models import Session, KPI, Advancetags, Ticket
    from .data_validation import create_data_validator, ComprehensiveDataValidator
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
            'viewer_id': {
                'exact': ['viewer_id', 'Viewer ID', 'viewerid', 'ViewerID'],
                'contains': ['viewer', 'viewer_id'],
                'patterns': [r'viewer.*id', r'id.*viewer']
            },
            'session_id': {
                'exact': ['session_id', 'Session ID', 'Session_id', 'Session Id'],
                'contains': ['session', 'session_id'],
                'patterns': [r'session.*id', r'id.*session']
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
            'viewer_id': {
                'exact': ['viewer_id', 'Viewer ID', 'viewerid', 'ViewerID'],
                'contains': ['viewer', 'viewer_id'],
                'patterns': [r'viewer.*id', r'id.*viewer']
            },
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
        """
        FIXED: Apply flexible column mapping with PROPER detection

        Flow:
        1. Check if columns are already in standard format (viewer_id, session_id)
        2. If not, apply flexible mapping
        3. Verify critical columns exist after mapping
        """
        if df is None or df.empty:
            logger.warning(f"Empty dataframe provided for {data_type} mapping")
            return df

        logger.info("=" * 70)
        logger.info(f"🔄 COLUMN MAPPING: {data_type.upper()}")
        logger.info("=" * 70)
        logger.info(f"Input columns: {list(df.columns)}")

        if data_type == 'sessions':
            mapping_dict = self.session_mappings
            critical_columns = ['viewer_id', 'session_id', 'asset_name', 'status']
        elif data_type == 'kpi_data':
            mapping_dict = self.kpi_mappings
            critical_columns = ['timestamp', 'plays']
        elif data_type == 'advancetags':
            mapping_dict = self.advancetags_mappings
            critical_columns = ['viewer_id', 'session_id']
        else:
            logger.warning(f"Unknown data type: {data_type}")
            return df

        # ✅ STEP 1: Check if columns are ALREADY in standard format
        already_standard = []
        for standard_name in mapping_dict.keys():
            if standard_name in df.columns:
                already_standard.append(standard_name)

        if already_standard:
            logger.info(f"✅ Found {len(already_standard)} columns already in standard format:")
            for col in already_standard:
                logger.info(f"   - {col}")

        # ✅ STEP 2: Build column mapping for non-standard columns
        column_map = {}
        available_columns = df.columns.tolist()

        for standard_name, match_config in mapping_dict.items():
            # Skip if already in standard format
            if standard_name in df.columns:
                continue
            
            # Find matching column
            mapped_column = self._find_column_flexible(available_columns, match_config)
            if mapped_column:
                column_map[mapped_column] = standard_name
                logger.info(f"🔀 Mapping: '{mapped_column}' → '{standard_name}'")

        # ✅ STEP 3: Apply mapping
        if column_map:
            df_mapped = df.rename(columns=column_map)
            logger.info(f"✅ Mapped {len(column_map)} columns")
        else:
            df_mapped = df
            if not already_standard:
                logger.warning(f"⚠️ No mapping applied and no standard columns found!")
            else:
                logger.info(f"✅ All columns already in standard format - no mapping needed")

        # ✅ STEP 4: CRITICAL VERIFICATION for sessions
        if data_type == 'sessions':
            logger.info("🔍 CRITICAL COLUMN VERIFICATION:")
            for critical_col in critical_columns:
                exists = critical_col in df_mapped.columns
                symbol = "✅" if exists else "❌"
                logger.info(f"   {symbol} {critical_col}: {exists}")

                if exists and critical_col in ['viewer_id', 'session_id']:
                    sample_values = df_mapped[critical_col].dropna().unique()[:3].tolist()
                    logger.info(f"      Sample values: {sample_values}")

            # CRITICAL ERROR if viewer_id missing
            if 'viewer_id' not in df_mapped.columns:
                logger.error("=" * 70)
                logger.error("❌ CRITICAL ERROR: viewer_id column NOT FOUND after mapping!")
                logger.error(f"Available columns: {list(df_mapped.columns)}")
                logger.error("Viewer-first architecture requires viewer_id!")
                logger.error("=" * 70)

        logger.info(f"Output columns: {list(df_mapped.columns)[:10]}")
        logger.info("=" * 70)

        return df_mapped

    def _find_column_flexible(self, available_columns: List[str], match_config: Dict[str, List]) -> Optional[str]:
        """ENHANCED: Find matching column with detailed logging"""

        # Try exact matches first (case-insensitive)
        for exact_name in match_config.get('exact', []):
            for col in available_columns:
                col_clean = str(col).lower().strip()
                exact_clean = str(exact_name).lower().strip()
                if col_clean == exact_clean:
                    logger.debug(f"✅ Exact match: '{col}' == '{exact_name}'")
                    return col

        # Try contains matches
        for contains_text in match_config.get('contains', []):
            contains_clean = str(contains_text).lower().strip()
            for col in available_columns:
                col_clean = str(col).lower().strip()
                if contains_clean in col_clean:
                    logger.debug(f"✅ Contains match: '{col}' contains '{contains_text}'")
                    return col

        # Try pattern matches
        for pattern in match_config.get('patterns', []):
            for col in available_columns:
                col_clean = str(col).lower().strip()
                if re.search(pattern, col_clean, re.IGNORECASE):
                    logger.debug(f"✅ Pattern match: '{col}' matches '{pattern}'")
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
                    "data_counts": {"sessions": 0, "kpi_data": 0, "advancetags": 0}
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
                        df_mapped = self._filter_by_channels(df_mapped, target_channels)

                    processed_data[datatype] = df_mapped
                    logger.info(f"✅ Processed {datatype}: {len(df_mapped)} records")
                else:
                    processed_data[datatype] = df

            # Step 3: Extract session IDs
            #session_ids = self._extract_session_ids(processed_data)
            
            # CHANGE TO:
            session_ids = self._extract_session_ids(processed_data)
            
            # HANDLE EMPTY RESULT GRACEFULLY
            if not session_ids:
                logger.warning("No session IDs extracted - database appears to be empty")
                
            
            logger.info(f"📋 Extracted {len(session_ids)} unique session IDs")

            # Step 4: Generate tickets
            tickets_generated = self._generate_session_tickets_flexible(processed_data, target_channels)

            # Step 5: Return success result
            return {
                "success": True,
                "processing_type": "mongodb_django_orm",
                "session_ids_processed": len(session_ids),
                "tickets_generated": tickets_generated,
                "target_channels": target_channels or [],
                "data_counts": {k: len(v) for k, v in processed_data.items()},
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
        """
        FIXED: Extract session IDs directly from sessions DataFrame
        No viewer grouping - just direct extraction
        """
        if 'sessions' not in data or data['sessions'].empty:
            logger.warning("No sessions data")
            return []

        df = data['sessions']

        logger.info("=" * 70)
        logger.info("📋 EXTRACTING SESSION IDs (Direct Approach)")
        logger.info("=" * 70)
        logger.info(f"Total sessions: {len(df)}")
        logger.info(f"Available columns: {list(df.columns)[:10]}")

        # ✅ FIXED: Direct extraction - no viewer grouping
        session_id_col = None

        # Try standard column names
        for col in ['session_id', 'Session ID', 'Session Id', 'sessionid', 'SessionID']:
            if col in df.columns:
                session_id_col = col
                logger.info(f"✅ Found session_id column: '{col}'")
                break
            
        if not session_id_col:
            logger.error(f"❌ No session_id column found. Columns: {list(df.columns)}")
            return []

        # Extract and validate session IDs
        values = df[session_id_col].dropna().astype(str)
        valid_ids = [v.strip() for v in values 
                     if v.strip() and v.strip().lower() not in ['nan', 'none', '', 'null']]

        unique_ids = list(set(valid_ids))

        logger.info(f"✅ Extracted {len(unique_ids)} unique session IDs")
        logger.info(f"📋 Sample IDs: {unique_ids[:5]}")
        logger.info("=" * 70)

        return unique_ids

    def _generate_session_tickets_flexible(self, session_data: Dict[str, pd.DataFrame], 
                                  target_channels: List[str] = None) -> int:
        """
        FIXED: Session-first ticket generation with VERIFIED column mapping
        """
        try:
            logger.info("=" * 70)
            logger.info("🎫 TICKET GENERATION: Session-First Architecture")
            logger.info("=" * 70)
    
            if session_data['sessions'].empty:
                logger.warning("❌ No session data available")
                return 0
    
            # ✅ STEP 1: Apply column mapping
            logger.info("📄 Step 1: Applying column mapping...")
            sessions_mapped = self.mapper.flexible_map_columns(
                session_data['sessions'], 'sessions'
            )
    
            # ✅ CRITICAL: Verify column mapping worked
            logger.info("=" * 70)
            logger.info("🔍 COLUMN MAPPING VERIFICATION")
            logger.info("=" * 70)
            logger.info(f"Before mapping: {list(session_data['sessions'].columns)[:10]}")
            logger.info(f"After mapping:  {list(sessions_mapped.columns)[:10]}")
            
            # Check for critical columns after mapping
            critical_cols = {
                'session_id': any('session' in str(c).lower() and 'id' in str(c).lower() 
                                for c in sessions_mapped.columns),
                'viewer_id': any('viewer' in str(c).lower() and 'id' in str(c).lower() 
                               for c in sessions_mapped.columns),
                'status': any('status' in str(c).lower() for c in sessions_mapped.columns)
            }
            
            logger.info("Critical columns present:")
            for col, present in critical_cols.items():
                symbol = "✅" if present else "⚠️"
                logger.info(f"  {symbol} {col}: {present}")
            
            if not critical_cols['session_id']:
                logger.error("❌ CRITICAL: No session_id column after mapping!")
                logger.error(f"Available columns: {list(sessions_mapped.columns)}")
                return 0
            
            logger.info("=" * 70)
    
            # ✅ STEP 2: Apply mapping to advancetags (if available)
            logger.info("📄 Step 2: Mapping advancetags...")
            advancetags_mapped = pd.DataFrame()
            if not session_data.get('advancetags', pd.DataFrame()).empty:
                advancetags_mapped = self.mapper.flexible_map_columns(
                    session_data['advancetags'], 'advancetags'
                )
                logger.info(f"  ✅ Mapped {len(advancetags_mapped)} advancetags records")
    
            # ✅ STEP 3: Create ticket engine with MAPPED data
            logger.info("📄 Step 3: Creating ticket engine...")
            engine = AutoTicketMVP(
                df_sessions=sessions_mapped,
                df_advancetags=advancetags_mapped,
                target_channels=target_channels
            )
    
            # ✅ STEP 4: Generate tickets
            logger.info("📄 Step 4: Generating tickets...")
            tickets = engine.process()
    
            if not tickets:
                logger.warning("⚠️ No tickets generated")
                return 0
    
            # ✅ STEP 5: Prepare tickets for save
            logger.info("📄 Step 5: Preparing tickets for save...")
            prepared_tickets = self._prepare_tickets_for_save(tickets, sessions_mapped)
    
            if not prepared_tickets:
                logger.warning("⚠️ No valid tickets to save")
                return 0
    
            # ✅ STEP 6: Save tickets
            logger.info("📄 Step 6: Saving tickets...")
            tickets_saved = save_tickets(prepared_tickets)
    
            logger.info(f"✅ Successfully saved {tickets_saved} tickets")
            logger.info("=" * 70)
            return tickets_saved
    
        except Exception as e:
            logger.error(f"❌ Ticket generation failed: {e}", exc_info=True)
            return 0
 
    def _prepare_tickets_for_save(self, tickets_data: List[Dict[str, Any]], 
                     sessions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare tickets with FULL enhanced structure preservation"""
        prepared = []

        logger.info(f"Preparing {len(tickets_data)} tickets for save with enhanced structure")

        for ticket in tickets_data:
            try:
                # Extract session_id
                session_id = ticket.get('session_id')

                if not session_id:
                    logger.warning(f"Ticket missing session_id: {ticket.get('ticket_id')}")
                    continue
                
                session_id = str(session_id).strip()

                # Basic validation only
                if not session_id or session_id.lower() in ['nan', 'none']:
                    logger.warning(f"Invalid session_id: '{session_id}'")
                    continue
                
                # Extract enhanced fields
                failure_details = ticket.get('failure_details', {})
                context_data = ticket.get('context_data', {})

                # Prepare ticket with full enhanced structure
                prepared_ticket = {
                    
                    'session_id': session_id,
                    'ticket_id': ticket.get('ticket_id', f"TKT_{session_id}"),
                    'title': ticket.get('title', 'Auto-generated Ticket'),
                    'description': ticket.get('description', ''),
                    'priority': ticket.get('priority', 'medium'),
                    'status': ticket.get('status', 'new'),
                    'assign_team': ticket.get('assign_team', 'technical'),
                    'issue_type': ticket.get('issue_type', 'video_start_failure'),

                    # ENHANCED: Include confidence and severity
                    'confidence_score': ticket.get('confidence_score', 0.6),
                    'severity_score': ticket.get('severity_score', failure_details.get('severity_score', 5)),

                    # ENHANCED: Preserve complete failure_details structure
                    'failure_details': {
                        'root_cause': failure_details.get('root_cause', 'Technical Investigation Needed'),
                        'confidence': failure_details.get('confidence', 0.6),
                        'confidence_percentage': failure_details.get('confidence_percentage', 60),
                        'evidence': failure_details.get('evidence', ''),
                        'failure_type': failure_details.get('failure_type', 'VSF-T'),
                        'severity_score': failure_details.get('severity_score', 5),
                        'severity_label': failure_details.get('severity_label', 'MEDIUM'),

                        # ENHANCED: Multi-layer diagnostic data
                        'user_behavior': failure_details.get('user_behavior', {}),
                        'temporal_analysis': failure_details.get('temporal_analysis', {}),
                        'geographic_analysis': failure_details.get('geographic_analysis', {})
                    },

                    # ENHANCED: Context and actions
                    'context_data': {
                        'asset_name': context_data.get('asset_name', 'Unknown'),
                        'viewer_id': context_data.get('viewer_id', 'Unknown'),
                        'failure_time': context_data.get('failure_time'),
                        'deep_link': context_data.get('deep_link', '')
                    },

                    'suggested_actions': ticket.get('suggested_actions', []),
                    'alerts': ticket.get('alerts', []),
                    'data_source': 'auto'
                }

                prepared.append(prepared_ticket)

            except Exception as e:
                logger.error(f"Error preparing ticket: {e}")
                continue
            
        logger.info(f"Prepared {len(prepared)}/{len(tickets_data)} tickets with enhanced structure")

        # Log sample of prepared ticket structure
        if prepared:
            sample = prepared[0]
            logger.info(f"Sample ticket structure - Root cause: {sample['failure_details'].get('root_cause')}, "
                       f"Confidence: {sample['confidence_score']:.2f}, "
                       f"Severity: {sample['severity_score']}/10")

        return prepared
    
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
