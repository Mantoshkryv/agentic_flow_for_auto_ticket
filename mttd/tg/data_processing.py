# data_processing.py - PERFECT MERGED VERSION with exact field mappings and MongoDB integration

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

# Django imports
from django.db import transaction
from django.utils import timezone
from django.core import serializers

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    chunk_size: int = 10000
    memory_threshold_mb: int = 500
    max_columns: int = 100
    enable_parallel: bool = True
    skip_validation: bool = False

class MongoDBCollectionMapper:
    """Maps data columns to MongoDB collections based on EXACT patterns and field mappings"""
    
    def __init__(self):
        self.collection_patterns = {
            'kpi_data': {  # Match KPI model db_table
                'required': ['timestamp', 'plays'],
                'patterns': [
                    # EXACT PATTERN MATCHING FOR ACTUAL COLUMNS
                    'timestamp', 'plays', 'playing time', 'streaming performance',
                    'video start failures', 'exit before video starts',
                    'video playback failures', 'video start time',
                    'rebuffering ratio', 'connection induced rebuffering',
                    'video restart time', 'peak bitrate', 'avg', 'mbps'
                ],
                'model': 'KPI'
            },
            'sessions': {  # Match Session model db_table
                'required': ['session id', 'session_id'],
                'patterns': [
                    # EXACT PATTERN MATCHING FOR ACTUAL COLUMNS  
                    'session id', 'session end time', 'session start time',
                    'asset name', 'ended session', 'impacted session',
                    'video start time', 'rebuffering ratio', 'total video restart',
                    'peak bitrate', 'average bitrate', 'framerate', 'starting bitrate',
                    'channel', 'bitrate switches', 'ended status', 'status',
                    'video start failure', 'exit before video'
                ],
                'model': 'Session'
            },
            'advancetags': {  # Match Advancetags model db_table
                'required': ['session id', 'session_id'],
                'patterns': [
                    # EXACT PATTERN MATCHING FOR ACTUAL COLUMNS
                    'session id', 'browser name', 'browser version',
                    'device hardware', 'device manufacturer', 'device marketing',
                    'device model', 'device name', 'device operating system',
                    'app name', 'app version', 'player framework',
                    'cdn', 'city', 'ip', 'ipv6', 'state', 'country',
                    'address', 'asnname', 'ispname', 'streamurl',
                    'content category', 'channel'
                ],
                'model': 'Advancetags'
            }
        }

    def get_kpi_field_mapping(self):
        """Complete KPI field mapping for exact column names"""
        return {
            'Timestamp': 'timestamp',
            'Plays': 'plays',
            'Playing Time (Ended) (mins)': 'playing_time_mins',
            'Streaming Performance Index': 'streaming_performance_index',
            'Video Start Failures Technical': 'video_start_failures_technical',
            'Video Start Failures Business': 'video_start_failures_business',
            'Exit Before Video Starts': 'exit_before_video_starts',
            'Video Playback Failures Technical': 'video_playback_failures_technical',
            'Video Playback Failures Business': 'video_playback_failures_business',
            'Video Start Time(sec)': 'video_start_time_sec',
            'Rebuffering Ratio(%)': 'rebuffering_ratio_pct',
            'Connection Induced Rebuffering Ratio(%)': 'connection_induced_rebuffering_pct',
            'Video Restart Time(sec)': 'video_restart_time_sec',
            'Avg. Peak Bitrate(Mbps)': 'avg_peak_bitrate_mbps'
        }
    
    def get_sessions_field_mapping(self):
        """Complete Sessions field mapping for exact column names"""
        return {
            'Session ID': 'session_id',
            'Session End Time': 'session_end_time',
            'Playing Time': 'playing_time',
            'Asset Name': 'asset_name',
            'Ended Session': 'ended_session',
            'Impacted Session': 'impacted_session',
            'Video Start Time': 'video_start_time',
            'Rebuffering Ratio': 'rebuffering_ratio',
            'Connection Induced Rebuffering Ratio': 'connection_induced_rebuffering_ratio',
            'Total Video Restart Time': 'total_video_restart_time',
            'Avg. Peak Bitrate': 'avg_peak_bitrate',
            'Avg. Average Bitrate': 'avg_average_bitrate',
            'Average Framerate': 'average_framerate',
            'Starting Bitrate': 'starting_bitrate',
            'channel': 'channel',
            'Bitrate Switches': 'bitrate_switches',
            'Ended Status': 'ended_status',
            'Exit Before Video Starts': 'exit_before_video_starts',
            'Session Start Time': 'session_start_time',
            'Status': 'status',
            'Video Start Failure': 'video_start_failure'
        }
    
    def get_advancetags_field_mapping(self):
        """Complete Advancetags field mapping for exact column names"""
        return {
            'Session Id': 'session_id',
            'Asset Name': 'asset_name',
            'Content Category': 'content_category',
            'Browser Name': 'browser_name',
            'Browser Version': 'browser_version',
            'Device Hardware Type': 'device_hardware_type',
            'Device Manufacturer': 'device_manufacturer',
            'Device Marketing Name': 'device_marketing_name',
            'Device Model': 'device_model',
            'Device Name': 'device_name',
            'Device Operating System': 'device_os',
            'Device Operating System Family': 'device_os_family',
            'Device Operating System Version': 'device_os_version',
            'App Name': 'app_name',
            'Player Framework Name': 'player_framework_name',
            'Player Framework Version': 'player_framework_version',
            'Last CDN': 'last_cdn',
            'App Version': 'app_version',
            'Channel': 'channel',
            'city': 'city',
            'ip': 'ip',
            'ipv6': 'ipv6',
            'cdn': 'cdn',
            'state': 'state',
            'country': 'country',
            'address': 'address',
            'asnName': 'asname',  # Note: your data uses 'asnName'
            'ispName': 'isp_name',  # Note: your data uses 'ispName'
            'streamUrl': 'stream_url'  # Note: your data uses 'streamUrl'
        }

    def detect_collection_type(self, columns: List[str]) -> Dict[str, float]:
        """Detect which collection type the columns belong to using exact patterns"""
        columns_lower = [col.lower().strip() for col in columns if col and str(col).strip()]
        
        scores = {}
        for collection, config in self.collection_patterns.items():
            score = 0
            total_patterns = len(config['patterns'])
            
            # Check required fields
            required_found = 0
            for req in config['required']:
                if any(req.lower() in col for col in columns_lower):
                    required_found += 1
            
            # Must have at least one required field
            if required_found == 0:
                scores[collection] = 0.0
                continue
                
            # Check pattern matches
            pattern_matches = 0
            for pattern in config['patterns']:
                if any(pattern.lower() in col for col in columns_lower):
                    pattern_matches += 1
            
            # Calculate score: (pattern_matches / total_patterns) * required_weight
            pattern_score = pattern_matches / total_patterns
            required_weight = required_found / len(config['required'])
            scores[collection] = (pattern_score * 0.7) + (required_weight * 0.3)
        
        return scores

    def get_best_collection_match(self, columns: List[str]) -> Optional[str]:
        """Get the best matching collection for given columns"""
        scores = self.detect_collection_type(columns)
        if not scores or max(scores.values()) < 0.2:  # Minimum threshold
            return None
        return max(scores, key=scores.get)

    def clean_kpi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process KPI data with EXACT column mapping"""
        if df.empty:
            return df
            
        df_cleaned = df.copy()
        field_mapping = self.get_kpi_field_mapping()
        
        # Apply exact field mapping
        df_mapped = pd.DataFrame()
        for source_col, target_field in field_mapping.items():
            if source_col in df_cleaned.columns:
                df_mapped[target_field] = df_cleaned[source_col]
                
                # Special handling for different data types
                if target_field == 'timestamp':
                    df_mapped[target_field] = pd.to_datetime(df_mapped[target_field], errors='coerce')
                elif target_field in ['plays', 'video_start_failures_technical', 'video_start_failures_business', 
                                    'exit_before_video_starts', 'video_playback_failures_technical', 
                                    'video_playback_failures_business']:
                    df_mapped[target_field] = pd.to_numeric(df_mapped[target_field], errors='coerce').fillna(0).astype(int)
                elif target_field in ['playing_time_mins', 'streaming_performance_index', 
                                    'video_start_time_sec', 'rebuffering_ratio_pct', 
                                    'connection_induced_rebuffering_pct', 'video_restart_time_sec', 
                                    'avg_peak_bitrate_mbps']:
                    # Handle percentage strings and convert to float
                    if df_mapped[target_field].dtype == 'object':
                        df_mapped[target_field] = df_mapped[target_field].astype(str).str.replace('%', '').str.replace(',', '')
                    df_mapped[target_field] = pd.to_numeric(df_mapped[target_field], errors='coerce')
        
        logger.info(f"Cleaned KPI data: {len(df_mapped)} rows with exact column mapping")
        return df_mapped

    def clean_session_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process session data with EXACT column mapping"""
        if df.empty:
            return df
            
        df_cleaned = df.copy()
        field_mapping = self.get_sessions_field_mapping()
        
        # Apply exact field mapping
        df_mapped = pd.DataFrame()
        for source_col, target_field in field_mapping.items():
            if source_col in df_cleaned.columns:
                df_mapped[target_field] = df_cleaned[source_col]
                
                # Special handling for different data types
                if target_field in ['session_start_time', 'session_end_time', 'video_start_time']:
                    df_mapped[target_field] = pd.to_datetime(df_mapped[target_field], errors='coerce')
                elif target_field in ['playing_time', 'total_video_restart_time',
                                    'avg_peak_bitrate', 'avg_average_bitrate', 'average_framerate',
                                    'starting_bitrate', 'rebuffering_ratio', 'connection_induced_rebuffering_ratio']:
                    # Handle percentage and numeric conversions
                    if df_mapped[target_field].dtype == 'object':
                        df_mapped[target_field] = df_mapped[target_field].astype(str).str.replace('%', '').str.replace(',', '')
                    df_mapped[target_field] = pd.to_numeric(df_mapped[target_field], errors='coerce')
                elif target_field == 'bitrate_switches':
                    df_mapped[target_field] = pd.to_numeric(df_mapped[target_field], errors='coerce').fillna(0).astype('Int64')
                elif target_field in ['ended_session', 'impacted_session', 'exit_before_video_starts', 'video_start_failure']:
                    # Boolean conversion
                    df_mapped[target_field] = df_mapped[target_field].astype(str).str.lower().isin(['true', '1', 'yes'])
                else:
                    # String fields
                    df_mapped[target_field] = df_mapped[target_field].astype(str).replace('nan', None)
        
        # Handle viewer_id extraction if not present
        if 'viewer_id' not in df_mapped.columns and 'session_id' in df_mapped.columns:
            # Extract from session_id only if viewer_id is not in source data
            df_mapped['viewer_id'] = df_mapped['session_id'].astype(str).str.split('-').str[0]
            logger.info("Extracted viewer_id from session_id")
        
        logger.info(f"Cleaned session data: {len(df_mapped)} rows with exact column mapping")
        return df_mapped

    def clean_advancetags_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process advancetags data with EXACT column mapping"""
        if df.empty:
            return df
            
        df_cleaned = df.copy()
        field_mapping = self.get_advancetags_field_mapping()
        
        # Apply exact field mapping
        df_mapped = pd.DataFrame()
        for source_col, target_field in field_mapping.items():
            if source_col in df_cleaned.columns:
                df_mapped[target_field] = df_cleaned[source_col]
                
                # Convert all to string and clean
                df_mapped[target_field] = df_mapped[target_field].astype(str).replace('nan', None)
                
                # Special handling for IP addresses
                if target_field in ['ip', 'ipv6']:
                    df_mapped[target_field] = df_mapped[target_field].replace('None', None)
        
        logger.info(f"Cleaned advancetags data: {len(df_mapped)} rows with exact column mapping")
        return df_mapped

class DataProcessor:
    """Enhanced data processor for MongoDB collections with exact field mapping"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.mapper = MongoDBCollectionMapper()
        
    def find_data_start_row(self, df: pd.DataFrame) -> int:
        """Find where actual data starts, skipping instruction rows"""
        for i in range(min(20, len(df))):  # Check first 20 rows max
            row = df.iloc[i]
            # Skip rows with mostly NaN or string instructions
            non_na_count = row.notna().sum()
            if non_na_count >= 3:  # At least 3 non-empty columns
                # Check if this looks like a header row
                str_values = row.astype(str).str.lower()
                if any(keyword in ' '.join(str_values.values) 
                      for keyword in ['session', 'timestamp', 'plays', 'isp', 'city', 'browser']):
                    return i
        return 0

    def intelligently_process_any_file(self, file_path: str, filename: str = None) -> Dict[str, pd.DataFrame]:
        """
        Enhanced file processing with exact column mapping for actual data
        Returns data keyed by db_table names: kpi_data, sessions, advancetags
        """
        results = {
            'kpi_data': pd.DataFrame(),  # Match KPI model db_table
            'sessions': pd.DataFrame(),  # Match Session model db_table
            'advancetags': pd.DataFrame()  # Match Advancetags model db_table
        }
        
        try:
            # Use provided filename or extract from path
            if not filename:
                filename = os.path.basename(file_path)
                
            logger.info(f"Processing file: {filename} at path: {file_path}")
            
            # Read file with flexible encoding
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    if file_path.endswith('.xlsx'):
                        df = pd.read_excel(file_path, header=None)
                    else:
                        df = pd.read_csv(file_path, header=None, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None or df.empty:
                logger.error(f"Could not read file: {file_path}")
                return results
            
            logger.info(f"File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Find where data actually starts
            data_start_row = self.find_data_start_row(df)
            logger.info(f"Data starts at row: {data_start_row}")
            
            # Use the detected row as header
            if data_start_row < len(df):
                df.columns = df.iloc[data_start_row]
                df = df.iloc[data_start_row + 1:].reset_index(drop=True)
            
            # Clean column names - preserve exact casing for mapping
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove completely empty columns and rows
            df = df.dropna(how='all', axis=1)
            df = df.dropna(how='all', axis=0)
            
            if df.empty:
                logger.warning("No data remaining after cleaning")
                return results
                
            logger.info(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Detect collection type using exact column names
            collection_type = self.mapper.get_best_collection_match(df.columns)
            
            if not collection_type:
                logger.warning("Could not determine collection type, attempting to split data")
                results = self._split_mixed_data(df)
            else:
                logger.info(f"Detected collection type: {collection_type}")
                # Process using exact field mapping methods
                if collection_type == 'kpi_data':
                    results['kpi_data'] = self.mapper.clean_kpi_data(df)
                elif collection_type == 'sessions':
                    results['sessions'] = self.mapper.clean_session_data(df)
                elif collection_type == 'advancetags':
                    results['advancetags'] = self.mapper.clean_advancetags_data(df)
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            
        return results

    def _split_mixed_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split mixed data into different collection types"""
        results = {
            'kpi_data': pd.DataFrame(),
            'sessions': pd.DataFrame(),
            'advancetags': pd.DataFrame()
        }
        
        # Group columns by collection type
        column_groups = {'kpi_data': [], 'sessions': [], 'advancetags': []}
        
        for col in df.columns:
            scores = self.mapper.detect_collection_type([col])
            if scores and max(scores.values()) > 0.1:
                best_match = max(scores, key=scores.get)
                
                if best_match in column_groups:
                    column_groups[best_match].append(col)
        
        # Process each group
        for collection_type, columns in column_groups.items():
            if columns:
                subset_df = df[columns].copy()
                subset_df = subset_df.dropna(how='all')
                
                if not subset_df.empty:
                    if collection_type == 'kpi_data':
                        results['kpi_data'] = self.mapper.clean_kpi_data(subset_df)
                    elif collection_type == 'sessions':
                        results['sessions'] = self.mapper.clean_session_data(subset_df)
                    elif collection_type == 'advancetags':
                        results['advancetags'] = self.mapper.clean_advancetags_data(subset_df)
        
        return results

    def fetch_database_df(self, model_class):
        """Fetch data from Django model and convert to DataFrame for MongoDB processing"""
        try:
            from django.core import serializers
            import json
            
            # Get all records from the model
            queryset = model_class.objects.all()
            
            if not queryset.exists():
                logger.info(f"No data found in {model_class.__name__}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            serialized = serializers.serialize('json', queryset)
            data = json.loads(serialized)
            
            # Extract fields from Django serialization format
            records = []
            for item in data:
                record = item['fields'].copy()
                record['id'] = item['pk']  # Add primary key
                records.append(record)
            
            df = pd.DataFrame(records)
            logger.info(f"Fetched {len(df)} records from {model_class.__name__} collection")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {model_class.__name__}: {e}")
            return pd.DataFrame()

    def save_to_mongodb_collection(self, df: pd.DataFrame, model_class, collection_type: str) -> int:
        """Save DataFrame to MongoDB collection via Django model"""
        if df.empty:
            logger.warning(f"No data to save to {collection_type} collection")
            return 0
        
        saved_count = 0
        try:
            with transaction.atomic():
                for _, row in df.iterrows():
                    try:
                        # Convert row to dict and handle NaN values
                        data = row.to_dict()
                        data = {k: v for k, v in data.items() 
                               if pd.notna(v) and v is not None and str(v).strip() != ''}
                        
                        # Create and save model instance
                        instance = model_class(**data)
                        instance.save()
                        saved_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error saving row to {collection_type}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in batch save to {collection_type}: {e}")
        
        logger.info(f"Saved {saved_count} records to {collection_type} collection")
        return saved_count
