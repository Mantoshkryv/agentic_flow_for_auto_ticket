# forms.py - ENHANCED FLEXIBLE UPLOAD FORMS

"""
Enhanced Forms with Flexible Upload Support
===========================================

Features:
- Flexible multi-file upload support
- Smart validation with existing functions
- Variable channel support (no hardcoded names)
- Session-only ticket generation options
- Prevents code duplicacy through imports
"""

from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from typing import List
import logging
from pathlib import Path
import tempfile
import os
import pandas as pd

# Import existing functions to prevent duplicacy
try:
    from .data_processing import detect_file_structure
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import validation dependencies")

logger = logging.getLogger(__name__)

# ============================================================================
# MAIN FLEXIBLE UPLOAD FORM
# ============================================================================

class FlexibleUploadForm(forms.Form):
    """
    Enhanced flexible upload form supporting multiple files and data types
    Maintains existing interface while adding new features
    """
    
    # Primary file upload (required)
    file_1 = forms.FileField(
        label=_("Primary Data File (Required)"),
        widget=forms.FileInput(attrs={
            "accept": ".csv,.xlsx,.xls",
            "class": "form-control",
            "data-file-number": "1"
        }),
        help_text=_("Main data file containing session, KPI, and/or metadata. Supports CSV and Excel formats.")
    )
    
    file_1_contains = forms.MultipleChoiceField(
        label=_("Primary file contains:"),
        choices=[
            ('sessions', _('Session Data')),
            ('kpi_data', _('KPI Data')), 
            ('advancetags', _('Metadata/Advanced Tags')),
            ('auto_detect', _('Auto-detect data types'))
        ],
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input',
            'data-file-number': '1'
        }),
        help_text=_("Select data types present, or choose auto-detect for intelligent recognition"),
        initial=['auto_detect']
    )
    
    # Optional secondary file
    file_2 = forms.FileField(
        label=_("Secondary Data File (Optional)"),
        widget=forms.FileInput(attrs={
            "accept": ".csv,.xlsx,.xls",
            "class": "form-control",
            "data-file-number": "2"
        }),
        required=False,
        help_text=_("Additional data file for supplementary data types")
    )
    
    file_2_contains = forms.MultipleChoiceField(
        label=_("Secondary file contains:"),
        choices=[
            ('sessions', _('Session Data')),
            ('kpi_data', _('KPI Data')),
            ('advancetags', _('Metadata/Advanced Tags')),
            ('auto_detect', _('Auto-detect data types'))
        ],
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input',
            'data-file-number': '2'
        }),
        required=False,
        help_text=_("Select data types in secondary file")
    )
    
    # Optional tertiary file
    file_3 = forms.FileField(
        label=_("Tertiary Data File (Optional)"),
        widget=forms.FileInput(attrs={
            "accept": ".csv,.xlsx,.xls",
            "class": "form-control",
            "data-file-number": "3"
        }),
        required=False,
        help_text=_("Third data file for remaining data types")
    )
    
    file_3_contains = forms.MultipleChoiceField(
        label=_("Tertiary file contains:"),
        choices=[
            ('sessions', _('Session Data')),
            ('kpi_data', _('KPI Data')),
            ('advancetags', _('Metadata/Advanced Tags')),
            ('auto_detect', _('Auto-detect data types'))
        ],
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input',
            'data-file-number': '3'
        }),
        required=False,
        help_text=_("Select data types in tertiary file")
    )
    
    # VARIABLE channel support - NO HARDCODED NAMES
    target_channels = forms.CharField(
        label=_("Target Channels (Optional)"),
        max_length=1000,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Sun TV HD, KTV, Colors HD, Star Sports (comma-separated)'
        }),
        help_text=_("Specify channels to focus on for ticket generation. Leave empty to process all channels.")
    )
    
    # Processing configuration options
    generate_tickets = forms.BooleanField(
        label=_("Generate Tickets for Video Start Failures"),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Automatically generate tickets for Video Start Failures using MVP diagnosis rules")
    )
    
    session_only_tickets = forms.BooleanField(
        label=_("Use Session-Only Ticket Generation"),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Generate tickets using only session IDs (recommended MVP approach)")
    )
    
    skip_transient = forms.BooleanField(
        label=_("Skip Transient Network Issues"),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Skip creating tickets for temporary network problems (confidence < 70%)")
    )
    
    flexible_column_mapping = forms.BooleanField(
        label=_("Enable Flexible Column Mapping"),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Intelligently map column names from any position in the data")
    )
    
    remove_blank_data = forms.BooleanField(
        label=_("Remove Blank Columns and Rows"),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Automatically remove mostly empty columns (95%+ blank) and rows (90%+ blank)")
    )
    
    clean_instructions = forms.BooleanField(
        label=_("Clean Instruction Headers"),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Remove instruction text, headers, and unnecessary content from files")
    )

    def clean(self):
        """Enhanced validation with flexible processing"""
        cleaned_data = super().clean()
        
        # Get uploaded files and their data type specifications
        files_data = [
            (cleaned_data.get('file_1'), cleaned_data.get('file_1_contains', []), 'Primary File'),
            (cleaned_data.get('file_2'), cleaned_data.get('file_2_contains', []), 'Secondary File'),
            (cleaned_data.get('file_3'), cleaned_data.get('file_3_contains', []), 'Tertiary File'),
        ]
        
        # Validate primary file (required)
        if not files_data[0][0]:
            raise ValidationError(_("Primary data file is required."))
        if not files_data[0][1]:
            raise ValidationError(_("Please specify what data types are in the primary file."))
        
        # Validate optional files
        for file_obj, data_types, file_label in files_data[1:]:
            if file_obj and not data_types:
                raise ValidationError(
                    _("Please specify what data types are in %(file)s.") % {'file': file_label}
                )
        
        # Process target channels - VARIABLE SUPPORT
        target_channels_str = cleaned_data.get('target_channels', '')
        if target_channels_str:
            # Parse comma-separated channels with flexible handling
            channels = []
            for channel in target_channels_str.split(','):
                channel = channel.strip()
                if channel:
                    channels.append(channel)
            cleaned_data['target_channels_list'] = channels
        else:
            cleaned_data['target_channels_list'] = []
        
        # Enhanced file validation with structure detection
        for file_obj, data_types, file_label in files_data:
            if file_obj:
                try:
                    # Validate file content and structure
                    validation_result = self._validate_file_content_enhanced(file_obj, data_types, file_label)
                    
                    if not validation_result['is_valid']:
                        raise ValidationError(
                            _("%(file)s: %(error)s") % {
                                'file': file_label,
                                'error': validation_result['error']
                            }
                        )
                    
                    # Add structure info to cleaned data
                    cleaned_data[f'{file_label.lower().replace(" ", "_")}_structure'] = validation_result['structure']
                    
                    logger.info(f"{file_label} validation successful: {validation_result['info']}")
                    
                except Exception as e:
                    raise ValidationError(
                        _("Error validating %(file)s: %(error)s") % {
                            'file': file_label,
                            'error': str(e)
                        }
                    )
        
        return cleaned_data

    def _validate_file_content_enhanced(self, file_obj, data_types: list, file_label: str) -> dict:
        """Enhanced file content validation with structure detection"""
        # ensure file_label is referenced so linters don't flag it as unused;
        # also include the label in returned errors for clearer context
        if not file_obj:
            return {'is_valid': False, 'error': f"No file provided for {file_label}"}
            
        filename = getattr(file_obj, 'name', f'{file_label}_uploaded_file')
        file_extension = Path(filename).suffix.lower()
        # Check file extension
        if file_extension not in ['.csv', '.xlsx', '.xls']:
            return {
                'is_valid': False, 
                'error': f"Unsupported format: {file_extension}. Use CSV or Excel files."
            }
        
        try:
            # Create temporary file for analysis
            temp_path = self._create_temp_file(file_obj, file_extension)
            
            try:
                # Read and analyze file structure
                df = self._read_file_safely(temp_path)
                
                if df.empty:
                    return {'is_valid': False, 'error': "File contains no readable data"}
                
                if len(df.columns) == 0:
                    return {'is_valid': False, 'error': "File contains no columns"}
                
                # Detect file structure using existing function
                try:
                    structure_info = detect_file_structure(df)
                except:
                    # Fallback structure detection
                    structure_info = self._basic_structure_detection(df)
                
                # Validate against specified data types
                if 'auto_detect' not in data_types:
                    validation_errors = self._validate_data_types_match(df, data_types)
                    if validation_errors:
                        return {
                            'is_valid': False,
                            'error': f"Data type validation failed: {'; '.join(validation_errors)}"
                        }
                
                return {
                    'is_valid': True,
                    'error': None,
                    'info': f"{len(df)} rows, {len(df.columns)} columns",
                    'structure': structure_info
                }
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            return {'is_valid': False, 'error': f"Error reading file: {str(e)}"}

    def _create_temp_file(self, file_obj, extension):
        """Create temporary file for validation - uses existing method pattern"""
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(suffix=extension)
        
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                # Reset file pointer
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                    
                # Write file content
                if hasattr(file_obj, 'chunks'):
                    for chunk in file_obj.chunks():
                        temp_file.write(chunk)
                else:
                    content = file_obj.read()
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    temp_file.write(content)
            return temp_path
        except Exception:
            os.close(temp_fd)
            raise

    def _read_file_safely(self, path: str) -> pd.DataFrame:
        """Read a file from disk (temp path) safely and return a DataFrame.

        Supports CSV and Excel files. Uses simple encoding and separator detection
        for CSVs and falls back to robust pandas engines.
        """
        file_extension = Path(path).suffix.lower()

        if file_extension in ['.xlsx', '.xls']:
            for engine in ['openpyxl', 'xlrd']:
                try:
                    return pd.read_excel(path, engine=engine)
                except Exception:
                    continue
            raise ValueError("Could not read Excel file")

        if file_extension == '.csv':
            # Read raw bytes and attempt decoding
            with open(path, 'rb') as f:
                content = f.read()

            # Try common encodings
            text = None
            for enc in ('utf-8', 'utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1'):
                try:
                    text = content.decode(enc)
                    break
                except Exception:
                    continue

            if text is None:
                raise ValueError('Could not decode CSV file with common encodings')

            from io import StringIO
            first_line = text.split('\n', 1)[0] if text else ''
            separators = [',', ';', '\t', '|']
            best_sep = ','
            max_count = -1
            for sep in separators:
                cnt = first_line.count(sep)
                if cnt > max_count:
                    max_count = cnt
                    best_sep = sep

            # Try parsing with best separator first
            try:
                df = pd.read_csv(StringIO(text), sep=best_sep, engine='python', on_bad_lines='skip')
                if not df.empty:
                    return df
            except Exception:
                pass

            # Fallback: try all separators
            for sep in separators:
                try:
                    df = pd.read_csv(StringIO(text), sep=sep, engine='python', on_bad_lines='skip')
                    if not df.empty:
                        return df
                except Exception:
                    continue

            # Last resort: try pandas autodetect
            try:
                from io import BytesIO
                return pd.read_csv(BytesIO(content))
            except Exception:
                raise ValueError('Could not parse CSV file')

        raise ValueError(f'Unsupported file format: {file_extension}')

    def _read_file_flexible(self, file_obj, filename: str) -> pd.DataFrame:
        """Enhanced file reading optimized for Conviva reports"""
        file_extension = Path(filename).suffix.lower()
        logger.info(f"Reading {filename} ({file_extension})")
        
        try:
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
                # Read entire file as text for Conviva report processing
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                
                content = file_obj.read()
                if isinstance(content, bytes):
                    # Try common encodings
                    for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                        try:
                            text_content = content.decode(encoding)
                            break
                        except:
                            continue
                    else:
                        raise ValueError("Could not decode file")
                else:
                    text_content = content
                
                # Split into lines and find data section
                lines = text_content.split('\n')
                logger.info(f"File has {len(lines)} lines")
                
                # Look for Conviva data patterns
                data_start_idx = None
                headers = None
                
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    
                    # Skip obvious metadata/instruction lines
                    if any(skip in line_lower for skip in [
                        'report generated', 'copyright', 'conviva', 'confidential',
                         'data policy', 'summary data'
                    ]):
                        continue
                    
                    # Look for data header indicators
                    if any(indicator in line_lower for indicator in [
                        'Session Id', 'timestamp', 'plays',
                        'asset name', 'browser name', 'device name'
                    ]):
                        # This line contains data headers
                        # Try different separators to parse it
                        for sep in ['\t', ',', ';', '|']:
                            if sep in line:
                                potential_headers = [h.strip() for h in line.split(sep) if h.strip()]
                                if len(potential_headers) >= 5:  # Reasonable number of columns
                                    headers = potential_headers
                                    data_start_idx = i + 1
                                    logger.info(f"Found headers at line {i}: {len(headers)} columns")
                                    break
                        
                        if headers:
                            break
                
                if not headers or data_start_idx is None:
                    raise ValueError("Could not find data headers in Conviva report")
                
                # Extract data rows
                data_rows = []
                separator = '\t'  # Conviva typically uses tab separation
                
                # Auto-detect separator from header line
                header_line = lines[data_start_idx - 1]
                for sep in ['\t', ',', ';', '|']:
                    if header_line.count(sep) >= len(headers) - 1:
                        separator = sep
                        break
                
                for line in lines[data_start_idx:]:
                    if line.strip():
                        # Split line by separator
                        row_data = [cell.strip() for cell in line.split(separator)]
                        
                        # Pad or truncate to match header count
                        while len(row_data) < len(headers):
                            row_data.append('')
                        row_data = row_data[:len(headers)]
                        
                        data_rows.append(row_data)
                
                if not data_rows:
                    raise ValueError("No data rows found after headers")
                
                # Create DataFrame
                df = pd.DataFrame(data_rows, columns=headers)
                logger.info(f"Created DataFrame: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
                
                return df
                
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            raise
    
    def _basic_structure_detection(self, df: pd.DataFrame) -> dict:
        """Basic structure detection fallback"""
        columns = df.columns.tolist()
        
        # Simple pattern detection
        session_patterns = ['Session', 'Session End Time', 'Asset', 'Status']
        kpi_patterns = ['Plays', 'Performance', 'Failures', 'Technical']
        meta_patterns = ['Browser', 'Device', 'ISP', 'City']

        scores = {}
        for data_type, patterns in [
            ('sessions', session_patterns),
            ('kpi_data', kpi_patterns),
            ('advancetags', meta_patterns)
        ]:
            score = sum(1 for col in columns if any(pattern in col.lower() for pattern in patterns))
            if score > 0:
                scores[data_type] = score / len(columns)
        
        return {
            'original_shape': df.shape,
            'cleaned_shape': df.shape,  # Would be different after cleaning
            'detected_types': scores,
            'column_analysis': {
                'total_columns': len(columns),
                'data_columns': columns,
                'blank_columns': []
            }
        }

    def _validate_data_types_match(self, df: pd.DataFrame, specified_types: list) -> list:
        """Validate that file content matches specified data types"""
        errors = []
        columns_lower = [str(col).lower() for col in df.columns]
        
        # Expected patterns for each data type
        type_patterns = {
            'sessions': ['session', 'viewer', 'asset', 'status'],
            'kpi_data': ['plays', 'timestamp', 'performance', 'failures'],
            'advancetags': ['browser', 'device', 'isp', 'city', 'cdn']
        }
        
        for data_type in specified_types:
            if data_type in type_patterns:
                patterns = type_patterns[data_type]
                matches = sum(1 for col in columns_lower if any(pattern in col for pattern in patterns))
                
                if matches == 0:
                    errors.append(f"No {data_type} columns found")
        
        return errors

    def get_file_mapping(self):
        """Get file mapping dictionary - maintains existing interface"""
        if not hasattr(self, 'cleaned_data') or not self.cleaned_data:
            return {}
            
        mapping = {}
        
        # Process each file
        files_data = [
            (self.cleaned_data.get('file_1'), self.cleaned_data.get('file_1_contains', [])),
            (self.cleaned_data.get('file_2'), self.cleaned_data.get('file_2_contains', [])),
            (self.cleaned_data.get('file_3'), self.cleaned_data.get('file_3_contains', [])),
        ]
        
        for file_obj, data_types in files_data:
            if file_obj and data_types:
                mapping[file_obj] = data_types
                
        return mapping

    def get_processing_config(self):
        """Get enhanced processing configuration"""
        if not hasattr(self, 'cleaned_data') or not self.cleaned_data:
            return {}
            
        return {
            'target_channels': self.cleaned_data.get('target_channels_list', []),  # VARIABLE channels
            'generate_tickets': self.cleaned_data.get('generate_tickets', True),
            'session_only_tickets': self.cleaned_data.get('session_only_tickets', True),
            'skip_transient': self.cleaned_data.get('skip_transient', True),
            'flexible_column_mapping': self.cleaned_data.get('flexible_column_mapping', True),
            'remove_blank_data': self.cleaned_data.get('remove_blank_data', True),
            'clean_instructions': self.cleaned_data.get('clean_instructions', True),
        }

    def get_upload_summary(self):
        """Get upload summary - maintains existing interface"""
        if not hasattr(self, 'cleaned_data') or not self.cleaned_data:
            return _("Form not validated")
            
        mapping = self.get_file_mapping()
        config = self.get_processing_config()
        
        summary_parts = []
        
        # File summary
        for i, (file_obj, data_types) in enumerate(mapping.items(), 1):
            file_name = getattr(file_obj, 'name', f'File {i}')
            data_list = ', '.join(data_types)
            summary_parts.append(f"{file_name}: {data_list}")
        
        # Configuration summary
        if config['target_channels']:
            summary_parts.append(f"Target channels: {', '.join(config['target_channels'])}")
        
        features = []
        if config['session_only_tickets']:
            features.append("Session-only tickets")
        if config['flexible_column_mapping']:
            features.append("Flexible mapping")
        if config['remove_blank_data']:
            features.append("Blank data removal")
        if config['clean_instructions']:
            features.append("Instruction cleaning")
        
        if features:
            summary_parts.append(f"Features: {', '.join(features)}")
            
        return " | ".join(summary_parts)

# ============================================================================
# SMART UPLOAD FORM - STREAMLINED VERSION
# ============================================================================

class SmartUploadForm(forms.Form):
    """
    Streamlined smart upload form with automatic detection
    Maintains existing interface while leveraging new capabilities
    """
    
    # Single primary file with smart detection
    primary_file = forms.FileField(
        label=_("Upload Data File"),
        help_text=_("Upload your data file (CSV/Excel). Data types will be automatically detected and processed."),
        widget=forms.ClearableFileInput(attrs={
            "accept": ".csv,.xlsx,.xls",
            "class": "form-control",
            "data-file-type": "smart"
        })
    )
    
    # Optional additional files (allow multiple)
    additional_files = forms.FileField(
        label=_("Additional Files (Optional)"),
        help_text=_("Upload additional data files if you have multiple files with different data types (use request.FILES.getlist in your view to handle multiple uploads)."),
        required=False,
        widget=forms.ClearableFileInput(attrs={
            "accept": ".csv,.xlsx,.xls",
            "class": "form-control"
        })
    )
    
    # VARIABLE channel support
    target_channels = forms.CharField(
        label=_("Focus on Specific Channels (Optional)"),
        max_length=1000,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter channel names separated by commas'
        }),
        help_text=_("Leave empty to process all channels")
    )
    
    # Smart processing options
    auto_ticket_generation = forms.BooleanField(
        label=_("Generate Tickets Automatically"),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Automatically generate tickets for Video Start Failures using MVP diagnosis")
    )
    
    advanced_cleaning = forms.BooleanField(
        label=_("Advanced Data Cleaning"),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Remove blank data, instructions, and apply flexible column mapping")
    )

    def clean(self):
        """Smart form validation"""
        cleaned_data = super().clean()
        
        # Validate primary file
        primary_file = cleaned_data.get('primary_file')
        if not primary_file:
            raise ValidationError(_("Please upload a data file."))
        
        # Process target channels - VARIABLE SUPPORT
        target_channels_str = cleaned_data.get('target_channels', '')
        if target_channels_str:
            channels = [ch.strip() for ch in target_channels_str.split(',') if ch.strip()]
            cleaned_data['target_channels_list'] = channels
        else:
            cleaned_data['target_channels_list'] = []
        
        # Validate primary file content
        try:
            filename = getattr(primary_file, 'name', 'uploaded_file')
            file_extension = Path(filename).suffix.lower()
            
            if file_extension not in ['.csv', '.xlsx', '.xls']:
                raise ValidationError(
                    _("Unsupported file format: %(ext)s. Please use CSV or Excel files.") % 
                    {'ext': file_extension}
                )
            
            # Basic content validation
            temp_path = self._create_temp_file_basic(primary_file, file_extension)
            
            try:
                # Quick validation
                if file_extension in ['.xlsx', '.xls']:
                    df_sample = pd.read_excel(temp_path, nrows=5)
                else:
                    df_sample = pd.read_csv(temp_path, nrows=5)
                
                if df_sample.empty:
                    raise ValidationError(_("File appears to be empty or unreadable."))
                
                if len(df_sample.columns) == 0:
                    raise ValidationError(_("File contains no data columns."))
                    
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(_("Error validating file: %(error)s") % {'error': str(e)})
        
        return cleaned_data

    def _create_temp_file_basic(self, file_obj, extension):
        """Basic temp file creation for smart form"""
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(suffix=extension)
        
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                
                if hasattr(file_obj, 'chunks'):
                    for chunk in file_obj.chunks():
                        temp_file.write(chunk)
                else:
                    content = file_obj.read()
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    temp_file.write(content)
            return temp_path
        except Exception:
            os.close(temp_fd)
            raise

    def get_file_mapping(self):
        """Get file mapping for smart form - uses auto-detect"""
        if not hasattr(self, 'cleaned_data') or not self.cleaned_data:
            return {}
            
        mapping = {}
        
        # Primary file with auto-detect
        primary_file = self.cleaned_data.get('primary_file')
        if primary_file:
            mapping[primary_file] = ['auto_detect']
        
        # Additional files if supported in future
        additional_files = self.cleaned_data.get('additional_files')
        if additional_files:
            mapping[additional_files] = ['auto_detect']
            
        return mapping

    def get_processing_config(self):
        """Get processing configuration for smart form"""
        if not hasattr(self, 'cleaned_data') or not self.cleaned_data:
            return {}
            
        return {
            'auto_detect_types': True,  # Key difference from flexible form
            'target_channels': self.cleaned_data.get('target_channels_list', []),  # VARIABLE channels
            'generate_tickets': self.cleaned_data.get('auto_ticket_generation', True),
            'session_only_tickets': True,  # Always use session-only for smart form
            'skip_transient': True,
            'flexible_column_mapping': self.cleaned_data.get('advanced_cleaning', True),
            'remove_blank_data': self.cleaned_data.get('advanced_cleaning', True),
            'clean_instructions': self.cleaned_data.get('advanced_cleaning', True),
        }

# ============================================================================
# UTILITY FUNCTIONS - PREVENT DUPLICACY
# ============================================================================

def get_form_by_preference(preference='flexible'):
    """Get form class based on user preference - maintains existing interface"""
    if preference == 'smart':
        return SmartUploadForm
    else:
        return FlexibleUploadForm  # Default to flexible form

def validate_uploaded_file_enhanced(file_obj):
    """Enhanced standalone file validation utility"""
    try:
        form = FlexibleUploadForm()
        return form._validate_file_content_enhanced(file_obj, ['auto_detect'], 'Standalone File')
    except Exception as e:
        return {
            'is_valid': False, 
            'error': f"Validation error: {str(e)}"
        }

def parse_target_channels(channels_str: str) -> List[str]:
    """Parse comma-separated channel string to list - VARIABLE channel support"""
    if not channels_str:
        return []
    
    # Flexible parsing with cleanup
    channels = []
    for channel in channels_str.split(','):
        channel = channel.strip()
        if channel:
            channels.append(channel)
    
    return channels

def detect_file_data_types(file_obj) -> dict:
    """Detect data types in uploaded file"""
    try:
        # Create temporary file
        filename = getattr(file_obj, 'name', 'test_file.csv')
        file_extension = Path(filename).suffix.lower()
        
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_extension)
        
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                
                if hasattr(file_obj, 'chunks'):
                    for chunk in file_obj.chunks():
                        temp_file.write(chunk)
                else:
                    content = file_obj.read()
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    temp_file.write(content)
            
            # Read and analyze
            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(temp_path, nrows=100)
            else:
                df = pd.read_csv(temp_path, nrows=100)
            
            # Use existing structure detection if available
            try:
                structure_info = detect_file_structure(df)
                return structure_info
            except:
                # Fallback detection
                return {
                    'detected_types': {'sessions': 0.5},  # Default assumption
                    'column_analysis': {
                        'total_columns': len(df.columns),
                        'data_columns': df.columns.tolist()
                    }
                }
                
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error detecting file data types: {e}")
        return {'error': str(e)}

# Backward compatibility aliases
UploadFilesForm = FlexibleUploadForm  # Alias for existing references
