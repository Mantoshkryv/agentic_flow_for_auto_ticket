from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
import logging
from pathlib import Path
import tempfile
import os
import pandas as pd

# Initialize logger
logger = logging.getLogger(__name__)

# Import data processing modules with proper error handling
try:
    from .data_processing import DataProcessor
    from .data_validation import DataValidator, ValidationResult
    PROCESSING_AVAILABLE = True
    logger.info("Data processing modules imported successfully")
except ImportError as e:
    # Graceful fallback if modules don't exist yet
    DataProcessor = None
    DataValidator = None
    ValidationResult = None
    PROCESSING_AVAILABLE = False
    logger.warning(f"Data processing modules not available: {e}")

class UploadFilesForm(forms.Form):
    """
    Flexible file upload form supporting 1-3 files with any data combination.
    FIXED: Properly validates files by reading content, not filename.
    """

    # File 1 (Required) - Primary data file
    file_1 = forms.FileField(
        label=_("File 1 (Required)"),
        widget=forms.FileInput(attrs={
            "accept": ".csv,.xlsx,.xls",
            "class": "form-control",
            "data-file-number": "1"
        }),
        help_text=_("Primary data file containing session, KPI, and/or metadata")
    )

    file_1_contains = forms.MultipleChoiceField(
        label=_("File 1 contains:"),
        choices=[
            ('sessions', _('Session Data')),
            ('kpi_data', _('KPI Data')),
            ('advancetags', _('Metadata'))
        ],
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input',
            'data-file-number': '1'
        }),
        help_text=_("Select all data types present in this file")
    )

    # File 2 (Optional) - Secondary data file
    file_2 = forms.FileField(
        label=_("File 2 (Optional)"),
        widget=forms.FileInput(attrs={
            "accept": ".csv,.xlsx,.xls",
            "class": "form-control",
            "data-file-number": "2"
        }),
        required=False,
        help_text=_("Additional data file (leave empty if not needed)")
    )

    file_2_contains = forms.MultipleChoiceField(
        label=_("File 2 contains:"),
        choices=[
            ('sessions', _('Session Data')),
            ('kpi_data', _('KPI Data')),
            ('advancetags', _('Metadata'))
        ],
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input',
            'data-file-number': '2'
        }),
        required=False,
        help_text=_("Select data types in File 2 (if uploaded)")
    )

    # File 3 (Optional) - Third data file
    file_3 = forms.FileField(
        label=_("File 3 (Optional)"),
        widget=forms.FileInput(attrs={
            "accept": ".csv,.xlsx,.xls",
            "class": "form-control",
            "data-file-number": "3"
        }),
        required=False,
        help_text=_("Third data file (leave empty if not needed)")
    )

    file_3_contains = forms.MultipleChoiceField(
        label=_("File 3 contains:"),
        choices=[
            ('sessions', _('Session Data')),
            ('kpi_data', _('KPI Data')),
            ('advancetags', _('Metadata'))
        ],
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input',
            'data-file-number': '3'
        }),
        required=False,
        help_text=_("Select data types in File 3 (if uploaded)")
    )

    def clean(self):
        """FIXED: Validate form data with proper file content reading."""
        cleaned_data = super().clean()

        # Get uploaded files
        file_1 = cleaned_data.get('file_1')
        file_2 = cleaned_data.get('file_2')
        file_3 = cleaned_data.get('file_3')

        # Get data type selections
        file_1_contains = cleaned_data.get('file_1_contains', [])
        file_2_contains = cleaned_data.get('file_2_contains', [])
        file_3_contains = cleaned_data.get('file_3_contains', [])

        # Validate File 1 (required)
        if not file_1:
            raise ValidationError(_("File 1 is required."))

        if not file_1_contains:
            raise ValidationError(_("Please specify what data types are in File 1."))

        # Validate File 2 if uploaded
        if file_2 and not file_2_contains:
            raise ValidationError(_("Please specify what data types are in File 2."))

        # Validate File 3 if uploaded  
        if file_3 and not file_3_contains:
            raise ValidationError(_("Please specify what data types are in File 3."))

        # FIXED: Validate files by reading their actual content
        files_to_validate = [
            (file_1, 'File 1'),
            (file_2, 'File 2'),
            (file_3, 'File 3')
        ]

        for file_obj, file_label in files_to_validate:
            if file_obj:
                try:
                    is_valid, error_message, column_info = validate_file_format_and_content(file_obj)
                    if not is_valid:
                        raise ValidationError(
                            _("%(file)s: %(error)s") % {
                                'file': file_label, 
                                'error': error_message
                            }
                        )
                    else:
                        # Log successful validation with column info
                        logger.info(f"{file_label} validation successful: {column_info}")

                except Exception as e:
                    logger.error(f"Error validating {file_label}: {e}")
                    raise ValidationError(
                        _("Error validating %(file)s: %(error)s") % {
                            'file': file_label, 
                            'error': str(e)
                        }
                    )

        return cleaned_data

    def get_file_mapping(self):
        """
        Return dictionary mapping uploaded files to their data types.
        Used by views.py for processing uploaded files.

        Returns: 
            dict: {file_obj: ['sessions', 'kpi_data'], file_obj2: ['advancetags'], ...}
        """
        if not hasattr(self, 'cleaned_data') or not self.cleaned_data:
            return {}

        mapping = {}

        # File 1 (always present if form is valid)
        file_1 = self.cleaned_data.get('file_1')
        file_1_contains = self.cleaned_data.get('file_1_contains', [])
        if file_1 and file_1_contains:
            mapping[file_1] = file_1_contains

        # File 2 (optional)
        file_2 = self.cleaned_data.get('file_2')
        file_2_contains = self.cleaned_data.get('file_2_contains', [])
        if file_2 and file_2_contains:
            mapping[file_2] = file_2_contains

        # File 3 (optional)
        file_3 = self.cleaned_data.get('file_3')
        file_3_contains = self.cleaned_data.get('file_3_contains', [])
        if file_3 and file_3_contains:
            mapping[file_3] = file_3_contains

        return mapping

    def get_upload_summary(self):
        """
        Return a summary of what will be uploaded.
        Used for confirmation display in templates.
        """
        if not hasattr(self, 'cleaned_data') or not self.cleaned_data:
            return _("Form not validated")

        mapping = self.get_file_mapping()
        summary_parts = []

        for i, (file_obj, data_types) in enumerate(mapping.items(), 1):
            file_name = getattr(file_obj, 'name', f'File {i}')
            data_list = ', '.join(data_types)
            summary_parts.append(f"{file_name}: {data_list}")

        return " | ".join(summary_parts)


class AutoUploadConfigForm(forms.Form):
    """
    Form for configuring automatic data fetching from database collections.
    Aligned with Django models and DataProcessor capabilities.
    """

    # Enable/disable auto upload
    enabled = forms.BooleanField(
        label=_("Enable Auto Upload"),
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Automatically fetch and process data from database collections")
    )

    # Collection configuration - aligned with Django model names
    session_collection = forms.CharField(
        label=_("Session Collection Name"),
        max_length=100,
        initial="sessions",  # Matches Session model db_table
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        help_text=_("Database collection containing session data")
    )

    kpi_collection = forms.CharField(
        label=_("KPI Collection Name"), 
        max_length=100,
        initial="kpi_data",  # Matches KPI model db_table
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        help_text=_("Database collection containing KPI data")
    )

    metadata_collection = forms.CharField(
        label=_("Metadata Collection Name"),
        max_length=100,
        initial="advancetags",  # Matches Advancetags model db_table
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        help_text=_("Database collection containing metadata/advancetags data")
    )

    # Time filter options
    TIME_RANGE_CHOICES = [
        ('last_hour', _('Last Hour')),
        ('last_6_hours', _('Last 6 Hours')), 
        ('last_24_hours', _('Last 24 Hours')),
        ('last_week', _('Last Week')),
        ('custom', _('Custom Date Range')),
        ('all', _('All Data'))
    ]

    time_range = forms.ChoiceField(
        choices=TIME_RANGE_CHOICES,
        initial='last_hour',
        label=_("Time Range"),
        widget=forms.Select(attrs={'class': 'form-select'}),
        help_text=_("Filter data by time range")
    )

    # Custom date range fields
    start_date = forms.DateTimeField(
        label=_("Start Date"),
        required=False,
        widget=forms.DateTimeInput(attrs={
            'class': 'form-control', 
            'type': 'datetime-local'
        }),
        help_text=_("Start date for custom range (leave empty to use time range)")
    )

    end_date = forms.DateTimeField(
        label=_("End Date"), 
        required=False,
        widget=forms.DateTimeInput(attrs={
            'class': 'form-control', 
            'type': 'datetime-local'
        }),
        help_text=_("End date for custom range (leave empty for current time)")
    )

    # Processing options - aligned with ticket_engine capabilities
    auto_generate_tickets = forms.BooleanField(
        label=_("Auto Generate Tickets"),
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Automatically generate tickets using AutoTicketMVP engine")
    )

    auto_save_to_database = forms.BooleanField(
        label=_("Auto Save to Database"),
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text=_("Automatically save generated tickets to database")
    )

    def clean(self):
        """Validate auto upload configuration."""
        cleaned_data = super().clean()
        time_range = cleaned_data.get('time_range')
        start_date = cleaned_data.get('start_date')
        end_date = cleaned_data.get('end_date')

        # Validate custom time range
        if time_range == 'custom':
            if not start_date:
                raise ValidationError(_("Start date is required for custom time range."))
            if end_date and start_date and end_date <= start_date:
                raise ValidationError(_("End date must be after start date."))

        # Validate collection names
        collection_fields = ['session_collection', 'kpi_collection', 'metadata_collection']
        for field_name in collection_fields:
            collection_name = cleaned_data.get(field_name, '')
            if collection_name and not collection_name.replace('_', '').replace('-', '').isalnum():
                raise ValidationError(
                    _("%(field)s must contain only letters, numbers, underscores, and hyphens.") % {
                        'field': field_name.replace('_', ' ').title()
                    }
                )

        return cleaned_data

    def get_database_config(self):
        """
        Return database configuration dictionary.
        Used by views.py for automated data processing.
        """
        if not hasattr(self, 'cleaned_data') or not self.cleaned_data:
            return {}

        config = {
            'enabled': self.cleaned_data.get('enabled', False),
            'collections': {
                'session': self.cleaned_data.get('session_collection', 'sessions'),
                'kpi': self.cleaned_data.get('kpi_collection', 'kpi_data'),
                'metadata': self.cleaned_data.get('metadata_collection', 'advancetags')
            },
            'time_filter': {
                'range': self.cleaned_data.get('time_range', 'last_hour'),
                'start_date': self.cleaned_data.get('start_date'),
                'end_date': self.cleaned_data.get('end_date')
            },
            'processing': {
                'auto_generate_tickets': self.cleaned_data.get('auto_generate_tickets', True),
                'auto_save_to_database': self.cleaned_data.get('auto_save_to_database', True)
            }
        }

        return config


# FIXED: Utility functions for file validation - reads actual file content, not filename
def validate_file_format_and_content(file_obj):
    """
    FIXED: Validate uploaded file by reading its actual content and columns.
    Returns (is_valid, error_message, column_info)
    """
    if not file_obj:
        return False, _("No file provided"), ""

    filename = getattr(file_obj, 'name', 'uploaded_file')

    # Check file extension first
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    file_extension = Path(filename).suffix.lower()

    if file_extension not in allowed_extensions:
        return False, _("Unsupported file format. Please upload CSV or Excel files only."), ""

    try:
        # FIXED: Read file content properly for validation
        df = None
        temp_file_path = None

        # Handle different file upload scenarios
        if hasattr(file_obj, 'temporary_file_path'):
            # Large uploaded file stored in temp directory
            temp_file_path = file_obj.temporary_file_path()

        elif hasattr(file_obj, 'read'):
            # Small uploaded file in memory - need to save to temp file
            file_ext = os.path.splitext(filename)[1].lower()

            # Create temporary file
            temp_fd, temp_file_path = tempfile.mkstemp(suffix=file_ext)
            try:
                with os.fdopen(temp_fd, 'wb') as temp_file:
                    # Reset file pointer to beginning
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

                logger.info(f"Created temporary file for validation: {temp_file_path}")

            except Exception:
                os.close(temp_fd)
                raise
        else:
            # Regular file path
            temp_file_path = str(file_obj)

        # Verify file exists and is not empty
        if not os.path.exists(temp_file_path):
            return False, f"Temporary file not created: {filename}", ""

        file_size = os.path.getsize(temp_file_path)
        if file_size == 0:
            return False, f"File is empty: {filename} (0 bytes)", ""

        # Read the file based on its extension
        try:
            if file_extension == '.csv':
                # Try different encodings and separators for CSV
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(temp_file_path, encoding=encoding, nrows=5)  # Read only first 5 rows for validation
                        if not df.empty and len(df.columns) > 0:
                            logger.info(f"Successfully read CSV with {encoding} encoding")
                            break
                    except (UnicodeDecodeError, pd.errors.EmptyDataError):
                        continue

                # If still no success, try different separators
                if df is None or df.empty:
                    for sep in [',', ';', '\t', '|']:
                        try:
                            df = pd.read_csv(temp_file_path, sep=sep, encoding='utf-8', nrows=5)
                            if not df.empty and len(df.columns) > 1:
                                logger.info(f"Successfully read CSV with separator '{sep}'")
                                break
                        except:
                            continue

            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(temp_file_path, nrows=5)  # Read only first 5 rows for validation

        except Exception as e:
            return False, f"Could not read file: {str(e)}", ""

        # Clean up temporary file if we created one
        if hasattr(file_obj, 'read') and not hasattr(file_obj, 'temporary_file_path'):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary validation file: {temp_file_path}")
            except:
                pass

        # Validate DataFrame content
        if df is None or df.empty:
            return False, "File contains no readable data", ""

        if len(df.columns) == 0:
            return False, "File contains no columns", ""

        # Success! Return column information
        column_info = f"{len(df)} rows, {len(df.columns)} columns: {list(df.columns)[:5]}"
        if len(df.columns) > 5:
            column_info += "..."

        logger.info(f"File validation successful: {filename} - {column_info}")
        return True, "File format is valid", column_info

    except Exception as e:
        logger.error(f"File validation error for {filename}: {e}")
        return False, f"Error reading file: {str(e)}", ""


def get_file_data_types(file_obj):
    """
    Auto-detect data types in uploaded file by reading its columns.
    Returns list matching form choice values: ['session', 'kpi', 'metadata']
    """
    if not PROCESSING_AVAILABLE or not DataProcessor:
        logger.warning("DataProcessor not available for data type detection")
        return []

    try:
        processor = DataProcessor()
        filename = getattr(file_obj, 'name', 'uploaded_file')

        # Use existing method from data_processing.py
        file_results = processor.intelligently_process_any_file(file_obj, filename)

        detected_types = []
        for data_type, df in file_results.items():
            if not df.empty:
                # Map internal names to form choice values
                if data_type == 'meta':
                    detected_types.append('metadata')
                elif data_type in ['session', 'kpi']:
                    detected_types.append(data_type)

        logger.info(f"Detected data types in {filename}: {detected_types}")
        return detected_types

    except Exception as e:
        logger.error(f"Error detecting data types in file: {e}")
        return []


def validate_data_compatibility(session_df, kpi_df, metadata_df):
    """
    Validate data compatibility for ticket generation.
    Ensures required fields are present for AutoTicketMVP engine.
    """
    if not PROCESSING_AVAILABLE or not DataValidator:
        return True, []

    warnings = []

    try:
        validator = DataValidator()

        # Validate session data for ticket generation
        if not session_df.empty:
            session_result = validator.validate_dataframe(session_df, "session")
            if not session_result.is_valid:
                warnings.extend([f"Session data: {error}" for error in session_result.errors])

        # Validate KPI data
        if not kpi_df.empty:
            kpi_result = validator.validate_dataframe(kpi_df, "kpi")
            if not kpi_result.is_valid:
                warnings.extend([f"KPI data: {error}" for error in kpi_result.errors])

        # Validate metadata
        if not metadata_df.empty:
            meta_result = validator.validate_dataframe(metadata_df, "meta")
            if not meta_result.is_valid:
                warnings.extend([f"Metadata: {error}" for error in meta_result.errors])

        # Check for critical fields needed by ticket_engine.py
        critical_session_fields = ['session_id', 'ended_status', 'video_start_failure']
        if not session_df.empty:
            missing_critical = [field for field in critical_session_fields if field not in session_df.columns]
            if missing_critical:
                warnings.append(f"Missing critical session fields for ticket generation: {missing_critical}")

        # Check for correlation fields needed by AutoTicketMVP
        correlation_fields = ['isp_name', 'cdn', 'city']
        if not metadata_df.empty:
            missing_correlation = [field for field in correlation_fields if field not in metadata_df.columns]
            if missing_correlation:
                warnings.append(f"Missing correlation fields may limit ticket diagnosis: {missing_correlation}")

        is_compatible = len(warnings) == 0
        return is_compatible, warnings

    except Exception as e:
        logger.error(f"Error validating data compatibility: {e}")
        return False, [f"Validation error: {str(e)}"]
