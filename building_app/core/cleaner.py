import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Union
import re

logger = logging.getLogger(__name__)


class Cleaner:
    """
    A data cleaning class for CSV files and DataFrames.
    """
    
    def __init__(self):
        self.cleaning_log = []
    
    def clean_csv_to_df(self, 
                       file_path: str, 
                       remove_duplicates: bool = True,
                       handle_missing: str = 'drop',  # 'drop', 'fill', 'interpolate'
                       fill_value: Any = None,
                       normalize_columns: bool = True,
                       remove_special_chars: bool = True,
                       strip_whitespace: bool = True,
                       **kwargs) -> pd.DataFrame:
        """
        Clean a CSV file and convert it to a DataFrame.
        
        Args:
            file_path: Path to the CSV file
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: How to handle missing values ('drop', 'fill', 'interpolate')
            fill_value: Value to fill missing data with (if handle_missing='fill')
            normalize_columns: Whether to normalize column names
            remove_special_chars: Whether to remove special characters from column names
            strip_whitespace: Whether to strip whitespace from string columns
            **kwargs: Additional arguments to pass to pd.read_csv()
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Loading CSV file from: {file_path}")
            
            # Load the CSV file
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Loaded DataFrame with shape: {df.shape}")
            
            # Apply cleaning methods
            df = self._clean_dataframe(
                df=df,
                remove_duplicates=remove_duplicates,
                handle_missing=handle_missing,
                fill_value=fill_value,
                normalize_columns=normalize_columns,
                remove_special_chars=remove_special_chars,
                strip_whitespace=strip_whitespace
            )
            
            logger.info(f"Cleaned DataFrame with shape: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error cleaning CSV file: {e}")
            raise
    
    def _clean_dataframe(self, 
                        df: pd.DataFrame,
                        remove_duplicates: bool = True,
                        handle_missing: str = 'drop',
                        fill_value: Any = None,
                        normalize_columns: bool = True,
                        remove_special_chars: bool = True,
                        strip_whitespace: bool = True) -> pd.DataFrame:
        """
        Clean a DataFrame with various cleaning operations.
        
        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: How to handle missing values
            fill_value: Value to fill missing data with
            normalize_columns: Whether to normalize column names
            remove_special_chars: Whether to remove special characters
            strip_whitespace: Whether to strip whitespace
            
        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()
        
        # Log initial state
        initial_shape = df_cleaned.shape
        self.cleaning_log.append(f"Initial shape: {initial_shape}")
        
        # 1. Normalize column names
        if normalize_columns:
            df_cleaned = self._normalize_column_names(df_cleaned)
        
        # 2. Remove special characters from column names
        if remove_special_chars:
            df_cleaned = self._remove_special_chars_from_columns(df_cleaned)
        
        # 3. Strip whitespace from string columns
        if strip_whitespace:
            df_cleaned = self._strip_whitespace(df_cleaned)
        
        # 4. Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned, handle_missing, fill_value)
        
        # 5. Remove duplicates
        if remove_duplicates:
            df_cleaned = self._remove_duplicates(df_cleaned)
        
        # Log final state
        final_shape = df_cleaned.shape
        self.cleaning_log.append(f"Final shape: {final_shape}")
        
        return df_cleaned
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase with underscores."""
        df_cleaned = df.copy()
        
        # Convert to lowercase and replace spaces with underscores
        new_columns = {}
        for col in df_cleaned.columns:
            new_name = str(col).lower().replace(' ', '_').replace('-', '_')
            new_columns[col] = new_name
        
        df_cleaned = df_cleaned.rename(columns=new_columns)
        self.cleaning_log.append(f"Normalized column names: {list(new_columns.values())}")
        
        return df_cleaned
    
    def _remove_special_chars_from_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove special characters from column names."""
        df_cleaned = df.copy()
        
        new_columns = {}
        for col in df_cleaned.columns:
            # Remove special characters except underscores
            new_name = re.sub(r'[^a-zA-Z0-9_]', '', str(col))
            new_columns[col] = new_name
        
        df_cleaned = df_cleaned.rename(columns=new_columns)
        self.cleaning_log.append(f"Removed special chars from columns: {list(new_columns.values())}")
        
        return df_cleaned
    
    def _strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from string columns."""
        df_cleaned = df.copy()
        
        # Apply strip to string columns
        for col in df_cleaned.select_dtypes(include=['object']).columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
        
        self.cleaning_log.append("Stripped whitespace from string columns")
        return df_cleaned
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str, fill_value: Any) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        df_cleaned = df.copy()
        
        missing_count = df_cleaned.isnull().sum().sum()
        if missing_count == 0:
            self.cleaning_log.append("No missing values found")
            return df_cleaned
        
        self.cleaning_log.append(f"Found {missing_count} missing values")
        
        if method == 'drop':
            df_cleaned = df_cleaned.dropna()
            self.cleaning_log.append(f"Dropped rows with missing values")
        elif method == 'fill':
            if fill_value is not None:
                df_cleaned = df_cleaned.fillna(fill_value)
                self.cleaning_log.append(f"Filled missing values with: {fill_value}")
            else:
                # Fill with appropriate defaults based on column type
                for col in df_cleaned.columns:
                    if df_cleaned[col].dtype in ['int64', 'float64']:
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown')
                self.cleaning_log.append("Filled missing values with appropriate defaults")
        elif method == 'interpolate':
            df_cleaned = df_cleaned.interpolate()
            self.cleaning_log.append("Interpolated missing values")
        
        return df_cleaned
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the DataFrame."""
        df_cleaned = df.copy()
        
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        final_rows = len(df_cleaned)
        
        removed_count = initial_rows - final_rows
        if removed_count > 0:
            self.cleaning_log.append(f"Removed {removed_count} duplicate rows")
        else:
            self.cleaning_log.append("No duplicate rows found")
        
        return df_cleaned
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get a summary of the cleaning operations performed."""
        return {
            "cleaning_log": self.cleaning_log,
            "total_operations": len(self.cleaning_log)
        }
    
    def reset_cleaning_log(self):
        """Reset the cleaning log."""
        self.cleaning_log = []
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a DataFrame and return validation results.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict()
        }
        
        # Check for issues
        if df.empty:
            validation_results["is_valid"] = False
            validation_results["issues"].append("DataFrame is empty")
        
        if df.isnull().sum().sum() > 0:
            validation_results["issues"].append("Contains missing values")
        
        if df.duplicated().sum() > 0:
            validation_results["issues"].append("Contains duplicate rows")
        
        # Check for columns with all null values
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            validation_results["issues"].append(f"Columns with all null values: {null_columns}")
        
        return validation_results