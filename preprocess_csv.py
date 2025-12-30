import pandas as pd
import numpy as np

def preprocess_and_save_csv(input_file, output_file):
    """
    Preprocess the eye-tracking dataset and save to a new CSV file
    """
    # Load the CSV file
    print(f"Loading data from '{input_file}'...")
    df = pd.read_csv(input_file, delimiter=';')
    
    # Create a copy to store the preprocessed data
    df_processed = df.copy()
    
    # Clean column names
    df_processed.columns = df_processed.columns.str.strip()
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Cleaned columns: {df_processed.columns.tolist()}")
    
    # Identify key columns
    gender_col = None
    group_col = None
    
    for col in df_processed.columns:
        if 'gender' in col.lower():
            gender_col = col
        elif 'group' in col.lower():
            group_col = col
    
    print(f"\nIdentified columns:")
    print(f"  Gender column: '{gender_col}'")
    print(f"  Group column: '{group_col}'")
    
    if gender_col:
        print(f"\nPreprocessing '{gender_col}' column...")
        print(f"  Original values: {df_processed[gender_col].unique()}")
        
        # Strip whitespace and standardize gender values
        df_processed[gender_col] = df_processed[gender_col].astype(str).str.strip()
        
        # Map gender values to standardized format
        gender_mapping = {
            'F': 'Female',
            'M': 'Male',
            'FEMALE': 'Female',
            'MALE': 'Male',
            'FEMALE ': 'Female',  # Handle trailing spaces
            'MALE ': 'Male',      # Handle trailing spaces
            ' F': 'Female',       # Handle leading spaces
            ' M': 'Male'          # Handle leading spaces
        }
        
        # Apply mapping, keep original if not in mapping
        df_processed[gender_col] = df_processed[gender_col].apply(
            lambda x: gender_mapping.get(x.upper(), x)
        )
        
        print(f"  Processed values: {df_processed[gender_col].unique()}")
    
    # Process numeric columns
    print(f"\nProcessing numeric columns...")
    
    # Define which columns should be numeric (excluding gender and group)
    numeric_cols = []
    for col in df_processed.columns:
        if col not in [gender_col, group_col]:
            # Check if column looks numeric
            try:
                sample = df_processed[col].dropna().iloc[0]
                float(sample)
                numeric_cols.append(col)
            except:
                # Try to convert if it's a string with numbers
                if df_processed[col].dtype == 'object':
                    # Check if most values can be converted to numeric
                    try:
                        pd.to_numeric(df_processed[col], errors='coerce')
                        numeric_cols.append(col)
                    except:
                        pass
    
    print(f"  Numeric columns identified: {numeric_cols}")
    
    # Convert numeric columns
    for col in numeric_cols:
        print(f"  Processing '{col}'...")
        
        # Save original dtype for reference
        original_dtype = df_processed[col].dtype
        
        # Convert to string, replace commas with dots for decimals
        df_processed[col] = df_processed[col].astype(str)
        df_processed[col] = df_processed[col].str.replace(',', '.')
        
        # Convert to numeric, coerce errors to NaN
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Replace -1 with NaN (missing values)
        df_processed[col] = df_processed[col].replace(-1, np.nan)
        
        # Count missing values
        missing_count = df_processed[col].isna().sum()
        if missing_count > 0:
            # Fill missing values with column median
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
            print(f"    Filled {missing_count} missing values with median: {median_val:.4f}")
        
        # Report statistics
        print(f"    Min: {df_processed[col].min():.4f}, Max: {df_processed[col].max():.4f}, Mean: {df_processed[col].mean():.4f}")
    
    # Create a summary DataFrame with statistics
    print(f"\nCreating summary statistics...")
    summary_data = []
    
    for col in df_processed.columns:
        if df_processed[col].dtype in [np.float64, np.int64]:
            summary_data.append({
                'Column': col,
                'Type': 'Numeric',
                'Missing': df_processed[col].isna().sum(),
                'Min': df_processed[col].min(),
                'Max': df_processed[col].max(),
                'Mean': df_processed[col].mean(),
                'Median': df_processed[col].median(),
                'Std': df_processed[col].std()
            })
        else:
            summary_data.append({
                'Column': col,
                'Type': 'Categorical',
                'Missing': df_processed[col].isna().sum(),
                'Unique Values': len(df_processed[col].unique()),
                'Most Common': df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 'N/A'
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save the processed data
    print(f"\nSaving processed data to '{output_file}'...")
    df_processed.to_csv(output_file, index=False, sep=';')
    
    # Save summary statistics
    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nProcessing complete!")
    print(f"  Original shape: {df.shape}")
    print(f"  Processed shape: {df_processed.shape}")
    print(f"  Processed file saved as: {output_file}")
    print(f"  Summary statistics saved as: {summary_file}")
    
    # Display sample of processed data
    print(f"\nSample of processed data (first 5 rows):")
    print(df_processed.head())
    
    return df_processed, summary_df

def create_sample_preview(original_df, processed_df, output_file='preprocessing_comparison.csv'):
    """
    Create a comparison file showing original vs processed values
    """
    comparison_data = []
    
    # Take first 20 rows for comparison
    sample_size = min(20, len(original_df))
    
    for i in range(sample_size):
        row_data = {'Row': i+1}
        
        # Add original and processed values for each column
        for col in original_df.columns:
            original_val = original_df.iloc[i][col]
            processed_val = processed_df.iloc[i][col] if col in processed_df.columns else 'N/A'
            
            # Check if values differ
            changed = 'YES' if str(original_val) != str(processed_val) else 'NO'
            
            row_data[f'{col}_Original'] = original_val
            row_data[f'{col}_Processed'] = processed_val
            row_data[f'{col}_Changed'] = changed
        
        comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_file, index=False)
    print(f"\nComparison file saved as: {output_file}")
    
    return comparison_df

# Main execution
if __name__ == "__main__":
    input_csv = 'Web_browse_all_for_Py.csv'
    output_csv = 'Web_browse_all_for_Py_PREPROCESSED.csv'
    
    try:
        print("="*70)
        print("CSV PREPROCESSING TOOL")
        print("="*70)
        
        # Load original data first for comparison
        original_df = pd.read_csv(input_csv, delimiter=';')
        
        # Process the data
        processed_df, summary_df = preprocess_and_save_csv(input_csv, output_csv)
        
        # Create comparison file
        comparison_df = create_sample_preview(original_df, processed_df)
        
        print("\n" + "="*70)
        print("FILES CREATED:")
        print("="*70)
        print("1. PREPROCESSED_DATA.csv - The cleaned dataset ready for analysis")
        print("2. PREPROCESSED_DATA_summary.csv - Summary statistics of all columns")
        print("3. preprocessing_comparison.csv - Shows original vs processed values")
        
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY")
        print("="*70)
        print("The following transformations were applied:")
        print("1. Column names were cleaned (stripped of whitespace)")
        print("2. Gender column was standardized (F→Female, M→Male)")
        print("3. All numeric columns were converted to proper numeric types")
        print("4. Decimal commas were replaced with dots (if present)")
        print("5. Missing values (-1) were replaced with column medians")
        print("6. All data was preserved with the same structure")
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find file '{input_csv}'")
        print("Please make sure the file is in the current directory.")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}")
        print(f"Details: {e}")