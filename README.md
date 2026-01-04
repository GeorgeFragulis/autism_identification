# autism_identification
This is the public Repository of our paper entitled : Applying Machine Learning to Eye-Tracking Data for Autism Identification in High-Functioning Adults

In our study, we applied the following preprocessing steps using Python script.
                
        
        1. Strip whitespace and standardize gender values
        
        
        2. Map gender values to standardized format (e.g. 'F': 'Female','M': 'Male','FEMALE': 'Female','MALE': 'Male' etc. )
        
        
        3. All numeric columns were converted to proper numeric types
        
        
        4. Decimal commas were replaced with dots (if present) 
        
        
        5. Convert to numeric, coerce errors to NaN
        
        
        6. Fill missing values with column median
