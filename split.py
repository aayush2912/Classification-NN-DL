import pandas as pd
import os
from collections import Counter
import shutil
from sklearn.model_selection import train_test_split

def create_directory_structure(base_path, classes):
    """
    Create the directory structure for train, validation, and test sets
    """
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        os.makedirs(split_path, exist_ok=True)
        
        # Create class subdirectories
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            os.makedirs(class_path, exist_ok=True)


def analyze_dataset(csv_path):
    """
    Analyze the dataset and return top 10 classes
    """
    df = pd.read_csv(csv_path)
    
    # Filter out kids' clothes
    df_adult = df[~df['kids']]
    
    # Get class distribution
    class_distribution = df_adult['label'].value_counts()
    
    # Get top 10 classes
    top_10_classes = class_distribution.head(10)
    
    print("\nDataset Analysis:")
    print(f"Total number of samples: {len(df)}")
    print(f"Adult clothes samples: {len(df_adult)}")
    print(f"Number of unique classes: {df_adult['label'].nunique()}")
    
    print("\nTop 10 Classes Distribution:")
    for class_name, count in top_10_classes.items():
        print(f"{class_name}: {count} items")
    
    return list(top_10_classes.index)


def split_dataset(csv_path, top_10_classes):
    """
    Split the dataset into train, validation, and test sets
    """
    df = pd.read_csv(csv_path)
    
    # Filter out kids' clothes and keep only top 10 classes
    df_filtered = df[~df['kids'] & df['label'].isin(top_10_classes)]
    
    # First split: 80% train, 20% remaining
    train_df, remaining_df = train_test_split(
        df_filtered, 
        train_size=0.8,  # 80% for training
        stratify=df_filtered['label'], 
        random_state=42
    )
    
    # Second split: split the remaining 20% into validation and test (75-25 split)
    val_df, test_df = train_test_split(
        remaining_df, 
        train_size=0.75,  # 75% of remaining (15% of total) for validation
        stratify=remaining_df['label'], 
        random_state=42
    )
    
    print("\nDataset Split Sizes:")
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df_filtered)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df_filtered)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df_filtered)*100:.1f}%)")
    
    # Verify class distribution in each split
    print("\nClass distribution in splits:")
    for label in top_10_classes:
        train_count = len(train_df[train_df['label'] == label])
        val_count = len(val_df[val_df['label'] == label])
        test_count = len(test_df[test_df['label'] == label])
        total_count = train_count + val_count + test_count
        
        print(f"\n{label}:")
        print(f"Train: {train_count} ({train_count/total_count*100:.1f}%)")
        print(f"Validation: {val_count} ({val_count/total_count*100:.1f}%)")
        print(f"Test: {test_count} ({test_count/total_count*100:.1f}%)")
    
    return train_df, val_df, test_df


def main():
    # Set paths
    csv_path = 'clothing-dataset-full.csv'  
    base_path = 'data-full'  
    
    # Create base directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Analyze dataset and get top 10 classes
    top_10_classes = analyze_dataset(csv_path)
    
    # Create directory structure
    create_directory_structure(base_path, top_10_classes)
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(csv_path, top_10_classes)
    
    print("\nSetup complete! Directory structure has been created.")


if __name__ == "__main__":
    main()