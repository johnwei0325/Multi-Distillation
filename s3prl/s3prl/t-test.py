import numpy as np
from scipy import stats
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
import csv
from sklearn.preprocessing import LabelEncoder

KS_CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "_unknown_",
    "_silence_",
]

VOCAL_TECHNIQUE_CLASSES = [
    "belt",
    "breathy",
    "inhaled",
    "lip_trill",
    "spoken",
    "straight",
    "trill",
    "trillo",
    "vibrato",
    "vocal_fry",
]

SINGER_CLASSES = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m6",
    "m7",
    "m8",
    "m9",
    "m10",
    "m11",
]

def load_prediction_file(file_path, task_type='ks'):
    """Load prediction file and return a dictionary of filename: prediction pairs"""
    predictions = {}
    if task_type == 'ks':
        with open(file_path, 'r') as f:
            for line in f:
                filename, pred = line.strip().split()
                class_to_number = {cls: idx for idx, cls in enumerate(KS_CLASSES)}
                predictions[filename] = class_to_number[pred]  # Convert prediction to integer
    elif task_type == 'vid':
        with open(file_path, 'r') as f:
            for line in f:
                filename, pred = line.strip().split()
                class_to_number = {cls: idx for idx, cls in enumerate(VOCAL_TECHNIQUE_CLASSES)}
                predictions[filename] = class_to_number[pred]  # Convert prediction to integer
    elif task_type == 'singer':
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:  # Ensure we have at least filename and prediction
                    # Last part is the prediction, everything else is the filename
                    pred = parts[-1]
                    filename = ' '.join(parts[:-1])
                    class_to_number = {cls: idx for idx, cls in enumerate(SINGER_CLASSES)}
                    predictions[filename] = class_to_number[pred]  # Convert prediction to integer
    elif task_type == 'ic':
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) >= 4:  # Ensure row has filename, action, object, and location
                    filename = row[0]
                    predictions[filename] = {
                        'action': row[1],
                        'object': row[2],
                        'location': row[3]
                    }
    return predictions

def calculate_accuracy(pred1, pred2, task_type='ks'):
    """Calculate accuracy between two predictions"""
    if task_type == 'ks':
        return np.mean(np.array(pred1) == np.array(pred2))
    elif task_type == 'ic':
        action_acc = np.mean([p1['action'] == p2['action'] for p1, p2 in zip(pred1, pred2)])
        object_acc = np.mean([p1['object'] == p2['object'] for p1, p2 in zip(pred1, pred2)])
        location_acc = np.mean([p1['location'] == p2['location'] for p1, p2 in zip(pred1, pred2)])
        return {
            'action': action_acc,
            'object': object_acc,
            'location': location_acc,
            'overall': np.mean([action_acc, object_acc, location_acc])
        }

def compare_predictions(file1_path, file2_path, task_type='ks'):
    # Load both prediction files
    pred1 = load_prediction_file(file1_path, task_type)
    pred2 = load_prediction_file(file2_path, task_type)
    
    # Get common filenames
    common_files = set(pred1.keys()) & set(pred2.keys())
    
    if task_type == 'ks' or task_type == 'vid' or task_type == 'singer':
        # Extract predictions for common files
        pred1_values = [pred1[f] for f in common_files]
        pred2_values = [pred2[f] for f in common_files]
        
        # Convert to numpy arrays
        pred1_array = np.array(pred1_values)
        pred2_array = np.array(pred2_values)
        
        # Calculate differences
        differences = pred1_array - pred2_array
        
        # Check if predictions are identical
        if np.all(differences == 0):
            print("\nWARNING: The predictions in both files are identical!")
            t_stat = np.nan
            p_value = np.nan
            cohens_d = 0.0
            ci = (0.0, 0.0)
        else:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(pred1_array, pred2_array, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            cohens_d = np.mean(differences) / np.sqrt((np.std(pred1_array)**2 + np.std(pred2_array)**2) / 2)
            
            # Calculate confidence interval
            n = len(pred1_array)
            dof = n - 1
            ci = stats.t.interval(confidence=0.95, df=dof, loc=np.mean(differences), scale=stats.sem(differences))
        
        # Calculate additional statistics
        mean1 = np.mean(pred1_array)
        mean2 = np.mean(pred2_array)
        std1 = np.std(pred1_array)
        std2 = np.std(pred2_array)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        # Calculate correlation
        correlation = np.corrcoef(pred1_array, pred2_array)[0,1]
        
        # Calculate accuracy
        accuracy = calculate_accuracy(pred1_values, pred2_values, task_type)
        
        # Create a detailed summary DataFrame
        summary = pd.DataFrame({
            'File1': [mean1, std1, len(pred1_array), np.min(pred1_array), np.max(pred1_array)],
            'File2': [mean2, std2, len(pred2_array), np.min(pred2_array), np.max(pred2_array)],
            'Difference': [mean_diff, std_diff, len(differences), np.min(differences), np.max(differences)]
        }, index=['Mean', 'Std Dev', 'Sample Size', 'Min', 'Max'])
        
        print("\nDetailed Prediction Comparison Summary:")
        print(summary)
        # print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nT-test Results:")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Degrees of freedom: {len(pred1_array)-1}")
        print(f"\nEffect Size:")
        print(f"Cohen's d: {cohens_d:.4f}")
        print(f"\nConfidence Interval (95%):")
        print(f"Lower bound: {ci[0]:.4f}")
        print(f"Upper bound: {ci[1]:.4f}")
        print(f"\nCorrelation between predictions: {correlation:.4f}")
        
        return t_stat, p_value, summary, cohens_d, ci, correlation, accuracy, len(pred1_array)
        
    elif task_type == 'ic':
        # Extract predictions for common files
        pred1_values = [pred1[f] for f in common_files]
        pred2_values = [pred2[f] for f in common_files]
        
        # Calculate accuracies for each component
        accuracies = calculate_accuracy(pred1_values, pred2_values, task_type)
        
        # Create confusion matrices for each component
        confusion_matrices = {
            'action': defaultdict(lambda: defaultdict(int)),
            'object': defaultdict(lambda: defaultdict(int)),
            'location': defaultdict(lambda: defaultdict(int))
        }
        
        # Create label encoders for each component
        label_encoders = {
            'action': LabelEncoder(),
            'object': LabelEncoder(),
            'location': LabelEncoder()
        }
        
        # Convert categorical predictions to numerical values for t-test
        all_pred1_numerical = []
        all_pred2_numerical = []
        
        for component in ['action', 'object', 'location']:
            # Get all unique labels for this component
            all_labels = set()
            for pred in pred1_values + pred2_values:
                all_labels.add(pred[component])
            
            # Fit label encoder
            label_encoders[component].fit(list(all_labels))
            
            # Transform predictions to numerical values
            pred1_numerical = label_encoders[component].transform([p[component] for p in pred1_values])
            pred2_numerical = label_encoders[component].transform([p[component] for p in pred2_values])
            
            # Append to overall arrays
            all_pred1_numerical.extend(pred1_numerical)
            all_pred2_numerical.extend(pred2_numerical)
            
            # Update confusion matrix
            for p1, p2 in zip(pred1_values, pred2_values):
                confusion_matrices[component][p1[component]][p2[component]] += 1
        
        # Convert to numpy arrays
        all_pred1_numerical = np.array(all_pred1_numerical)
        all_pred2_numerical = np.array(all_pred2_numerical)
        
        # Calculate differences
        differences = all_pred1_numerical - all_pred2_numerical
        
        # Check if predictions are identical
        if np.all(differences == 0):
            print("\nWARNING: The predictions in both files are identical!")
            t_stat = np.nan
            p_value = np.nan
            cohens_d = 0.0
            ci = (0.0, 0.0)
        else:
            # Perform t-test
            t_stat, p_value = stats.ttest_rel(all_pred1_numerical, all_pred2_numerical)
            
            # Calculate effect size (Cohen's d)
            cohens_d = np.mean(differences) / np.sqrt((np.std(all_pred1_numerical)**2 + np.std(all_pred2_numerical)**2) / 2)
            
            # Calculate confidence interval
            n = len(all_pred1_numerical)
            dof = n - 1
            ci = stats.t.interval(confidence=0.95, df=dof, loc=np.mean(differences), scale=stats.sem(differences))
        
        print("\nIntent Classification Comparison Summary:")
        # print("\nAccuracies:")
        # for component, acc in accuracies.items():
        #     print(f"{component}: {acc:.4f}")
        
        print("\nOverall T-test Results:")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Cohen's d: {cohens_d:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"Confidence Interval (95%): ({ci[0]:.4f}, {ci[1]:.4f})")
        
        # print("\nConfusion Matrices:")
        # for component, matrix in confusion_matrices.items():
        #     print(f"\n{component.upper()} Confusion Matrix:")
        #     df = pd.DataFrame(matrix).fillna(0)
        #     print(df)
        
        return accuracies, confusion_matrices, {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'confidence_interval': ci,
            'degrees_of_freedom': dof
        }

def main():
    parser = argparse.ArgumentParser(description='Compare two prediction files using t-test')
    parser.add_argument('file1', type=str, help='Path to first prediction file')
    parser.add_argument('file2', type=str, help='Path to second prediction file')
    parser.add_argument('--output', type=str, help='Path to save the comparison results (optional)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default: 0.05)')
    parser.add_argument('--task', type=str, choices=['ks', 'ic', 'vid', 'singer'], default='ks',
                      help='Task type: ks (Keyword Spotting) or ic (Intent Classification) or vid (Vocal Technique Identification)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.file1).exists():
        raise FileNotFoundError(f"File not found: {args.file1}")
    if not Path(args.file2).exists():
        raise FileNotFoundError(f"File not found: {args.file2}")
    
    # Perform comparison
    if args.task == 'ks' or args.task == 'vid' or args.task == 'singer':
        t_stat, p_value, summary, cohens_d, ci, correlation, accuracy, sample_size = compare_predictions(
            args.file1, args.file2, args.task)
        
        # Save results if output path is provided
        if args.output:
            with open(args.output, 'w') as f:
                f.write("Detailed Prediction Comparison Summary:\n")
                f.write(str(summary))
                #f.write(f"\n\nAccuracy: {accuracy:.4f}")
                f.write(f"\n\nT-test Results:")
                f.write(f"\nt-statistic: {t_stat:.4f}")
                f.write(f"\np-value: {p_value:.4f}")
                f.write(f"\nDegrees of freedom: {sample_size-1}")
                f.write(f"\n\nEffect Size:")
                f.write(f"\nCohen's d: {cohens_d:.4f}")
                f.write(f"\n\nConfidence Interval (95%):")
                f.write(f"\nLower bound: {ci[0]:.4f}")
                f.write(f"\nUpper bound: {ci[1]:.4f}")
                f.write(f"\n\nCorrelation between predictions: {correlation:.4f}")
    else:  # IC task
        accuracies, confusion_matrices, t_test_results = compare_predictions(args.file1, args.file2, args.task)
        
        # Save results if output path is provided
        if args.output:
            with open(args.output, 'w') as f:
                f.write("Intent Classification Comparison Summary:\n")
                f.write("\nAccuracies:\n")
                for component, acc in accuracies.items():
                    f.write(f"{component}: {acc:.4f}\n")
                
                f.write("\nOverall T-test Results:\n")
                f.write(f"t-statistic: {t_test_results['t_statistic']:.4f}\n")
                f.write(f"p-value: {t_test_results['p_value']:.4f}\n")
                f.write(f"Cohen's d: {t_test_results['cohens_d']:.4f}\n")
                f.write(f"Degrees of freedom: {t_test_results['degrees_of_freedom']}\n")
                f.write(f"Confidence Interval (95%): ({t_test_results['confidence_interval'][0]:.4f}, {t_test_results['confidence_interval'][1]:.4f})\n")
                
                f.write("\nConfusion Matrices:\n")
                for component, matrix in confusion_matrices.items():
                    f.write(f"\n{component.upper()} Confusion Matrix:\n")
                    df = pd.DataFrame(matrix).fillna(0)
                    f.write(str(df))
                    f.write("\n")

if __name__ == "__main__":
    main() 
