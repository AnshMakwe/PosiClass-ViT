import os
import glob
from ultralytics import YOLO
import numpy as np
from pathlib import Path

def predict_probabilities_limuc_uncertain():
    """
    Predict probabilities for each Mayo class and only show uncertain cases
    (where difference between highest and second highest probability is = 0.1)
    """
    
    # Load the trained YOLO model
    model_path = "/home/user/Ansh/labeled-images-for-ulcerative-colitis/yolo/ultralytics/test_yolo11_project/test_architecture27/weights/best.pt"
    model = YOLO(model_path)
    
    # Path to validation dataset
    val_dataset_path = "/home/user/Ansh/labeled-images-for-ulcerative-colitis/yolo/dataset/val/"
    
    # Mayo score classes
    mayo_classes = ["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"]
    
    print("LIMUC Dataset - Uncertain Predictions (Probability Difference = 0.1)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Validation Dataset: {val_dataset_path}")
    print(f"Classes: {mayo_classes}")
    print("=" * 70)
    
    # Statistics tracking
    total_images = 0
    uncertain_cases = 0
    correct_predictions = 0
    uncertain_correct = 0
    class_stats = {i: {"total": 0, "correct": 0, "uncertain": 0, "uncertain_correct": 0} for i in range(4)}
    
    # Iterate through each Mayo class folder
    for true_class_idx, mayo_class in enumerate(mayo_classes):
        folder_path = os.path.join(val_dataset_path, mayo_class)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist!")
            continue
        
        print(f"\n{'-' * 50}")
        print(f"Processing {mayo_class} images:")
        print(f"{'-' * 50}")
        
        # Get all image files in the folder
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.JPG", "*.JPEG", "*.PNG"]
        image_files = []
        
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        
        if not image_files:
            print(f"No images found in {folder_path}")
            continue
        
        print(f"Found {len(image_files)} images")
        uncertain_in_class = 0
        
        # Process each image in the current Mayo class folder
        for img_idx, image_path in enumerate(image_files, 1):
            try:
                # Run prediction
                results = model(image_path, verbose=False)
                
                # Extract probabilities
                if hasattr(results[0], 'probs') and results[0].probs is not None:
                    # Get probability scores
                    probabilities = results[0].probs.data.cpu().numpy()
                    predicted_class_idx = results[0].probs.top1
                    confidence = results[0].probs.top1conf.item()
                    
                    # Sort probabilities to get highest and second highest
                    sorted_probs = np.sort(probabilities)[::-1]  # Sort in descending order
                    highest_prob = sorted_probs[0]
                    second_highest_prob = sorted_probs[1]
                    prob_difference = highest_prob - second_highest_prob
                    
                    # Update statistics
                    total_images += 1
                    class_stats[true_class_idx]["total"] += 1
                    
                    if predicted_class_idx == true_class_idx:
                        correct_predictions += 1
                        class_stats[true_class_idx]["correct"] += 1
                    
                    # Only print if probability difference is = 0.1 (uncertain cases)
                    if prob_difference <= 0.1:
                        uncertain_cases += 1
                        uncertain_in_class += 1
                        class_stats[true_class_idx]["uncertain"] += 1
                        
                        if predicted_class_idx == true_class_idx:
                            uncertain_correct += 1
                            class_stats[true_class_idx]["uncertain_correct"] += 1
                        
                        # Get indices of highest and second highest probabilities
                        sorted_indices = np.argsort(probabilities)[::-1]
                        highest_class_idx = sorted_indices[0]
                        second_highest_class_idx = sorted_indices[1]
                        
                        # Print results for this uncertain image
                        image_name = os.path.basename(image_path)
                        print(f"\n?? UNCERTAIN Case {uncertain_in_class} - Image: {image_name}")
                        print(f"True Class: {mayo_class}")
                        print(f"Predicted: {mayo_classes[predicted_class_idx]} (confidence: {confidence:.4f})")
                        print(f"Probability Difference: {prob_difference:.4f}")
                        print(f"Top 2 Classes: {mayo_classes[highest_class_idx]} ({highest_prob:.4f}) vs {mayo_classes[second_highest_class_idx]} ({second_highest_prob:.4f})")
                        
                        # Print full probability distribution
                        print("All Class Probabilities:")
                        for class_idx, (class_name, prob) in enumerate(zip(mayo_classes, probabilities)):
                            marker = " ?" if class_idx == predicted_class_idx else ""
                            correct_marker = " [TRUE]" if class_idx == true_class_idx else ""
                            rank_marker = ""
                            if class_idx == highest_class_idx:
                                rank_marker = " [1st]"
                            elif class_idx == second_highest_class_idx:
                                rank_marker = " [2nd]"
                            print(f"  {class_name}: {prob:.4f}{marker}{correct_marker}{rank_marker}")
                    
                else:
                    print(f"No classification probabilities available for {os.path.basename(image_path)}")
                    
            except Exception as e:
                print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        
        print(f"Uncertain cases in {mayo_class}: {uncertain_in_class}")
    
    # Print overall statistics
    print(f"\n{'=' * 70}")
    print("OVERALL RESULTS - UNCERTAIN CASES ANALYSIS")
    print(f"{'=' * 70}")
    
    if total_images > 0:
        overall_accuracy = correct_predictions / total_images
        uncertain_rate = uncertain_cases / total_images
        
        print(f"Total Images Processed: {total_images}")
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_images})")
        print(f"Uncertain Cases (prob diff = 0.1): {uncertain_cases} ({uncertain_rate:.4f})")
        
        if uncertain_cases > 0:
            uncertain_accuracy = uncertain_correct / uncertain_cases
            print(f"Accuracy on Uncertain Cases: {uncertain_accuracy:.4f} ({uncertain_correct}/{uncertain_cases})")
        
        print(f"\nPer-Class Uncertain Cases Analysis:")
        for class_idx, mayo_class in enumerate(mayo_classes):
            if class_stats[class_idx]["total"] > 0:
                class_accuracy = class_stats[class_idx]["correct"] / class_stats[class_idx]["total"]
                uncertain_rate_class = class_stats[class_idx]["uncertain"] / class_stats[class_idx]["total"]
                
                print(f"\n  {mayo_class}:")
                print(f"    Total: {class_stats[class_idx]['total']}")
                print(f"    Overall Accuracy: {class_accuracy:.4f}")
                print(f"    Uncertain Cases: {class_stats[class_idx]['uncertain']} ({uncertain_rate_class:.4f})")
                
                if class_stats[class_idx]["uncertain"] > 0:
                    uncertain_class_accuracy = class_stats[class_idx]["uncertain_correct"] / class_stats[class_idx]["uncertain"]
                    print(f"    Uncertain Accuracy: {uncertain_class_accuracy:.4f}")
            else:
                print(f"  {mayo_class}: No images found")
    else:
        print("No images were processed successfully.")

def predict_single_image_uncertain(image_path, threshold=0.1):
    """
    Predict probabilities for a single image and show if it's uncertain
    """
    model_path = "/home/user/Ansh/labeled-images-for-ulcerative-colitis/yolo/ultralytics/test_yolo11_project/test_architecture27/weights/best.pt"
    model = YOLO(model_path)
    mayo_classes = ["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"]
    
    try:
        results = model(image_path, verbose=False)
        
        if hasattr(results[0], 'probs') and results[0].probs is not None:
            probabilities = results[0].probs.data.cpu().numpy()
            predicted_class_idx = results[0].probs.top1
            confidence = results[0].probs.top1conf.item()
            
            # Calculate probability difference
            sorted_probs = np.sort(probabilities)[::-1]
            prob_difference = sorted_probs[0] - sorted_probs[1]
            
            print(f"\nSingle Image Prediction")
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Predicted Class: {mayo_classes[predicted_class_idx]} (confidence: {confidence:.4f})")
            print(f"Probability Difference: {prob_difference:.4f}")
            
            if prob_difference <= threshold:
                print("?? This is an UNCERTAIN case!")
                
                # Get top 2 classes
                sorted_indices = np.argsort(probabilities)[::-1]
                print(f"Top 2 Classes: {mayo_classes[sorted_indices[0]]} ({sorted_probs[0]:.4f}) vs {mayo_classes[sorted_indices[1]]} ({sorted_probs[1]:.4f})")
                
                print("Class Probabilities:")
                for class_idx, (class_name, prob) in enumerate(zip(mayo_classes, probabilities)):
                    marker = " ?" if class_idx == predicted_class_idx else ""
                    print(f"  {class_name}: {prob:.4f}{marker}")
            else:
                print("? This is a CONFIDENT prediction.")
            
            return probabilities, prob_difference <= threshold
        else:
            print("No classification probabilities available")
            return None, False
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, False

if __name__ == "__main__":
    # Run prediction on entire validation dataset (only uncertain cases)
    predict_probabilities_limuc_uncertain()
    
    # Example: Predict for a single image (uncomment to use)
    # single_image_path = "/path/to/your/image.jpg"
    # predict_single_image_uncertain(single_image_path)
