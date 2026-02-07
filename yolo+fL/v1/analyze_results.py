import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_saved_metrics(experiment_dir):
    """Analyze saved metrics from a completed experiment"""
    
    # Load metrics
    with open(f"{experiment_dir}/client_metrics.json", 'r') as f:
        client_metrics = json.load(f)
    
    with open(f"{experiment_dir}/server_metrics.json", 'r') as f:
        server_metrics = json.load(f)
    
    # Create comprehensive analysis
    analysis_results = {
        'experiment': experiment_dir,
        'total_rounds': len(server_metrics),
        'clients': len(client_metrics),
    }
    
    # Calculate overall improvements
    client_improvements = []
    for client_name, client_data in client_metrics.items():
        rounds = sorted(map(int, client_data.keys()))
        if len(rounds) >= 2:
            first_round = str(rounds[0])
            last_round = str(rounds[-1])
            
            initial_mAP = client_data[first_round]['metrics'].get('mAP50', 0)
            final_mAP = client_data[last_round]['metrics'].get('mAP50', 0)
            
            if isinstance(initial_mAP, list):
                initial_mAP = initial_mAP[-1] if initial_mAP else 0
            if isinstance(final_mAP, list):
                final_mAP = final_mAP[-1] if final_mAP else 0
            
            improvement = ((final_mAP - initial_mAP) / initial_mAP * 100) if initial_mAP > 0 else 0
            client_improvements.append(improvement)
    
    if client_improvements:
        analysis_results['avg_improvement'] = np.mean(client_improvements)
        analysis_results['std_improvement'] = np.std(client_improvements)
    
    # Print analysis
    print("\n" + "="*60)
    print("FEDERATED LEARNING EXPERIMENT ANALYSIS")
    print("="*60)
    print(f"Experiment: {analysis_results['experiment']}")
    print(f"Total Rounds: {analysis_results['total_rounds']}")
    print(f"Number of Clients: {analysis_results['clients']}")
    if 'avg_improvement' in analysis_results:
        print(f"Average mAP50 Improvement: {analysis_results['avg_improvement']:.2f}%")
        print(f"Improvement Std Dev: {analysis_results['std_improvement']:.2f}%")
    print("="*60)
    
    return analysis_results

# Usage
if __name__ == "__main__":
    # Point to your saved experiment directory
    experiment_dir = "fl_metrics_20260202_140801"  # Replace with your actual directory
    results = analyze_saved_metrics(experiment_dir)