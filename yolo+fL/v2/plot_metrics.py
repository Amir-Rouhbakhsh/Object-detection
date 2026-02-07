import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
with open("fl_metrics/metrics.json", "r") as f:
    metrics_data = json.load(f)

client_metrics = metrics_data["client_metrics"]

# Plot 1: mAP50 progression
plt.figure(figsize=(10, 6))

for client_id in sorted(client_metrics.keys()):
    client_data = client_metrics[client_id]
    rounds = sorted([int(r) for r in client_data.keys()])
    rounds = sorted(rounds)
    
    mAP50_values = []
    for round_num in rounds:
        mAP50 = client_data[str(round_num)]['mAP50']
        if isinstance(mAP50, list):
            mAP50 = mAP50[-1] if mAP50 else 0
        mAP50_values.append(float(mAP50))
    
    plt.plot(rounds, mAP50_values, marker='o', linewidth=2, 
            markersize=8, label=f'Client {client_id}')

plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('mAP50', fontsize=12)
plt.title('Federated Learning Convergence: YOLO Object Detection', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("fl_metrics/mAP50_convergence.png", dpi=300, bbox_inches='tight')
plt.savefig("fl_metrics/mAP50_convergence.pdf", bbox_inches='tight')
print("Plot saved: fl_metrics/mAP50_convergence.png and .pdf")

# Plot 2: Client comparison bar chart
plt.figure(figsize=(8, 6))

client_ids = []
final_mAPs = []

for client_id in sorted(client_metrics.keys()):
    client_data = client_metrics[client_id]
    if client_data:
        last_round = max([int(r) for r in client_data.keys()])
        mAP50 = client_data[str(last_round)]['mAP50']
        if isinstance(mAP50, list):
            mAP50 = mAP50[-1] if mAP50 else 0
        
        client_ids.append(f'Client {client_id}')
        final_mAPs.append(float(mAP50))

bars = plt.bar(client_ids, final_mAPs, color=['#FF9A76', '#87CEEB', '#98FB98'][:len(client_ids)])
plt.xlabel('Client', fontsize=12)
plt.ylabel('Final mAP50', fontsize=12)
plt.title('Client Performance Comparison (Final Round)', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("fl_metrics/client_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("fl_metrics/client_comparison.pdf", bbox_inches='tight')
print("Plot saved: fl_metrics/client_comparison.png and .pdf")

# Create summary table
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for client_id in sorted(client_metrics.keys()):
    client_data = client_metrics[client_id]
    rounds = sorted([int(r) for r in client_data.keys()])
    
    if len(rounds) >= 2:
        first_mAP = client_data[str(rounds[0])]['mAP50']
        last_mAP = client_data[str(rounds[-1])]['mAP50']
        
        if isinstance(first_mAP, list):
            first_mAP = first_mAP[-1] if first_mAP else 0
        if isinstance(last_mAP, list):
            last_mAP = last_mAP[-1] if last_mAP else 0
        
        first_mAP = float(first_mAP)
        last_mAP = float(last_mAP)
        
        improvement = ((last_mAP - first_mAP) / first_mAP * 100) if first_mAP > 0 else 0
        
        print(f"Client {client_id}:")
        print(f"  Rounds: {len(rounds)}")
        print(f"  Initial mAP50: {first_mAP:.4f}")
        print(f"  Final mAP50: {last_mAP:.4f}")
        print(f"  Improvement: {improvement:.2f}%")
        print()

plt.show()