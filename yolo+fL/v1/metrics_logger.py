import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd
import os

class FederatedMetricsLogger:
    def __init__(self, experiment_name="federated_yolo", base_save_path=None):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use custom base path or default to current directory
        if base_save_path is None:
            base_save_path = "."
        
        # Create the full save directory path
        self.save_dir = os.path.join(base_save_path, f"fl_metrics_{self.timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize data structures
        self.server_rounds = []
        self.client_metrics = {}
        self.server_metrics = {}
        self.training_times = []
        self.communication_times = []
        
        print(f"✓ Metrics logger initialized")
        print(f"  Save location: {self.save_dir}")
        print(f"  Experiment: {experiment_name}")
        
    def log_client_training(self, client_id: int, server_round: int, metrics: Dict):
        """Log client training metrics"""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = {}
        
        self.client_metrics[client_id][server_round] = {
            'timestamp': time.time(),
            'metrics': metrics.copy()
        }
        
        # Also add to server rounds if not already
        if server_round not in self.server_rounds:
            self.server_rounds.append(server_round)
        
        print(f"Logged metrics for Client {client_id}, Round {server_round}")
    
    def log_server_aggregation(self, server_round: int, metrics: Dict):
        """Log server aggregation metrics"""
        self.server_metrics[server_round] = {
            'timestamp': time.time(),
            'metrics': metrics.copy()
        }
        print(f"Logged server aggregation for Round {server_round}")
    
    def log_training_time(self, training_time: float):
        """Log training time"""
        self.training_times.append(training_time)
    
    def log_communication_time(self, communication_time: float):
        """Log communication time"""
        self.communication_times.append(communication_time)
    
    def save_metrics(self):
        """Save all metrics to JSON files"""
        # Save client metrics
        client_data = {}
        for client_id, rounds_data in self.client_metrics.items():
            client_data[f"client_{client_id}"] = {
                str(round_num): data for round_num, data in rounds_data.items()
            }
        
        with open(f"{self.save_dir}/client_metrics.json", 'w') as f:
            json.dump(client_data, f, indent=2)
        
        # Save server metrics
        server_data = {
            str(round_num): data for round_num, data in self.server_metrics.items()
        }
        
        with open(f"{self.save_dir}/server_metrics.json", 'w') as f:
            json.dump(server_data, f, indent=2)
        
        # Save timing data
        timing_data = {
            'training_times': self.training_times,
            'communication_times': self.communication_times,
            'total_rounds': len(self.server_rounds),
            'total_clients': len(self.client_metrics)
        }
        
        with open(f"{self.save_dir}/timing_metrics.json", 'w') as f:
            json.dump(timing_data, f, indent=2)
        
        # Save experiment info
        experiment_info = {
            'experiment_name': self.experiment_name,
            'start_timestamp': self.timestamp,
            'end_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'save_dir': self.save_dir,
            'total_clients': len(self.client_metrics),
            'total_rounds': len(self.server_rounds),
            'server_rounds': sorted(self.server_rounds)
        }
        
        with open(f"{self.save_dir}/experiment_info.json", 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        # NEW: Save detailed CSV with all rounds data
        self.save_detailed_csv()
        
        # NEW: Save per-round statistics
        self.save_per_round_statistics()
        
        print(f"\nMetrics saved to {self.save_dir}/")
        print(f"  - client_metrics.json")
        print(f"  - server_metrics.json")
        print(f"  - timing_metrics.json")
        print(f"  - experiment_info.json")
        print(f"  - detailed_all_rounds.csv (NEW)")
        print(f"  - per_round_statistics.csv (NEW)")
    
    def save_detailed_csv(self):
        """Save detailed CSV with all rounds data"""
        if not self.client_metrics:
            print("No client metrics to save as CSV")
            return
        
        all_rows = []
        
        for client_id, rounds_data in self.client_metrics.items():
            for round_num, data in rounds_data.items():
                metrics = data['metrics']
                
                # Extract metrics with safe defaults
                mAP50 = metrics.get('mAP50', 0)
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                box_loss = metrics.get('box_loss', 0)
                cls_loss = metrics.get('cls_loss', 0)
                dfl_loss = metrics.get('dfl_loss', 0)
                training_time = metrics.get('training_time', 0)
                num_examples = metrics.get('num_examples', 0)
                epochs_per_round = metrics.get('epochs_per_round', 1)
                total_rounds = metrics.get('total_rounds', 1)
                
                # Ensure numeric values
                try:
                    mAP50 = float(mAP50) if mAP50 else 0.0
                    precision = float(precision) if precision else 0.0
                    recall = float(recall) if recall else 0.0
                    box_loss = float(box_loss) if box_loss else 0.0
                    cls_loss = float(cls_loss) if cls_loss else 0.0
                    dfl_loss = float(dfl_loss) if dfl_loss else 0.0
                    training_time = float(training_time) if training_time else 0.0
                    num_examples = int(num_examples) if num_examples else 0
                    epochs_per_round = int(epochs_per_round) if epochs_per_round else 1
                    total_rounds = int(total_rounds) if total_rounds else 1
                except (ValueError, TypeError):
                    # Keep defaults if conversion fails
                    pass
                
                row = {
                    'Client_ID': client_id,
                    'Round': round_num,
                    'mAP50': mAP50,
                    'Precision': precision,
                    'Recall': recall,
                    'Box_Loss': box_loss,
                    'Class_Loss': cls_loss,
                    'DFL_Loss': dfl_loss,
                    'Total_Loss': box_loss + cls_loss + dfl_loss,
                    'Training_Time_s': training_time,
                    'Num_Examples': num_examples,
                    'Epochs_per_Round': epochs_per_round,
                    'Total_Rounds': total_rounds,
                    'Timestamp': data.get('timestamp', ''),
                    'Cumulative_Training_Time_s': sum(self.training_times[:round_num]) if round_num <= len(self.training_times) else 0
                }
                all_rows.append(row)
        
        df = pd.DataFrame(all_rows)
        df = df.sort_values(['Client_ID', 'Round'])
        
        # Save detailed CSV
        csv_path = f"{self.save_dir}/detailed_all_rounds.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Detailed CSV saved: {csv_path}")
        print(f"  Contains {len(df)} rows across {df['Round'].nunique()} rounds")
        
        # Also save individual client CSVs
        for client_id in df['Client_ID'].unique():
            client_df = df[df['Client_ID'] == client_id]
            client_csv_path = f"{self.save_dir}/client_{client_id}_all_rounds.csv"
            client_df.to_csv(client_csv_path, index=False)
        
        return df
    
    def save_per_round_statistics(self):
        """Save per-round statistics CSV"""
        if not self.client_metrics:
            print("No client metrics to calculate round statistics")
            return
        
        # Load the detailed data if available, otherwise create it
        detailed_csv = f"{self.save_dir}/detailed_all_rounds.csv"
        if os.path.exists(detailed_csv):
            df = pd.read_csv(detailed_csv)
        else:
            df = self.save_detailed_csv()
            if df is None:
                return
        
        # Calculate per-round statistics
        round_stats = df.groupby('Round').agg({
            'mAP50': ['mean', 'std', 'min', 'max', 'count'],
            'Precision': ['mean', 'std'],
            'Recall': ['mean', 'std'],
            'Total_Loss': ['mean', 'std'],
            'Training_Time_s': ['mean', 'sum'],
            'Num_Examples': 'sum'
        }).round(4)
        
        # Flatten column names
        round_stats.columns = ['_'.join(col).strip() for col in round_stats.columns.values]
        round_stats.reset_index(inplace=True)
        
        # Rename columns for clarity
        column_rename = {
            'mAP50_mean': 'Avg_mAP50',
            'mAP50_std': 'Std_mAP50',
            'mAP50_min': 'Min_mAP50',
            'mAP50_max': 'Max_mAP50',
            'mAP50_count': 'Client_Count',
            'Precision_mean': 'Avg_Precision',
            'Precision_std': 'Std_Precision',
            'Recall_mean': 'Avg_Recall',
            'Recall_std': 'Std_Recall',
            'Total_Loss_mean': 'Avg_Total_Loss',
            'Total_Loss_std': 'Std_Total_Loss',
            'Training_Time_s_mean': 'Avg_Training_Time_s',
            'Training_Time_s_sum': 'Total_Training_Time_s',
            'Num_Examples_sum': 'Total_Examples'
        }
        
        round_stats = round_stats.rename(columns=column_rename)
        
        # Calculate cumulative improvements
        if len(round_stats) > 1:
            round_stats['Cumulative_Improvement_%'] = 0.0
            first_avg = round_stats.iloc[0]['Avg_mAP50']
            for i in range(len(round_stats)):
                current_avg = round_stats.iloc[i]['Avg_mAP50']
                if first_avg > 0:
                    round_stats.at[i, 'Cumulative_Improvement_%'] = ((current_avg - first_avg) / first_avg * 100)
            
            # Calculate round-over-round improvement
            round_stats['Round_Over_Round_Change'] = round_stats['Avg_mAP50'].diff()
        
        # Save to CSV
        stats_path = f"{self.save_dir}/per_round_statistics.csv"
        round_stats.to_csv(stats_path, index=False)
        
        print(f"✓ Per-round statistics saved: {stats_path}")
        print(f"  Statistics for {len(round_stats)} rounds")
        
        return round_stats
    
    def generate_all_plots(self, save_format='png', dpi=300):
        """Generate all plots for the paper"""
        print("\n" + "="*60)
        print("Generating plots for paper...")
        print("="*60)
        
        try:
            self._plot_accuracy_convergence(save_format, dpi)
            print("✓ Generated accuracy convergence plot")
            
            self._plot_loss_convergence(save_format, dpi)
            print("✓ Generated loss convergence plot")
            
            self._plot_client_comparison(save_format, dpi)
            print("✓ Generated client comparison plot")
            
            if self.training_times:
                self._plot_training_time_analysis(save_format, dpi)
                print("✓ Generated training time analysis plot")
            
            if self.communication_times:
                self._plot_communication_efficiency(save_format, dpi)
                print("✓ Generated communication efficiency plot")
            
            self._plot_mAP_progression(save_format, dpi)
            print("✓ Generated mAP progression plot")
            
            self._create_summary_table()
            print("✓ Created summary tables")
            
            # NEW: Generate comprehensive analysis files
            self._generate_comprehensive_analysis()
            print("✓ Generated comprehensive analysis files")
            
            print("\n" + "="*60)
            print("All plots and analysis files generated successfully!")
            print(f"Check directory: {self.save_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_accuracy_convergence(self, save_format='png', dpi=300):
        """Plot accuracy convergence across rounds"""
        if not self.client_metrics:
            print("No client metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Federated Learning Convergence Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: mAP50 progression
        ax1 = axes[0, 0]
        for client_id in sorted(self.client_metrics.keys()):
            rounds = sorted(self.client_metrics[client_id].keys())
            mAP50_values = []
            for round_num in rounds:
                metrics = self.client_metrics[client_id][round_num]['metrics']
                mAP50 = metrics.get('mAP50', 0)
                if isinstance(mAP50, list):
                    mAP50 = mAP50[-1] if mAP50 else 0
                mAP50_values.append(float(mAP50))
            
            ax1.plot(rounds, mAP50_values, marker='o', linewidth=2, markersize=8,
                    label=f'Client {client_id}', alpha=0.8)
        
        ax1.set_xlabel('Communication Round', fontsize=12)
        ax1.set_ylabel('mAP50', fontsize=12)
        ax1.set_title('Object Detection Accuracy (mAP50)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_ylim([0, 1])
        
        # Plot 2: Precision-Recall progression
        ax2 = axes[0, 1]
        precision_metrics = []
        recall_metrics = []
        
        for round_num in sorted(self.server_rounds):
            round_precisions = []
            round_recalls = []
            for client_id in self.client_metrics.keys():
                if round_num in self.client_metrics[client_id]:
                    metrics = self.client_metrics[client_id][round_num]['metrics']
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    if precision:
                        round_precisions.append(float(precision))
                    if recall:
                        round_recalls.append(float(recall))
            
            if round_precisions:
                precision_metrics.append(np.mean(round_precisions))
            if round_recalls:
                recall_metrics.append(np.mean(round_recalls))
        
        if precision_metrics and recall_metrics:
            rounds_to_plot = sorted(self.server_rounds)[:len(precision_metrics)]
            ax2.plot(rounds_to_plot, precision_metrics, marker='o', color='#95E1D3', 
                    linewidth=2, markersize=8, label='Precision', alpha=0.8)
            ax2.plot(rounds_to_plot, recall_metrics, marker='s', color='#F38181', 
                    linewidth=2, markersize=8, label='Recall', alpha=0.8)
        
        ax2.set_xlabel('Communication Round', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Precision-Recall Progression', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_ylim([0, 1])
        
        # Plot 3: Loss components
        ax3 = axes[1, 0]
        loss_components = ['box_loss', 'cls_loss', 'dfl_loss']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, loss_type in enumerate(loss_components):
            avg_loss_per_round = []
            for round_num in sorted(self.server_rounds):
                round_losses = []
                for client_id in self.client_metrics.keys():
                    if round_num in self.client_metrics[client_id]:
                        metrics = self.client_metrics[client_id][round_num]['metrics']
                        loss_val = metrics.get(loss_type, 0)
                        if loss_val:
                            round_losses.append(float(loss_val))
                
                if round_losses:
                    avg_loss_per_round.append(np.mean(round_losses))
                else:
                    avg_loss_per_round.append(0)
            
            if avg_loss_per_round:
                rounds_to_plot = sorted(self.server_rounds)[:len(avg_loss_per_round)]
                ax3.plot(rounds_to_plot, avg_loss_per_round, marker='s', color=colors[idx], 
                        linewidth=2, markersize=6, label=f'{loss_type}', alpha=0.8)
        
        ax3.set_xlabel('Communication Round', fontsize=12)
        ax3.set_ylabel('Loss Value', fontsize=12)
        ax3.set_title('Training Loss Components', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        
        # Plot 4: Client performance distribution (final round)
        ax4 = axes[1, 1]
        if self.server_rounds:
            final_round = max(self.server_rounds)
            client_final_mAPs = []
            client_ids = []
            
            for client_id in sorted(self.client_metrics.keys()):
                if final_round in self.client_metrics[client_id]:
                    metrics = self.client_metrics[client_id][final_round]['metrics']
                    mAP50 = metrics.get('mAP50', 0)
                    if isinstance(mAP50, list):
                        mAP50 = mAP50[-1] if mAP50 else 0
                    client_final_mAPs.append(float(mAP50))
                    client_ids.append(f'Client {client_id}')
            
            if client_final_mAPs:
                colors = ['#FF9A76', '#87CEEB', '#98FB98', '#DDA0DD']
                bars = ax4.bar(range(len(client_final_mAPs)), client_final_mAPs, 
                              color=colors[:len(client_final_mAPs)],
                              edgecolor='black', linewidth=1.5)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10)
                
                ax4.set_xticks(range(len(client_final_mAPs)))
                ax4.set_xticklabels(client_ids, rotation=45, ha='right')
                ax4.set_ylabel('Final mAP50', fontsize=12)
                ax4.set_title(f'Client Performance (Round {final_round})', fontsize=14)
                ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/accuracy_convergence.{save_format}', 
                   dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_loss_convergence(self, save_format='png', dpi=300):
        """Plot detailed loss convergence"""
        if not self.client_metrics:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Individual client loss curves
        ax1 = axes[0]
        for client_id in sorted(self.client_metrics.keys()):
            rounds = sorted(self.client_metrics[client_id].keys())
            total_loss = []
            for round_num in rounds:
                metrics = self.client_metrics[client_id][round_num]['metrics']
                box_loss = metrics.get('box_loss', 0)
                cls_loss = metrics.get('cls_loss', 0)
                dfl_loss = metrics.get('dfl_loss', 0)
                total = float(box_loss) + float(cls_loss) + float(dfl_loss)
                total_loss.append(total)
            
            ax1.plot(rounds, total_loss, marker='o', linewidth=2, 
                    markersize=6, label=f'Client {client_id}', alpha=0.7)
        
        ax1.set_xlabel('Communication Round', fontsize=12)
        ax1.set_ylabel('Total Loss', fontsize=12)
        ax1.set_title('Client-wise Total Loss Convergence', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Plot 2: Average loss with confidence intervals
        ax2 = axes[1]
        all_rounds = sorted(self.server_rounds)
        avg_losses = []
        std_losses = []
        min_losses = []
        max_losses = []
        
        for round_num in all_rounds:
            round_losses = []
            for client_id in self.client_metrics.keys():
                if round_num in self.client_metrics[client_id]:
                    metrics = self.client_metrics[client_id][round_num]['metrics']
                    box_loss = metrics.get('box_loss', 0)
                    cls_loss = metrics.get('cls_loss', 0)
                    dfl_loss = metrics.get('dfl_loss', 0)
                    total = float(box_loss) + float(cls_loss) + float(dfl_loss)
                    round_losses.append(total)
            
            if round_losses:
                avg_losses.append(np.mean(round_losses))
                std_losses.append(np.std(round_losses))
                min_losses.append(np.min(round_losses))
                max_losses.append(np.max(round_losses))
        
        if avg_losses:
            rounds_to_plot = all_rounds[:len(avg_losses)]
            ax2.fill_between(rounds_to_plot, 
                            np.array(min_losses), np.array(max_losses),
                            alpha=0.2, color='skyblue', label='Min-Max Range')
            ax2.fill_between(rounds_to_plot,
                            np.array(avg_losses) - np.array(std_losses),
                            np.array(avg_losses) + np.array(std_losses),
                            alpha=0.4, color='steelblue', label='±1 Std Dev')
            ax2.plot(rounds_to_plot, avg_losses, 
                    marker='s', color='darkblue', linewidth=2, 
                    markersize=8, label='Mean Loss')
        
        ax2.set_xlabel('Communication Round', fontsize=12)
        ax2.set_ylabel('Total Loss', fontsize=12)
        ax2.set_title('Federated Average Loss with Variability', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/loss_convergence.{save_format}', 
                   dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_client_comparison(self, save_format='png', dpi=300):
        """Plot detailed client comparison"""
        n_clients = len(self.client_metrics)
        if n_clients == 0:
            return
        
        fig, axes = plt.subplots(n_clients, 3, figsize=(15, 4*n_clients))
        if n_clients == 1:
            axes = axes.reshape(1, -1)
        
        metrics_to_plot = ['mAP50', 'precision', 'recall']
        
        for idx, client_id in enumerate(sorted(self.client_metrics.keys())):
            client_data = self.client_metrics[client_id]
            rounds = sorted(client_data.keys())
            
            for j, metric in enumerate(metrics_to_plot):
                ax = axes[idx, j] if n_clients > 1 else axes[j]
                metric_values = []
                
                for round_num in rounds:
                    metrics = client_data[round_num]['metrics']
                    val = metrics.get(metric, 0)
                    if isinstance(val, list):
                        val = val[-1] if val else 0
                    metric_values.append(float(val))
                
                ax.plot(rounds, metric_values, marker='o', color=f'C{idx}', 
                       linewidth=2, markersize=6)
                ax.set_xlabel('Communication Round', fontsize=10)
                ax.set_ylabel(metric.upper(), fontsize=10)
                ax.set_title(f'Client {client_id}: {metric.upper()} Progression', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
        
        plt.suptitle('Individual Client Performance Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/client_comparison.{save_format}', 
                   dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_training_time_analysis(self, save_format='png', dpi=300):
        """Plot training time analysis"""
        if not self.training_times:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Training time per round
        ax1 = axes[0]
        rounds = range(1, len(self.training_times) + 1)
        ax1.bar(rounds, self.training_times, color='lightcoral', 
                edgecolor='darkred', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Training Time (seconds)', fontsize=12)
        ax1.set_title('Training Time per Communication Round', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add average line
        if self.training_times:
            avg_time = np.mean(self.training_times)
            ax1.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, 
                       label=f'Average: {avg_time:.2f}s')
            ax1.legend()
        
        # Plot 2: Cumulative time
        ax2 = axes[1]
        if self.training_times:
            cumulative_time = np.cumsum(self.training_times)
            ax2.plot(rounds, cumulative_time, marker='o', color='darkgreen', 
                    linewidth=2, markersize=6)
            ax2.fill_between(rounds, 0, cumulative_time, alpha=0.3, color='lightgreen')
            ax2.set_xlabel('Round', fontsize=12)
            ax2.set_ylabel('Cumulative Time (seconds)', fontsize=12)
            ax2.set_title('Cumulative Training Time', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_time_analysis.{save_format}', 
                   dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_communication_efficiency(self, save_format='png', dpi=300):
        """Plot communication efficiency metrics"""
        if not self.communication_times:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rounds = range(1, len(self.communication_times) + 1)
        
        # Plot communication time
        bars = ax.bar(rounds, self.communication_times, color='skyblue', 
                     edgecolor='navy', linewidth=1.5, alpha=0.7, 
                     label='Communication Time')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
        
        # Calculate efficiency metrics
        total_comm_time = sum(self.communication_times)
        avg_comm_time = np.mean(self.communication_times)
        
        ax.axhline(y=avg_comm_time, color='red', linestyle='--', linewidth=2,
                  label=f'Average: {avg_comm_time:.2f}s')
        
        ax.set_xlabel('Communication Round', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title(f'Communication Efficiency\nTotal Communication Time: {total_comm_time:.2f}s', 
                    fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add text box with statistics
        stats_text = f'Statistics:\n'
        stats_text += f'• Total Rounds: {len(rounds)}\n'
        stats_text += f'• Avg Comm Time: {avg_comm_time:.2f}s\n'
        stats_text += f'• Max Comm Time: {max(self.communication_times):.2f}s\n'
        stats_text += f'• Min Comm Time: {min(self.communication_times):.2f}s'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/communication_efficiency.{save_format}', 
                   dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_mAP_progression(self, save_format='png', dpi=300):
        """Plot mAP progression with detailed analysis"""
        if not self.client_metrics:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Collect all mAP50 data
        all_client_data = {}
        for client_id in sorted(self.client_metrics.keys()):
            client_rounds = sorted(self.client_metrics[client_id].keys())
            client_mAPs = []
            for round_num in client_rounds:
                metrics = self.client_metrics[client_id][round_num]['metrics']
                mAP50 = metrics.get('mAP50', 0)
                if isinstance(mAP50, list):
                    mAP50 = mAP50[-1] if mAP50 else 0
                client_mAPs.append(float(mAP50))
            all_client_data[client_id] = (client_rounds, client_mAPs)
        
        # Plot 1: Individual client mAP50 with trend lines
        ax1 = axes[0]
        for client_id, (rounds, mAPs) in all_client_data.items():
            # Plot actual values
            line = ax1.plot(rounds, mAPs, marker='o', linewidth=2, 
                           markersize=8, label=f'Client {client_id}', alpha=0.7)
            
            # Add trend line if enough data points
            if len(rounds) > 1:
                try:
                    rounds_numeric = [int(r) for r in rounds]
                    z = np.polyfit(rounds_numeric, mAPs, 1)
                    p = np.poly1d(z)
                    ax1.plot(rounds, p(rounds_numeric), linestyle='--', 
                            color=line[0].get_color(), alpha=0.5, 
                            linewidth=1.5, label=f'Client {client_id} Trend')
                except:
                    pass  # Skip trend line if polyfit fails
        
        ax1.set_xlabel('Communication Round', fontsize=12)
        ax1.set_ylabel('mAP50', fontsize=12)
        ax1.set_title('Client-wise mAP50 Progression with Trend Lines', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_ylim([0, 1])
        
        # Plot 2: Federated average mAP with confidence intervals
        ax2 = axes[1]
        if all_client_data:
            # Calculate average across clients for each round
            max_rounds = max(len(rounds) for rounds, _ in all_client_data.values())
            avg_mAPs = []
            std_mAPs = []
            min_mAPs = []
            max_mAPs = []
            
            for round_idx in range(max_rounds):
                round_mAPs = []
                for rounds, mAPs in all_client_data.values():
                    if round_idx < len(rounds):
                        round_mAPs.append(mAPs[round_idx])
                
                if round_mAPs:
                    avg_mAPs.append(np.mean(round_mAPs))
                    std_mAPs.append(np.std(round_mAPs))
                    min_mAPs.append(np.min(round_mAPs))
                    max_mAPs.append(np.max(round_mAPs))
            
            if avg_mAPs:
                rounds_x = list(range(1, len(avg_mAPs) + 1))
                
                # Plot confidence intervals
                ax2.fill_between(rounds_x, min_mAPs, max_mAPs, 
                                alpha=0.2, color='lightgreen', label='Client Range')
                ax2.fill_between(rounds_x, 
                                np.array(avg_mAPs) - np.array(std_mAPs),
                                np.array(avg_mAPs) + np.array(std_mAPs),
                                alpha=0.4, color='mediumseagreen', label='±1 Std Dev')
                
                # Plot average line
                ax2.plot(rounds_x, avg_mAPs, marker='s', color='darkgreen',
                        linewidth=3, markersize=10, label='Federated Average')
                
                # Add improvement annotation
                if len(avg_mAPs) > 1 and avg_mAPs[0] > 0:
                    improvement = ((avg_mAPs[-1] - avg_mAPs[0]) / avg_mAPs[0] * 100)
                    ax2.annotate(f'+{improvement:.1f}%', 
                                xy=(rounds_x[-1], avg_mAPs[-1]),
                                xytext=(rounds_x[-1] - 0.5, avg_mAPs[-1] + 0.1),
                                arrowprops=dict(arrowstyle='->', color='red'),
                                fontsize=12, color='red', fontweight='bold')
        
        ax2.set_xlabel('Communication Round', fontsize=12)
        ax2.set_ylabel('Average mAP50', fontsize=12)
        ax2.set_title('Federated Average mAP50 with Variability', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/mAP_progression.{save_format}', 
                   dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _create_summary_table(self):
        """Create a summary table of all metrics"""
        if not self.client_metrics:
            return
        
        # Collect summary statistics
        summary_data = []
        final_round = max(self.server_rounds) if self.server_rounds else 0
        
        for client_id in sorted(self.client_metrics.keys()):
            client_rounds = sorted(self.client_metrics[client_id].keys())
            if not client_rounds:
                continue
            
            # Get initial and final metrics
            first_round = client_rounds[0]
            last_round = client_rounds[-1]
            
            first_metrics = self.client_metrics[client_id][first_round]['metrics']
            last_metrics = self.client_metrics[client_id][last_round]['metrics']
            
            # Calculate improvements
            initial_mAP = first_metrics.get('mAP50', 0)
            final_mAP = last_metrics.get('mAP50', 0)
            if isinstance(initial_mAP, list):
                initial_mAP = initial_mAP[-1] if initial_mAP else 0
            if isinstance(final_mAP, list):
                final_mAP = final_mAP[-1] if final_mAP else 0
            
            initial_mAP = float(initial_mAP)
            final_mAP = float(final_mAP)
            
            mAP_improvement = ((final_mAP - initial_mAP) / initial_mAP * 100) if initial_mAP > 0 else 0
            
            summary_data.append({
                'Client ID': client_id,
                'Rounds Completed': len(client_rounds),
                'Initial mAP50': f'{initial_mAP:.4f}',
                'Final mAP50': f'{final_mAP:.4f}',
                'Improvement (%)': f'{mAP_improvement:.2f}%',
                'Final Precision': f"{last_metrics.get('precision', 0):.4f}",
                'Final Recall': f"{last_metrics.get('recall', 0):.4f}",
            })
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(f'{self.save_dir}/summary_statistics.csv', index=False)
        
        # Also create a LaTeX table for papers
        try:
            latex_table = df.to_latex(index=False, float_format="%.4f")
            with open(f'{self.save_dir}/summary_table.tex', 'w') as f:
                f.write(latex_table)
            print(f"Summary tables saved to {self.save_dir}/")
        except:
            # If LaTeX export fails, just save as markdown
            with open(f'{self.save_dir}/summary_table.md', 'w') as f:
                f.write(df.to_markdown(index=False))
            print(f"Summary tables saved to {self.save_dir}/ (markdown format)")
    
    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis files including all rounds data"""
        if not self.client_metrics:
            print("No client metrics for comprehensive analysis")
            return
        
        try:
            # Create Excel file with multiple sheets
            excel_path = f"{self.save_dir}/comprehensive_analysis.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: All rounds data
                detailed_df = self.save_detailed_csv()
                if detailed_df is not None:
                    detailed_df.to_excel(writer, sheet_name='All_Rounds_Data', index=False)
                
                # Sheet 2: Per-round statistics
                stats_df = self.save_per_round_statistics()
                if stats_df is not None:
                    stats_df.to_excel(writer, sheet_name='Per_Round_Stats', index=False)
                
                # Sheet 3: Client comparison
                client_comparison = []
                for client_id in sorted(self.client_metrics.keys()):
                    client_rounds = sorted(self.client_metrics[client_id].keys())
                    if client_rounds:
                        first = client_rounds[0]
                        last = client_rounds[-1]
                        first_mAP = self.client_metrics[client_id][first]['metrics'].get('mAP50', 0)
                        last_mAP = self.client_metrics[client_id][last]['metrics'].get('mAP50', 0)
                        
                        client_comparison.append({
                            'Client ID': client_id,
                            'First Round mAP50': float(first_mAP),
                            'Last Round mAP50': float(last_mAP),
                            'Improvement': float(last_mAP) - float(first_mAP),
                            'Improvement %': ((float(last_mAP) - float(first_mAP)) / float(first_mAP) * 100) if float(first_mAP) > 0 else 0,
                            'Rounds Completed': len(client_rounds)
                        })
                
                if client_comparison:
                    comparison_df = pd.DataFrame(client_comparison)
                    comparison_df.to_excel(writer, sheet_name='Client_Comparison', index=False)
                
                # Sheet 4: Training time analysis
                if self.training_times:
                    time_data = {
                        'Round': list(range(1, len(self.training_times) + 1)),
                        'Training_Time_s': self.training_times,
                        'Cumulative_Time_s': np.cumsum(self.training_times).tolist()
                    }
                    time_df = pd.DataFrame(time_data)
                    time_df.to_excel(writer, sheet_name='Training_Times', index=False)
            
            print(f"✓ Comprehensive Excel analysis saved: {excel_path}")
            
            # Create a simple text summary
            summary_path = f"{self.save_dir}/experiment_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("FEDERATED LEARNING EXPERIMENT SUMMARY\n")
                f.write("="*60 + "\n\n")
                f.write(f"Experiment: {self.experiment_name}\n")
                f.write(f"Timestamp: {self.timestamp}\n")
                f.write(f"Total Clients: {len(self.client_metrics)}\n")
                f.write(f"Total Rounds: {len(self.server_rounds)}\n")
                f.write(f"Rounds: {sorted(self.server_rounds)}\n\n")
                
                f.write("CLIENT PERFORMANCE:\n")
                f.write("-"*40 + "\n")
                for client_id in sorted(self.client_metrics.keys()):
                    client_rounds = sorted(self.client_metrics[client_id].keys())
                    if client_rounds:
                        first = client_rounds[0]
                        last = client_rounds[-1]
                        first_mAP = self.client_metrics[client_id][first]['metrics'].get('mAP50', 0)
                        last_mAP = self.client_metrics[client_id][last]['metrics'].get('mAP50', 0)
                        improvement = ((float(last_mAP) - float(first_mAP)) / float(first_mAP) * 100) if float(first_mAP) > 0 else 0
                        f.write(f"Client {client_id}: Round {first} mAP50={float(first_mAP):.4f} → Round {last} mAP50={float(last_mAP):.4f} ({improvement:+.1f}%)\n")
                
                f.write(f"\nFiles generated in: {self.save_dir}/\n")
                f.write("="*60 + "\n")
            
            print(f"✓ Experiment summary saved: {summary_path}")
            
        except Exception as e:
            print(f"⚠ Error generating comprehensive analysis: {e}")

# Helper function to analyze saved experiments
def analyze_experiment(experiment_dir):
    """Load and analyze a saved experiment"""
    import json
    
    print(f"\nAnalyzing experiment: {experiment_dir}")
    print("="*60)
    
    # Load client metrics
    with open(f"{experiment_dir}/client_metrics.json", 'r') as f:
        client_metrics = json.load(f)
    
    # Convert to the format expected by the logger
    parsed_metrics = {}
    for client_key, rounds_data in client_metrics.items():
        client_id = int(client_key.replace('client_', ''))
        parsed_metrics[client_id] = {}
        
        for round_str, data in rounds_data.items():
            round_num = int(round_str)
            parsed_metrics[client_id][round_num] = {
                'timestamp': data.get('timestamp', 0),
                'metrics': data.get('metrics', {})
            }
    
    # Create a logger instance to generate plots
    logger = FederatedMetricsLogger()
    logger.client_metrics = parsed_metrics
    logger.server_rounds = sorted(set(
        round_num 
        for client_data in parsed_metrics.values() 
        for round_num in client_data.keys()
    ))
    logger.save_dir = experiment_dir
    
    # Generate plots and analysis
    logger.save_metrics()
    logger.generate_all_plots()
    
    return logger

# Main function for standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Metrics Logger')
    parser.add_argument('--analyze', type=str, help='Analyze a saved experiment directory')
    parser.add_argument('--demo', action='store_true', help='Generate demo plots with sample data')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_experiment(args.analyze)
    elif args.demo:
        # Create demo metrics
        logger = FederatedMetricsLogger(experiment_name="demo_experiment")
        
        # Add some sample data
        for client_id in [0, 1]:
            for round_num in range(1, 6):
                logger.log_client_training(
                    client_id=client_id,
                    server_round=round_num,
                    metrics={
                        'mAP50': 0.1 + (round_num-1)*0.15 + client_id*0.05 + np.random.normal(0, 0.02),
                        'precision': 0.15 + (round_num-1)*0.12 + client_id*0.03 + np.random.normal(0, 0.02),
                        'recall': 0.12 + (round_num-1)*0.1 + client_id*0.04 + np.random.normal(0, 0.02),
                        'box_loss': 1.5 - (round_num-1)*0.2 + np.random.normal(0, 0.1),
                        'cls_loss': 3.0 - (round_num-1)*0.3 + np.random.normal(0, 0.15),
                        'dfl_loss': 1.6 - (round_num-1)*0.15 + np.random.normal(0, 0.08),
                        'training_time': 30 + np.random.normal(0, 5),
                        'num_examples': 893,
                        'epochs_per_round': 3,
                        'total_rounds': 5
                    }
                )
        
        logger.training_times = [35, 32, 30, 28, 29]
        logger.communication_times = [2.1, 1.9, 2.0, 1.8, 2.1]
        
        # Save and generate plots
        logger.save_metrics()
        logger.generate_all_plots()
        
        print(f"\nDemo complete! Check directory: {logger.save_dir}")
        print(f"\nFiles created:")
        print(f"  • detailed_all_rounds.csv - Contains ALL rounds data")
        print(f"  • per_round_statistics.csv - Statistics for each round")
        print(f"  • comprehensive_analysis.xlsx - Excel with multiple sheets")
        print(f"  • client_*_all_rounds.csv - Individual client data")
    else:
        print("Usage:")
        print("  python metrics_logger.py --analyze <experiment_dir>  # Analyze saved experiment")
        print("  python metrics_logger.py --demo                      # Generate demo plots")
        print("\nFor your federated learning runs, the logger will automatically:")
        print("  1. Save detailed_all_rounds.csv with ALL rounds data")
        print("  2. Create per_round_statistics.csv for each round")
        print("  3. Generate comprehensive_analysis.xlsx")
        print("  4. Create individual client_*_all_rounds.csv files")