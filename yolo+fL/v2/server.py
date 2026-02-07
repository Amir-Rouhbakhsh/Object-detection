import flwr as fl
from flwr.server.strategy import FedAvg
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict, Optional
import time
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION - UPDATED FOR BETTER RESULTS
# ============================================================================
CUSTOM_SAVE_PATH = r"C:\Users\PC\Desktop\dentalresearch\models\federated\flcodes"
EPOCHS_PER_ROUND = 5        # Increased from 3 to 5 for dental detection
TOTAL_ROUNDS = 20           # Increased from 15 to 20 for better convergence
MIN_CLIENTS = 2
SERVER_PORT = 8080
EXPERIMENT_NAME = f"yolo_dental_epochs{EPOCHS_PER_ROUND}_rounds{TOTAL_ROUNDS}"

# Create custom save directory if it doesn't exist
os.makedirs(CUSTOM_SAVE_PATH, exist_ok=True)
print(f"✓ All results will be saved to: {CUSTOM_SAVE_PATH}")

# ============================================================================
# METRICS LOGGER INITIALIZATION
# ============================================================================
try:
    from metrics_logger import FederatedMetricsLogger
    
    # Initialize with custom save path
    metrics_logger = FederatedMetricsLogger(
        experiment_name=EXPERIMENT_NAME,
        base_save_path=CUSTOM_SAVE_PATH  # This uses the base_save_path parameter
    )
    
    print(f"✓ Metrics logger initialized successfully")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Save location: {metrics_logger.save_dir}")
    
except ImportError as e:
    print(f"⚠ ERROR: Could not import metrics_logger.py")
    print(f"  Error details: {e}")
    print(f"  Make sure metrics_logger.py is in the same directory as server.py")
    print("  Running without advanced metrics tracking...")
    metrics_logger = None
except Exception as e:
    print(f"⚠ ERROR: Failed to initialize metrics logger: {e}")
    print("  Running without advanced metrics tracking...")
    metrics_logger = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_weights(model: YOLO):
    """Return model parameters as list of NumPy arrays"""
    state_dict = model.model.state_dict()
    weights = []
    for key, val in state_dict.items():
        if 'num_batches_tracked' not in key and 'running_mean' not in key and 'running_var' not in key:
            if val.requires_grad or val.dim() > 0:
                weights.append(val.detach().cpu().numpy())
    print(f"Server: Extracting {len(weights)} parameters from model")
    return weights

def set_weights(model: YOLO, parameters):
    """Set model parameters from list of NumPy arrays"""
    state_dict = model.model.state_dict()
    keys = []
    for key in state_dict.keys():
        if 'num_batches_tracked' not in key and 'running_mean' not in key and 'running_var' not in key:
            if state_dict[key].requires_grad or state_dict[key].dim() > 0:
                keys.append(key)
    
    if len(parameters) != len(keys):
        print(f"Server: Warning - Expected {len(keys)} parameters, got {len(parameters)}")
        return False
    
    new_state_dict = state_dict.copy()
    for i, key in enumerate(keys):
        if i < len(parameters):
            param_tensor = torch.tensor(parameters[i], device=state_dict[key].device)
            if param_tensor.shape == state_dict[key].shape:
                new_state_dict[key] = param_tensor
            else:
                print(f"Server: Shape mismatch for {key}: {state_dict[key].shape} vs {param_tensor.shape}")
                return False
    
    model.model.load_state_dict(new_state_dict, strict=False)
    return True

def load_model():
    """Load YOLOv11n model"""
    model = YOLO('yolo11n.pt')
    return model

# ============================================================================
# CUSTOM FEDERATED STRATEGY WITH EPOCHS CONFIGURATION
# ============================================================================
class SaveModelStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        # Store configuration parameters
        self.total_rounds = kwargs.pop('total_rounds', TOTAL_ROUNDS)
        self.epochs_per_round = kwargs.pop('epochs_per_round', EPOCHS_PER_ROUND)
        self.start_time = time.time()
        self.round_start_times = {}
        self.round_metrics = {}
        super().__init__(*args, **kwargs)
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training with epochs configuration."""
        self.round_start_times[server_round] = time.time()
        
        print(f"\n{'='*60}")
        print(f"FEDERATED ROUND {server_round}/{self.total_rounds}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  • Epochs per client: {self.epochs_per_round}")
        print(f"  • Target clients: {self.min_fit_clients}")
        print(f"  • Total rounds: {self.total_rounds}")
        print(f"  • Save location: {CUSTOM_SAVE_PATH}")
        
        # Initialize round metrics
        self.round_metrics[server_round] = {
            'client_count': 0,
            'mAP50_values': [],
            'training_times': [],
            'example_counts': []
        }
        
        # Get base configuration from parent (returns list of tuples)
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Add epochs configuration to each client instruction
        updated_instructions = []
        for client_proxy, fit_instructions in client_instructions:
            # Ensure config is a dictionary
            if fit_instructions.config is None:
                fit_instructions.config = {}
            
            # Add epochs and round info to the config
            fit_instructions.config["epochs_per_round"] = self.epochs_per_round
            fit_instructions.config["server_round"] = server_round
            fit_instructions.config["total_rounds"] = self.total_rounds
            
            updated_instructions.append((client_proxy, fit_instructions))
        
        return updated_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, Exception]] = [],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, float]]:
        
        round_start_time = self.round_start_times.get(server_round, time.time())
        round_duration = time.time() - round_start_time
        
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            params = fl.common.parameters_to_ndarrays(aggregated_parameters)
            print(f"\n✓ ROUND {server_round} COMPLETE")
            print(f"  Parameters aggregated: {len(params)}")
            print(f"  Clients participated: {len(results)}")
            print(f"  Round duration: {round_duration:.2f} seconds")
            
            # Save aggregated model to custom directory
            try:
                aggregated_model = load_model()
                success = set_weights(aggregated_model, params)
                if success:
                    # Save model in the custom directory
                    model_filename = f"federated_model_round_{server_round}.pt"
                    model_path = os.path.join(CUSTOM_SAVE_PATH, model_filename)
                    aggregated_model.save(model_path)
                    print(f"  Model saved: {model_path}")
                    
                    # Also save in metrics directory for reference
                    if metrics_logger:
                        metrics_model_path = os.path.join(metrics_logger.save_dir, model_filename)
                        aggregated_model.save(metrics_model_path)
                    
                    # Save final model separately if it's the last round
                    if server_round == self.total_rounds:
                        final_model_path = os.path.join(CUSTOM_SAVE_PATH, "federated_model_final.pt")
                        aggregated_model.save(final_model_path)
                        print(f"  🏆 FINAL MODEL saved: {final_model_path}")
                        
                        # Also save as best model
                        best_model_path = os.path.join(CUSTOM_SAVE_PATH, "federated_model_best.pt")
                        aggregated_model.save(best_model_path)
                        print(f"  🥇 BEST MODEL saved: {best_model_path}")
                else:
                    print(f"  ⚠ Could not set weights for model saving")
            except Exception as e:
                print(f"  ⚠ Could not save model: {e}")
            
            # Log training time
            if metrics_logger:
                metrics_logger.log_training_time(round_duration)
        
        # Log detailed client metrics
        print(f"\n  Client Results Summary:")
        client_details = []
        
        for client_proxy, fit_res in results:
            client_metrics = fit_res.metrics
            
            # Ensure metrics is a dictionary
            if not isinstance(client_metrics, dict):
                client_metrics = {}
            
            client_id = client_metrics.get('client_id', 0)
            
            # Extract metrics with safe defaults
            mAP50 = client_metrics.get('mAP50', 0)
            precision = client_metrics.get('precision', 0)
            recall = client_metrics.get('recall', 0)
            training_time = client_metrics.get('training_time', 0)
            box_loss = client_metrics.get('box_loss', 0)
            cls_loss = client_metrics.get('cls_loss', 0)
            dfl_loss = client_metrics.get('dfl_loss', 0)
            
            # Ensure numeric values
            try:
                mAP50 = float(mAP50) if mAP50 else 0.0
                precision = float(precision) if precision else 0.0
                recall = float(recall) if recall else 0.0
                training_time = float(training_time) if training_time else 0.0
                box_loss = float(box_loss) if box_loss else 0.0
                cls_loss = float(cls_loss) if cls_loss else 0.0
                dfl_loss = float(dfl_loss) if dfl_loss else 0.0
            except (ValueError, TypeError):
                mAP50 = precision = recall = training_time = box_loss = cls_loss = dfl_loss = 0.0
            
            # Store for summary
            client_details.append({
                'id': client_id,
                'mAP50': mAP50,
                'examples': fit_res.num_examples,
                'training_time': training_time
            })
            
            # Prepare comprehensive metrics for logging
            comprehensive_metrics = {
                'client_id': client_id,
                'round': server_round,
                'mAP50': mAP50,
                'precision': precision,
                'recall': recall,
                'box_loss': box_loss,
                'cls_loss': cls_loss,
                'dfl_loss': dfl_loss,
                'training_time': training_time,
                'num_examples': fit_res.num_examples,
                'epochs_per_round': self.epochs_per_round,
                'total_rounds': self.total_rounds
            }
            
            # Log to metrics logger if available
            if metrics_logger:
                metrics_logger.log_client_training(
                    client_id=client_id,
                    server_round=server_round,
                    metrics=comprehensive_metrics
                )
        
        # Print formatted client details
        for detail in client_details:
            print(f"  • Client {detail['id']}:")
            print(f"      Examples: {detail['examples']}")
            print(f"      mAP50: {detail['mAP50']:.3f}")
            if detail['training_time'] > 0:
                print(f"      Training time: {detail['training_time']:.1f}s")
        
        # Calculate and display round statistics
        if results:
            mAP50_values = [detail['mAP50'] for detail in client_details]
            avg_mAP50 = sum(mAP50_values) / len(mAP50_values) if mAP50_values else 0
            total_examples = sum(detail['examples'] for detail in client_details)
            avg_training_time = sum(detail['training_time'] for detail in client_details) / len(client_details) if client_details else 0
            
            aggregated_metrics.update({
                'round_duration': round_duration,
                'total_duration': time.time() - self.start_time,
                'avg_mAP50': avg_mAP50,
                'num_clients': len(results),
                'epochs_per_round': self.epochs_per_round,
                'total_examples': total_examples,
                'avg_training_time': avg_training_time
            })
            
            print(f"\n  Round Statistics:")
            print(f"    Average mAP50: {avg_mAP50:.3f}")
            print(f"    Total examples processed: {total_examples}")
            print(f"    Average training time per client: {avg_training_time:.1f}s")
            
            # Log server aggregation metrics
            if metrics_logger:
                metrics_logger.log_server_aggregation(server_round, aggregated_metrics)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, Exception]] = [],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results."""
        # Log communication time (approximated as time since start)
        if metrics_logger:
            comm_time = time.time() - self.start_time
            metrics_logger.log_communication_time(comm_time)
        
        return super().aggregate_evaluate(server_round, results, failures)

# ============================================================================
# MAIN SERVER EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Display startup banner
    print(f"\n{'='*60}")
    print("FEDERATED LEARNING SERVER - DENTAL OBJECT DETECTION")
    print("="*60)
    print(f"Configuration:")
    print(f"  • Model: YOLOv11n")
    print(f"  • Epochs per round: {EPOCHS_PER_ROUND}")
    print(f"  • Total rounds: {TOTAL_ROUNDS}")
    print(f"  • Minimum clients: {MIN_CLIENTS}")
    print(f"  • Save location: {CUSTOM_SAVE_PATH}")
    print(f"  • Server address: 0.0.0.0:{SERVER_PORT}")
    print(f"  • Experiment: {EXPERIMENT_NAME}")
    print("="*60)
    
    # Load initial model
    try:
        model = load_model()
        initial_parameters = fl.common.ndarrays_to_parameters(get_weights(model))
        print(f"\n✓ Initial parameters prepared: {len(initial_parameters.tensors)} tensors")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load initial model: {e}")
        print("  Make sure 'yolo11n.pt' is in the current directory")
        exit(1)
    
    # Create strategy with epochs configuration
    try:
        strategy = SaveModelStrategy(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=MIN_CLIENTS,
            min_evaluate_clients=MIN_CLIENTS,
            min_available_clients=MIN_CLIENTS,
            initial_parameters=initial_parameters,
            total_rounds=TOTAL_ROUNDS,
            epochs_per_round=EPOCHS_PER_ROUND,
        )
    except Exception as e:
        print(f"\n❌ ERROR: Failed to create strategy: {e}")
        exit(1)
    
    print("\n" + "="*60)
    print("STARTING FEDERATED LEARNING SERVER")
    print("="*60)
    print(f"Waiting for {MIN_CLIENTS} clients to connect...")
    
    try:
        # Start Flower server
        fl.server.start_server(
            server_address=f"0.0.0.0:{SERVER_PORT}",
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
        )
        
        # ====================================================================
        # POST-TRAINING PROCESSING
        # ====================================================================
        print("\n" + "="*60)
        print("FEDERATED LEARNING COMPLETE!")
        print("="*60)
        
        # Save metrics and generate plots
        if metrics_logger:
            print("\n📊 SAVING METRICS AND GENERATING PLOTS...")
            try:
                # Save metrics to JSON files
                metrics_logger.save_metrics()
                print(f"  ✓ Metrics saved to JSON files")
                
                # Generate plots
                metrics_logger.generate_all_plots(save_format='pdf', dpi=300)
                print(f"  ✓ Plots generated in PDF format (suitable for papers)")
                
                # Generate PNG versions as well (for quick viewing)
                metrics_logger.generate_all_plots(save_format='png', dpi=150)
                print(f"  ✓ Plots generated in PNG format (for quick viewing)")
                
                print(f"\n📁 ALL RESULTS SAVED TO:")
                print(f"  {metrics_logger.save_dir}/")
                print(f"\n  Files created:")
                print(f"    • JSON Files:")
                print(f"        - client_metrics.json")
                print(f"        - server_metrics.json")
                print(f"        - timing_metrics.json")
                print(f"        - experiment_info.json")
                print(f"    • PDF Plots (for paper):")
                print(f"        - accuracy_convergence.pdf")
                print(f"        - loss_convergence.pdf")
                print(f"        - mAP_progression.pdf")
                print(f"        - client_comparison.pdf")
                print(f"    • Summary Tables:")
                print(f"        - summary_statistics.csv")
                print(f"        - summary_table.tex (for LaTeX papers)")
                
            except Exception as e:
                print(f"  ⚠ Error generating plots: {e}")
                import traceback
                traceback.print_exc()
                print(f"  But metrics were saved to JSON files.")
        else:
            print("\n⚠ No metrics logger available. Results were not saved.")
        
        # List saved models
        print(f"\n🤖 SAVED MODELS:")
        saved_models = []
        for i in range(1, TOTAL_ROUNDS + 1):
            model_path = os.path.join(CUSTOM_SAVE_PATH, f"federated_model_round_{i}.pt")
            if os.path.exists(model_path):
                saved_models.append(model_path)
        
        # Add final and best models
        final_model_path = os.path.join(CUSTOM_SAVE_PATH, "federated_model_final.pt")
        best_model_path = os.path.join(CUSTOM_SAVE_PATH, "federated_model_best.pt")
        
        if os.path.exists(final_model_path):
            saved_models.append(final_model_path)
        if os.path.exists(best_model_path):
            saved_models.append(best_model_path)
        
        if saved_models:
            for model_path in saved_models:
                try:
                    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                    filename = os.path.basename(model_path)
                    if "final" in filename:
                        print(f"  🏆 {filename} ({file_size:.1f} MB) - FINAL MODEL")
                    elif "best" in filename:
                        print(f"  🥇 {filename} ({file_size:.1f} MB) - BEST MODEL")
                    else:
                        print(f"  • {filename} ({file_size:.1f} MB)")
                except:
                    filename = os.path.basename(model_path)
                    if "final" in filename:
                        print(f"  🏆 {filename} - FINAL MODEL")
                    elif "best" in filename:
                        print(f"  🥇 {filename} - BEST MODEL")
                    else:
                        print(f"  • {filename}")
            
            print(f"\n  Recommended models for your paper:")
            print(f"    1. federated_model_final.pt - Complete training")
            print(f"    2. federated_model_best.pt - Best performing model")
        else:
            print("  No models were saved")
        
        # Display next steps
        print(f"\n📈 NEXT STEPS FOR YOUR PAPER:")
        print(f"  1. Check plots in: {CUSTOM_SAVE_PATH}/")
        if metrics_logger:
            print(f"  2. Use plots from: {metrics_logger.save_dir}/")
            print(f"  3. Include summary table from: summary_statistics.csv")
        print(f"  4. Use final model: federated_model_final.pt")
        print(f"  5. For additional analysis, run:")
        print(f"     python metrics_logger.py --analyze {metrics_logger.save_dir if metrics_logger else 'fl_metrics_*'}")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print(f"\n⚠ Server interrupted by user")
        if metrics_logger:
            print(f"  Saving collected metrics before exit...")
            try:
                metrics_logger.save_metrics()
                print(f"  Partial results saved to: {metrics_logger.save_dir}/")
            except:
                print(f"  Could not save metrics")
        print(f"  Partial models may be in: {CUSTOM_SAVE_PATH}/")
        
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        
        if metrics_logger:
            print(f"\n  Saving collected metrics before exit...")
            try:
                metrics_logger.save_metrics()
                print(f"  Partial results saved to: {metrics_logger.save_dir}/")
            except:
                print(f"  Could not save metrics")
        print(f"  Check {CUSTOM_SAVE_PATH}/ for partial results")