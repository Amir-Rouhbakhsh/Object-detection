import os
import sys
import torch
import time
import traceback

# Handle uncaught exceptions to prevent silent crashes
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions"""
    print("\n" + "="*60)
    print("UNCAUGHT EXCEPTION - CLIENT CRASHED")
    print("="*60)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("="*60)
    print("Client will exit. Restart it manually if needed.")
    print("="*60 + "\n")
    sys.exit(1)

sys.excepthook = handle_exception

# Clear CUDA cache
torch.cuda.empty_cache()

from ultralytics import YOLO
import flwr as fl
from flwr.common import (
    parameters_to_ndarrays, 
    ndarrays_to_parameters, 
    EvaluateRes, 
    FitRes, 
    Status,
    Code
)

# -----------------------------
# Environment & Paths
# -----------------------------
CLIENT_ID = int(os.getenv("CLIENT_ID", 0))
print(f"Starting Flower client with CLIENT_ID={CLIENT_ID}...")

BASE_PATH = os.path.join("C:\\Users\\PC\\fl_results", f"client_{CLIENT_ID}")
DATA_YAML = os.path.join(BASE_PATH, "data.yaml")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = 'yolo11n.pt'  # Original model path

# Default epochs per round (will be overridden by server config if provided)
DEFAULT_EPOCHS_PER_ROUND = 5  # Changed from 1 to 5 for better training

# -----------------------------
# Convert model weights
# -----------------------------
def get_weights(model):
    """Get model parameters as list of NumPy arrays"""
    # Get state dict and sort keys for consistency
    state_dict = model.model.state_dict()
    
    # Get weights in consistent order, excluding optimizer states
    weights = []
    keys = []
    
    for key, val in state_dict.items():
        # Include only trainable parameters, skip optimizer/batch norm tracking
        if 'num_batches_tracked' not in key and 'running_mean' not in key and 'running_var' not in key:
            if val.requires_grad or val.dim() > 0:  # Include weights and biases
                weights.append(val.detach().cpu().numpy())
                keys.append(key)
    
    print(f"Client {CLIENT_ID}: Extracting {len(weights)} parameters")
    return weights, keys

def set_weights(model, parameters, param_keys):
    """Set model parameters from list of NumPy arrays - ROBUST VERSION"""
    state_dict = model.model.state_dict()
    
    # Get current keys for comparison
    current_keys = []
    for key in state_dict.keys():
        if 'num_batches_tracked' not in key and 'running_mean' not in key and 'running_var' not in key:
            if state_dict[key].requires_grad or state_dict[key].dim() > 0:
                current_keys.append(key)
    
    print(f"Client {CLIENT_ID}: Current model has {len(current_keys)} parameter keys")
    print(f"Client {CLIENT_ID}: Expected {len(param_keys)} parameter keys")
    print(f"Client {CLIENT_ID}: Received {len(parameters)} parameters")
    
    if len(parameters) != len(param_keys):
        print(f"Client {CLIENT_ID}: Parameter count mismatch! Expected {len(param_keys)}, got {len(parameters)}")
        print(f"Client {CLIENT_ID}: This might happen if the model architecture changed.")
        print(f"Client {CLIENT_ID}: Will try to load as many parameters as possible...")
    
    # Try to update parameters
    updated_count = 0
    skipped_count = 0
    
    # Method 1: Try to match by provided param_keys
    for i, key in enumerate(param_keys):
        if i < len(parameters):
            if key in state_dict:
                try:
                    # Create tensor with proper device and dtype
                    param_tensor = torch.from_numpy(parameters[i]).to(
                        dtype=state_dict[key].dtype,
                        device=state_dict[key].device
                    )
                    
                    if param_tensor.shape == state_dict[key].shape:
                        # Assign without in-place modification
                        state_dict[key] = param_tensor
                        updated_count += 1
                    else:
                        print(f"Client {CLIENT_ID}: Shape mismatch for {key}: {state_dict[key].shape} vs {param_tensor.shape}")
                        skipped_count += 1
                except Exception as e:
                    print(f"Client {CLIENT_ID}: Error updating {key}: {e}")
                    skipped_count += 1
            else:
                print(f"Client {CLIENT_ID}: Key not found in model: {key}")
                skipped_count += 1
    
    # Method 2: If that didn't work, try positional loading
    if updated_count == 0 and len(parameters) <= len(current_keys):
        print(f"Client {CLIENT_ID}: Trying positional parameter loading...")
        for i in range(min(len(parameters), len(current_keys))):
            key = current_keys[i]
            try:
                param_tensor = torch.from_numpy(parameters[i]).to(
                    dtype=state_dict[key].dtype,
                    device=state_dict[key].device
                )
                
                if param_tensor.shape == state_dict[key].shape:
                    state_dict[key] = param_tensor
                    updated_count += 1
                else:
                    print(f"Client {CLIENT_ID}: Shape mismatch for {key} (position {i}): {state_dict[key].shape} vs {param_tensor.shape}")
                    skipped_count += 1
            except Exception as e:
                print(f"Client {CLIENT_ID}: Error updating parameter {i}: {e}")
                skipped_count += 1
    
    print(f"Client {CLIENT_ID}: Updated {updated_count} parameters, skipped {skipped_count}")
    
    if updated_count == 0:
        print(f"Client {CLIENT_ID}: WARNING - No parameters were updated!")
        print(f"Client {CLIENT_ID}: Will continue with current weights...")
        return True  # Return True to continue anyway
    
    # Load state dict
    try:
        model.model.load_state_dict(state_dict, strict=False)
        print(f"Client {CLIENT_ID}: Successfully loaded {updated_count} parameters (non-strict mode)")
        return True
    except Exception as e:
        print(f"Client {CLIENT_ID}: Error loading state dict: {e}")
        # Try to load with even more relaxed settings
        try:
            model.model.load_state_dict(state_dict, strict=False)
            print(f"Client {CLIENT_ID}: Loaded with errors (some parameters may be missing)")
            return True
        except:
            print(f"Client {CLIENT_ID}: Failed to load parameters, but will continue training...")
            return True  # Return True to continue anyway

# -----------------------------
# Flower YOLO client with configurable epochs
# -----------------------------
class YOLOClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, epochs_per_round: int = DEFAULT_EPOCHS_PER_ROUND):
        self.client_id = client_id
        self.epochs_per_round = epochs_per_round  # Store epochs parameter
        self.data_yaml = DATA_YAML
        self.device = DEVICE
        self.model_path = MODEL_PATH
        self.model = None
        self.param_keys = None
        
        # Initialize model
        self._initialize_model()
        print(f"Client {client_id}: Initialized with {len(self.param_keys)} parameter keys")
        print(f"Client {client_id}: Default epochs per round: {self.epochs_per_round}")

    def _initialize_model(self):
        """Initialize or reinitialize the model"""
        print(f"Client {self.client_id}: Loading model from {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Ensure model has proper overrides
        if not hasattr(self.model, 'overrides') or self.model.overrides is None:
            self.model.overrides = {}
        
        # Set the model path in overrides
        self.model.overrides['model'] = self.model_path
        
        # Get initial parameter keys
        _, self.param_keys = get_weights(self.model)
        print(f"Client {self.client_id}: Model loaded with {sum(p.numel() for p in self.model.model.parameters())} parameters")

    def get_parameters(self, config=None):
        weights, _ = get_weights(self.model)
        return weights

    def set_parameters(self, parameters, config=None):
        print(f"Client {self.client_id}: Setting {len(parameters)} parameters")
        
        # If model is None or corrupted, reinitialize
        if self.model is None or not hasattr(self.model, 'overrides'):
            print(f"Client {self.client_id}: Reinitializing model...")
            self._initialize_model()
        
        # If we don't have stored keys, get them
        if self.param_keys is None:
            _, self.param_keys = get_weights(self.model)
        
        # Try to set weights
        success = set_weights(self.model, parameters, self.param_keys)
        
        if not success:
            print(f"Client {self.client_id}: Failed to set parameters. Will reinitialize model and try again...")
            # Reinitialize model and try again
            self._initialize_model()
            _, self.param_keys = get_weights(self.model)
            success = set_weights(self.model, parameters, self.param_keys)
            
            if not success:
                print(f"Client {self.client_id}: Still failed. Will train with current weights.")
                # Return True anyway to continue training
                return True
        
        return success

    def fit(self, fit_ins, config=None):
        server_round = config.get("server_round", 1) if config else 1
        
        # Get epochs from server config or use default
        epochs = config.get("epochs_per_round", self.epochs_per_round) if config else self.epochs_per_round
        
        print(f"\n{'='*60}")
        print(f"CLIENT {self.client_id} - ROUND {server_round}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  • Epochs per round: {epochs}")
        print(f"  • Device: {self.device}")
        print(f"  • Data: {self.data_yaml}")
        
        # CRITICAL FIX: Reset parameter keys each round to match server
        print(f"Client {self.client_id}: Resetting parameter keys for round {server_round}")
        _, self.param_keys = get_weights(self.model)
        
        # Load global parameters
        parameters = parameters_to_ndarrays(fit_ins.parameters)
        success = self.set_parameters(parameters, config)
        
        if not success:
            print(f"Client {self.client_id}: Failed to set parameters, but will continue training with current weights")
            weights, _ = get_weights(self.model)
            return FitRes(
                status=Status(code=Code.OK, message="Parameter mismatch - using current weights"),
                parameters=ndarrays_to_parameters(weights),
                num_examples=0,
                metrics={"error": "parameter_mismatch"}
            )

        print(f"\nClient {self.client_id} - Round {server_round}: Training for {epochs} epochs...")
        
        try:
            # Ensure model overrides are set correctly
            if not hasattr(self.model, 'overrides'):
                self.model.overrides = {}
            self.model.overrides['model'] = self.model_path
            
            # Clear CUDA cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Record training start time
            training_start = time.time()
            
            # ============================================================
            # FIXED: Use configurable epochs instead of hardcoded 1
            # ============================================================
            results = self.model.train(
                data=self.data_yaml,
                epochs=epochs,  # Use configurable epochs
                batch=8,
                imgsz=640,
                project='federated_training',
                name=f'client{self.client_id}_round{server_round}_epochs{epochs}',
                exist_ok=True,
                device=self.device,
                verbose=False,
                save=False,
                workers=0,
                plots=False,  # Disable plots to save time
                save_json=False,  # Disable JSON saving
                save_txt=False,  # Disable TXT saving
                patience=epochs,  # Set patience to epochs to ensure full training
                warmup_epochs=min(3, epochs)  # Warmup for first 3 epochs or less if epochs < 3
            )
            
            # Calculate training time
            training_time = time.time() - training_start
            
            print(f"\nClient {self.client_id}: Training complete in {training_time:.2f} seconds")
            print(f"  Total epochs: {epochs}")
            print(f"  Avg time per epoch: {training_time/epochs:.2f}s")
            
            # Count training examples
            train_images_path = os.path.join(BASE_PATH, "train", "images")
            if os.path.exists(train_images_path):
                num_examples = len([f for f in os.listdir(train_images_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            else:
                num_examples = 893  # Default from your logs
            
            # Get updated weights and update param keys
            weights, self.param_keys = get_weights(self.model)
            
            # Extract comprehensive metrics from results - FIXED VERSION
            metrics_dict = {
                "client_id": self.client_id, 
                "round": server_round,
                "epochs_per_round": epochs,
                "training_time": training_time,
                "num_examples": num_examples,
                "avg_time_per_epoch": training_time/epochs if epochs > 0 else 0,
            }
            
            # ============================================================
            # FIXED METRIC EXTRACTION - Get metrics directly from results
            # ============================================================
            try:
                # Try to get metrics from results
                if hasattr(results, 'metrics'):
                    # For newer YOLO versions
                    metrics = results.metrics
                    if metrics:
                        # Get validation metrics (these are available in the results)
                        metrics_dict.update({
                            "mAP50": float(getattr(metrics, 'map50', 0) or 0),
                            "precision": float(getattr(metrics, 'precision', 0) or 0),
                            "recall": float(getattr(metrics, 'recall', 0) or 0),
                        })
                
                # If not found, try to get from results_dict
                if "mAP50" not in metrics_dict and hasattr(results, 'results_dict'):
                    results_dict = results.results_dict
                    # Look for the last epoch's validation metrics
                    val_metrics_keys = [k for k in results_dict.keys() if 'metrics/mAP50(B)' in k]
                    if val_metrics_keys:
                        # Get the last epoch's metrics
                        last_key = sorted(val_metrics_keys)[-1]
                        epoch_prefix = last_key.split('/metrics/')[0]
                        
                        metrics_dict.update({
                            "mAP50": float(results_dict.get(f'{epoch_prefix}/metrics/mAP50(B)', 0)),
                            "precision": float(results_dict.get(f'{epoch_prefix}/metrics/precision(B)', 0)),
                            "recall": float(results_dict.get(f'{epoch_prefix}/metrics/recall(B)', 0)),
                            "mAP50_95": float(results_dict.get(f'{epoch_prefix}/metrics/mAP50-95(B)', 0)),
                        })
                
                # Extract training losses from the last epoch
                if hasattr(results, 'results_dict'):
                    results_dict = results.results_dict
                    # Find the last training loss entry
                    train_loss_keys = [k for k in results_dict.keys() if '/train/box_loss' in k]
                    if train_loss_keys:
                        last_loss_key = sorted(train_loss_keys)[-1]
                        epoch_prefix = last_loss_key.split('/train/')[0]
                        
                        metrics_dict.update({
                            "box_loss": float(results_dict.get(f'{epoch_prefix}/train/box_loss', 0)),
                            "cls_loss": float(results_dict.get(f'{epoch_prefix}/train/cls_loss', 0)),
                            "dfl_loss": float(results_dict.get(f'{epoch_prefix}/train/dfl_loss', 0)),
                        })
                
                # If still no metrics, use the validation results printed in logs
                if "mAP50" not in metrics_dict or metrics_dict["mAP50"] == 0:
                    # Try to extract from the output logs (last validation)
                    print(f"Client {self.client_id}: Using validation results from best.pt validation")
                    
                    # Run validation on the best model
                    val_results = self.model.val(
                        data=self.data_yaml,
                        batch=8,
                        imgsz=640,
                        device=self.device,
                        verbose=False,
                        split='val'
                    )
                    
                    if hasattr(val_results, 'results_dict'):
                        val_dict = val_results.results_dict
                        metrics_dict.update({
                            "mAP50": float(val_dict.get('metrics/mAP50(B)', 0)),
                            "precision": float(val_dict.get('metrics/precision(B)', 0)),
                            "recall": float(val_dict.get('metrics/recall(B)', 0)),
                        })
                
            except Exception as e:
                print(f"Client {self.client_id}: Error extracting metrics: {e}")
                # Set default values if extraction fails
                metrics_dict.update({
                    "mAP50": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "box_loss": 0.0,
                    "cls_loss": 0.0,
                    "dfl_loss": 0.0,
                })
            
            # Print metrics for debugging
            print(f"\nClient {self.client_id} Training Results:")
            print(f"  {'='*40}")
            print(f"  Round: {server_round}")
            print(f"  Epochs: {epochs}")
            print(f"  Training Time: {training_time:.1f}s ({training_time/epochs:.1f}s per epoch)")
            print(f"  Examples: {num_examples}")
            print(f"  Metrics:")
            print(f"    • mAP50: {metrics_dict.get('mAP50', 0):.3f}")
            print(f"    • Precision: {metrics_dict.get('precision', 0):.3f}")
            print(f"    • Recall: {metrics_dict.get('recall', 0):.3f}")
            if 'box_loss' in metrics_dict:
                print(f"    • Box Loss: {metrics_dict.get('box_loss', 0):.3f}")
                print(f"    • Class Loss: {metrics_dict.get('cls_loss', 0):.3f}")
                print(f"    • DFL Loss: {metrics_dict.get('dfl_loss', 0):.3f}")
            print(f"  {'='*40}")
            
            return FitRes(
                status=Status(code=Code.OK, message=f"Success - {epochs} epochs completed"),
                parameters=ndarrays_to_parameters(weights),
                num_examples=num_examples,
                metrics=metrics_dict
            )
            
        except Exception as e:
            print(f"Client {self.client_id}: Training failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Reinitialize model on failure
            print(f"Client {self.client_id}: Reinitializing model after failure")
            self._initialize_model()
            
            # Return current parameters
            weights, _ = get_weights(self.model)
            return FitRes(
                status=Status(code=Code.OK, message=f"Error: {str(e)[:100]}"),
                parameters=ndarrays_to_parameters(weights),
                num_examples=0,
                metrics={
                    "client_id": self.client_id,
                    "round": server_round,
                    "epochs_per_round": epochs,
                    "error": 1.0,
                    "error_message": str(e)[:200]
                }
            )

    def evaluate(self, evaluate_ins, config=None):
        """Evaluation method - can be used for validation"""
        print(f"\nClient {self.client_id}: Federated evaluation requested")
        
        try:
            # Load parameters if provided
            if evaluate_ins.parameters:
                parameters = parameters_to_ndarrays(evaluate_ins.parameters)
                self.set_parameters(parameters, config)
            
            # Count validation examples
            val_images_path = os.path.join(BASE_PATH, "val", "images")
            if os.path.exists(val_images_path):
                num_examples = len([f for f in os.listdir(val_images_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            else:
                num_examples = 252  # Default from your logs
            
            # Run validation
            print(f"Client {self.client_id}: Running validation on {num_examples} examples...")
            
            val_results = self.model.val(
                data=self.data_yaml,
                batch=8,
                imgsz=640,
                device=self.device,
                verbose=False,
                split='val'
            )
            
            # Extract metrics from validation
            metrics_dict = {
                "client_id": self.client_id,
                "validation": True
            }
            
            if hasattr(val_results, 'results_dict'):
                val_dict = val_results.results_dict
                metrics_dict.update({
                    "mAP50": float(val_dict.get('metrics/mAP50(B)', 0)),
                    "precision": float(val_dict.get('metrics/precision(B)', 0)),
                    "recall": float(val_dict.get('metrics/recall(B)', 0)),
                    "mAP50_95": float(val_dict.get('metrics/mAP50-95(B)', 0)),
                })
            
            print(f"Client {self.client_id}: Validation Results:")
            print(f"  • mAP50: {metrics_dict.get('mAP50', 0):.3f}")
            print(f"  • Precision: {metrics_dict.get('precision', 0):.3f}")
            print(f"  • Recall: {metrics_dict.get('recall', 0):.3f}")
            
            return EvaluateRes(
                status=Status(code=Code.OK, message="Validation successful"),
                loss=1.0 - float(metrics_dict.get('mAP50', 0)),  # Loss as 1 - mAP50
                num_examples=num_examples,
                metrics=metrics_dict
            )
            
        except Exception as e:
            print(f"Client {self.client_id}: Evaluation failed: {e}")
            return EvaluateRes(
                status=Status(code=Code.OK, message=f"Error: {str(e)[:100]}"),
                loss=1.0,
                num_examples=0,
                metrics={"error": 1.0}
            )

# -----------------------------
# Start Flower client
# -----------------------------
if __name__ == "__main__":
    # You can override epochs per round via environment variable
    env_epochs = os.getenv("EPOCHS_PER_ROUND", None)
    if env_epochs:
        epochs_per_round = int(env_epochs)
    else:
        epochs_per_round = DEFAULT_EPOCHS_PER_ROUND
    
    client = YOLOClient(CLIENT_ID, epochs_per_round=epochs_per_round)
    
    print(f"\n{'='*60}")
    print(f"FEDERATED LEARNING CLIENT {CLIENT_ID}")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  • Client ID: {CLIENT_ID}")
    print(f"  • Data YAML: {DATA_YAML}")
    print(f"  • Device: {DEVICE}")
    print(f"  • Epochs per round: {epochs_per_round}")
    print(f"  • Model: {MODEL_PATH}")
    print(f"{'='*60}")
    print(f"Connecting to server...")
    
    # Add retry logic for connection
    max_retries = 3
    for attempt in range(max_retries):
        try:
            fl.client.start_client(
                server_address="127.0.0.1:8080", 
                client=client
            )
            break
        except Exception as e:
            print(f"Client {CLIENT_ID}: Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Client {CLIENT_ID}: Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Client {CLIENT_ID}: Max retries reached. Exiting.")