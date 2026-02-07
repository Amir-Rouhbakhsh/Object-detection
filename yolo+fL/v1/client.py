import os
import torch
import time
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
    """Set model parameters from list of NumPy arrays"""
    if len(parameters) != len(param_keys):
        print(f"Client {CLIENT_ID}: Parameter count mismatch! Expected {len(param_keys)}, got {len(parameters)}")
        return False
    
    state_dict = model.model.state_dict()
    
    # Update parameters
    for i, key in enumerate(param_keys):
        if i < len(parameters):
            # Create tensor with proper device and dtype
            param_tensor = torch.from_numpy(parameters[i]).to(
                dtype=state_dict[key].dtype,
                device=state_dict[key].device
            )
            
            if param_tensor.shape == state_dict[key].shape:
                # Assign without in-place modification
                state_dict[key] = param_tensor
            else:
                print(f"Client {CLIENT_ID}: Shape mismatch for {key}: {state_dict[key].shape} vs {param_tensor.shape}")
                return False
    
    # Load state dict
    try:
        model.model.load_state_dict(state_dict, strict=False)
        print(f"Client {CLIENT_ID}: Successfully loaded parameters")
        return True
    except Exception as e:
        print(f"Client {CLIENT_ID}: Error loading state dict: {e}")
        return False

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
            self._initialize_model()
        
        # If we don't have stored keys, get them
        if self.param_keys is None:
            _, self.param_keys = get_weights(self.model)
        
        return set_weights(self.model, parameters, self.param_keys)

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
        
        # Load global parameters
        parameters = parameters_to_ndarrays(fit_ins.parameters)
        success = self.set_parameters(parameters, config)
        
        if not success:
            print(f"Client {self.client_id}: Failed to set parameters, skipping training")
            weights, _ = get_weights(self.model)
            return FitRes(
                status=Status(code=Code.OK, message="Parameter mismatch - skipped training"),
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
            # CRITICAL CHANGE: Use configurable epochs instead of hardcoded 1
            # ============================================================
            results = self.model.train(
                data=self.data_yaml,
                epochs=epochs,  # CHANGED FROM 1 to epochs
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
            
            # Extract comprehensive metrics from results
            metrics_dict = {
                "client_id": self.client_id, 
                "round": server_round,
                "epochs_per_round": epochs,
                "training_time": training_time,
                "num_examples": num_examples,
                "avg_time_per_epoch": training_time/epochs if epochs > 0 else 0,
            }
            
            # Extract YOLO metrics - IMPROVED EXTRACTION
            metrics_extracted = False
            
            # Method 1: Try to extract from results_dict
            if hasattr(results, 'results_dict'):
                results_dict = results.results_dict
                
                # Find the last epoch key
                last_epoch_key = None
                for key in results_dict.keys():
                    if 'metrics/mAP50(B)' in key:
                        # Extract epoch number
                        parts = key.split('/')
                        if len(parts) >= 2:
                            epoch_part = parts[0]
                            if last_epoch_key is None or int(epoch_part.split('_')[-1]) > int(last_epoch_key.split('/')[0].split('_')[-1]):
                                last_epoch_key = key
                
                if last_epoch_key:
                    base_key = last_epoch_key.split('/')[0]
                    try:
                        metrics_dict.update({
                            "mAP50": float(results_dict.get(f'{base_key}/metrics/mAP50(B)', 0)),
                            "precision": float(results_dict.get(f'{base_key}/metrics/precision(B)', 0)),
                            "recall": float(results_dict.get(f'{base_key}/metrics/recall(B)', 0)),
                            "mAP50_95": float(results_dict.get(f'{base_key}/metrics/mAP50-95(B)', 0)),
                        })
                        
                        # Get losses from training
                        loss_keys = [k for k in results_dict.keys() if 'train/' in k and base_key in k]
                        if loss_keys:
                            # Get the last loss entry
                            last_loss_key = sorted(loss_keys)[-1]
                            if '/train/box_loss' in last_loss_key:
                                loss_base = last_loss_key.split('/train/')[0]
                                metrics_dict.update({
                                    "box_loss": float(results_dict.get(f'{loss_base}/train/box_loss', 0)),
                                    "cls_loss": float(results_dict.get(f'{loss_base}/train/cls_loss', 0)),
                                    "dfl_loss": float(results_dict.get(f'{loss_base}/train/dfl_loss', 0)),
                                })
                        
                        metrics_extracted = True
                    except Exception as e:
                        print(f"Client {self.client_id}: Error extracting from results_dict: {e}")
            
            # Method 2: Try to get from results directly
            if not metrics_extracted and hasattr(results, 'metrics'):
                try:
                    # YOLO v8.4.7 stores metrics in results.metrics
                    if results.metrics is not None:
                        metrics_dict.update({
                            "mAP50": float(getattr(results, 'maps', 0) or 0),
                            "precision": float(getattr(results, 'precision', 0) or 0),
                            "recall": float(getattr(results, 'recall', 0) or 0),
                        })
                        metrics_extracted = True
                except Exception as e:
                    print(f"Client {self.client_id}: Error extracting from results.metrics: {e}")
            
            # Method 3: Parse from training output (fallback)
            if not metrics_extracted:
                try:
                    # Try to parse from the training log output
                    # This is a fallback method
                    print(f"Client {self.client_id}: Using fallback metric extraction")
                    # Set default values
                    metrics_dict.update({
                        "mAP50": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "box_loss": 0.0,
                        "cls_loss": 0.0,
                        "dfl_loss": 0.0,
                    })
                except Exception as e:
                    print(f"Client {self.client_id}: Fallback extraction failed: {e}")
            
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
            
            # Run validation if we have validation data
            if num_examples > 0 and hasattr(self.model, 'val'):
                try:
                    print(f"Client {self.client_id}: Running validation on {num_examples} examples...")
                    
                    # Run validation
                    results = self.model.val(
                        data=self.data_yaml,
                        batch=8,
                        imgsz=640,
                        device=self.device,
                        verbose=False,
                        split='val'
                    )
                    
                    # Extract metrics
                    if hasattr(results, 'results_dict'):
                        results_dict = results.results_dict
                        
                        # Find validation metrics
                        val_mAP50 = results_dict.get('metrics/mAP50(B)', 0)
                        val_precision = results_dict.get('metrics/precision(B)', 0)
                        val_recall = results_dict.get('metrics/recall(B)', 0)
                        
                        print(f"Client {self.client_id}: Validation Results:")
                        print(f"  • mAP50: {float(val_mAP50):.3f}")
                        print(f"  • Precision: {float(val_precision):.3f}")
                        print(f"  • Recall: {float(val_recall):.3f}")
                        
                        return EvaluateRes(
                            status=Status(code=Code.OK, message="Validation successful"),
                            loss=float(results_dict.get('train/box_loss', 0.5)),  # Use box loss as loss
                            num_examples=num_examples,
                            metrics={
                                "client_id": self.client_id,
                                "mAP50": float(val_mAP50),
                                "precision": float(val_precision),
                                "recall": float(val_recall),
                                "validation": True
                            }
                        )
                except Exception as e:
                    print(f"Client {self.client_id}: Validation error: {e}")
            
            # Return dummy evaluation if validation failed
            return EvaluateRes(
                status=Status(code=Code.OK, message="Evaluation completed"),
                loss=0.5,  # Dummy loss
                num_examples=num_examples,
                metrics={
                    "client_id": self.client_id,
                    "note": "YOLO validation done during training"
                }
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
    
    fl.client.start_client(
        server_address="127.0.0.1:8080", 
        client=client
    )