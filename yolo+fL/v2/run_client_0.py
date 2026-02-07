import os
import flwr as fl

# Set environment variables
os.environ["CLIENT_ID"] = "0"

# Import the client class from client.py
import sys
sys.path.append('.')  # Add current directory to path

# Now create the client
from client import YOLOClient

if __name__ == "__main__":
    # Create client with ID from environment variable
    CLIENT_ID = int(os.getenv("CLIENT_ID", 0))
    client = YOLOClient(CLIENT_ID, epochs_per_round=5)
    
    print(f"Starting Client {CLIENT_ID}...")
    
    fl.client.start_client(
        server_address="localhost:8080",
        client=client
    )