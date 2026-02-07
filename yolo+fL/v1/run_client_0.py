import os
import flwr as fl
from yolo_client import YOLOClient

if __name__ == "__main__":
    model_path = "yolo11n.pt"
    data_yaml = "fl_results/client_0/data.yaml"

    client = YOLOClient(model_path, data_yaml)

    server_ip = os.getenv("SERVER_IP", "localhost")
    server_port = os.getenv("SERVER_PORT", "8080")

    fl.client.start_client(
        server_address=f"{server_ip}:{server_port}",
        client=client
    )
