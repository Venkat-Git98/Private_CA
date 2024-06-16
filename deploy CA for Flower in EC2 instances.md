# Federated Learning Setup with TLS on AWS EC2

This guide provides step-by-step instructions to set up a secure federated learning environment on AWS EC2 instances using the Flower framework with TLS encryption.

## Prerequisites

- AWS account with EC2 instances set up
- One EC2 instance for the FL server
- One EC2 instance for the FL client
- One EC2 instance for the private CA server
- Domain name pointing to the public IP of the FL server (e.g., scriptchain.co)
- OpenSSL installed on the private CA server
- Python and necessary libraries installed on both FL server and client

## Directory Structure

```
/etc/ssl/flower
├── ca.crt
├── ca.key
├── server.crt
├── server.csr
├── server.key
├── client.crt
├── client.csr
├── client.key
└── certificate.conf
```

## Configuration File (`certificate.conf`)

```
[req]
default_bits = 2048
encrypt_key = no
default_keyfile = FLServer.key
default_md = sha256
prompt = no
req_extensions = req_ext
distinguished_name = dn

[dn]
C = US
ST = New York
O = ScriptChain
OU = AI
CN = scriptchain.co

[req_ext]
basicConstraints = CA:FALSE
subjectAltName = @alt_names
subjectKeyIdentifier = hash

[alt_names]
DNS.1 = scriptchain.co
IP.1 = 100.27.120.77
IP.2 = 172.31.24.144
```

## Step-by-Step Instructions

### 1. Generate Certificates on Private CA Server

#### Generate CA Key and Certificate

```bash
openssl genpkey -algorithm RSA -out /etc/ssl/flower/ca.key -pkeyopt rsa_keygen_bits:2048
openssl req -new -x509 -key /etc/ssl/flower/ca.key -out /etc/ssl/flower/ca.crt -days 365 -subj "/C=US/ST=New York/O=ScriptChain/OU=AI/CN=PrivateCA"
```

#### Generate Server Key, CSR, and Certificate

```bash
openssl genpkey -algorithm RSA -out /etc/ssl/flower/server.key -pkeyopt rsa_keygen_bits:2048
openssl req -new -key /etc/ssl/flower/server.key -out /etc/ssl/flower/server.csr -config /etc/ssl/flower/certificate.conf
openssl x509 -req -in /etc/ssl/flower/server.csr -CA /etc/ssl/flower/ca.crt -CAkey /etc/ssl/flower/ca.key -CAcreateserial -out /etc/ssl/flower/server.crt -days 365 -extensions req_ext -extfile /etc/ssl/flower/certificate.conf
```

#### Generate Client Key, CSR, and Certificate

```bash
openssl genpkey -algorithm RSA -out /etc/ssl/flower/client.key -pkeyopt rsa_keygen_bits:2048
openssl req -new -key /etc/ssl/flower/client.key -out /etc/ssl/flower/client.csr -subj "/C=US/ST=New York/O=ScriptChain/OU=AI/CN=FLClient"
openssl x509 -req -in /etc/ssl/flower/client.csr -CA /etc/ssl/flower/ca.crt -CAkey /etc/ssl/flower/ca.key -CAcreateserial -out /etc/ssl/flower/client.crt -days 365
```

### 2. Deploy Server Code on FL Server EC2 Instance

```python
import flwr as fl
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from flwr.common import Scalar
import os

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            os.makedirs("./weights", exist_ok=True)
            np.save(f"./weights/round-{server_round}-weights.npy", aggregated_parameters)
        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy(min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1)

fl.server.start_server(
    server_address="0.0.0.0:9092",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=3),
    certificates=(
        Path("/etc/ssl/flower/ca.crt").read_bytes(),
        Path("/etc/ssl/flower/server.crt").read_bytes(),
        Path("/etc/ssl/flower/server.key").read_bytes(),
    ),
)
```

### 3. Deploy Client Code on FL Client EC2 Instance

```python
import flwr as fl
import numpy as np

class SimpleClient(fl.client.NumPyClient):
    def __init__(self):
        self.weights = np.array([1.0, 2.0, 3.0])

    def get_parameters(self, config):
        return self.weights

    def set_parameters(self, parameters):
        self.weights = parameters

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        return self.get_parameters(config), len(self.weights), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.weights), {"accuracy": 0.9}

def main():
    client = SimpleClient()
    fl.client.start_client(
        server_address="scriptchain.co:9092",
        client=client,
        root_certificates=open("/etc/ssl/flower/ca.crt", "rb").read(),
    )

if __name__ == "__main__":
    main()
```

### Verify SSL/TLS Connection

Run the following command to verify the SSL/TLS connection:

```bash
openssl s_client -connect scriptchain.co:9092 -CAfile /etc/ssl/flower/ca.crt
```

### Move Certificates from CA Server to FL Server and Client

1. **On CA Server:**

   ```bash
   sudo apt update
   sudo apt install awscli -y
   aws configure  # Enter your AWS credentials
   aws s3 sync /etc/ssl/flower s3://your-s3-bucket/flower
   ```

2. **On FL Server and Client:**

   ```bash
   sudo apt update
   sudo apt install awscli -y
   aws configure  # Enter your AWS credentials
   aws s3 sync s3://your-s3-bucket/flower /etc/ssl/flower
   ```

### Set Up Python Environment

1. **On FL Server and Client:**

   ```bash
   sudo apt update
   sudo apt install python3-venv -y
   python3 -m venv ~/myenv
   source ~/myenv/bin/activate
   pip install flwr numpy tensorflow
   ```

### Ensure Proper Permissions

```bash
sudo chmod -R 755 /etc/ssl/flower
sudo find /etc/ssl/flower -type f -exec chmod 644 {} \;
```

By following these steps, you will have set up a secure federated learning environment on AWS EC2 instances using the Flower framework with TLS encryption.
