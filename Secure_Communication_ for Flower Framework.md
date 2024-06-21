# Federated Learning Setup with mTLS on AWS EC2

This guide provides step-by-step instructions to set up a secure federated learning environment on AWS EC2 instances using the Flower framework with mutual TLS (mTLS) encryption.

## Prerequisites

- AWS account with EC2 instances set up
  - One EC2 instance for the FL server
  - Two EC2 instances for the FL clients
  - One EC2 instance for the private CA server
- Domain name pointing to the public IP of the FL server (e.g., scriptchain.co)
- OpenSSL installed on the private CA server
- Python and necessary libraries installed on both FL server and clients

## Directory Structure

```plaintext
/etc/ssl/fl_network
├── ca.crt
├── ca.key
├── server.crt
├── server.csr
├── server.key
├── client1.crt
├── client1.csr
├── client1.key
├── client2.crt
├── client2.csr
├── client2.key
└── certificate.conf
```

## Configuration File (`certificate.conf`)

```ini
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

#### Create the Configuration File

Create the configuration file `certificate.conf` as shown above.

#### Generate CA Key and Certificate

```bash
openssl genpkey -algorithm RSA -out /etc/ssl/fl_network/ca.key -pkeyopt rsa_keygen_bits:2048
openssl req -new -x509 -key /etc/ssl/fl_network/ca.key -out /etc/ssl/fl_network/ca.crt -days 730 -config /etc/ssl/fl_network/certificate.conf -extensions req_ext
```

#### Generate Server Key, CSR, and Certificate

```bash
openssl genpkey -algorithm RSA -out /etc/ssl/fl_network/server.key -pkeyopt rsa_keygen_bits:2048
openssl req -new -key /etc/ssl/fl_network/server.key -out /etc/ssl/fl_network/server.csr -config /etc/ssl/fl_network/certificate.conf
openssl x509 -req -in /etc/ssl/fl_network/server.csr -CA /etc/ssl/fl_network/ca.crt -CAkey /etc/ssl/fl_network/ca.key -CAcreateserial -out /etc/ssl/fl_network/server.crt -days 730 -extensions req_ext -extfile /etc/ssl/fl_network/certificate.conf
```

#### Generate Client Keys, CSRs, and Certificates

Repeat the following commands for `client1` and `client2`:

For `client1`:

```bash
openssl genpkey -algorithm RSA -out /etc/ssl/fl_network/client1.key -pkeyopt rsa_keygen_bits:2048
openssl req -new -key /etc/ssl/fl_network/client1.key -out /etc/ssl/fl_network/client1.csr -subj "/C=US/ST=New York/O=ScriptChain/OU=AI/CN=FLClient1"
openssl x509 -req -in /etc/ssl/fl_network/client1.csr -CA /etc/ssl/fl_network/ca.crt -CAkey /etc/ssl/fl_network/ca.key -CAcreateserial -out /etc/ssl/fl_network/client1.crt -days 730 -extensions req_ext -extfile /etc/ssl/fl_network/certificate.conf
```

For `client2`:

```bash
openssl genpkey -algorithm RSA -out /etc/ssl/fl_network/client2.key -pkeyopt rsa_keygen_bits:2048
openssl req -new -key /etc/ssl/fl_network/client2.key -out /etc/ssl/fl_network/client2.csr -subj "/C=US/ST=New York/O=ScriptChain/OU=AI/CN=FLClient2"
openssl x509 -req -in /etc/ssl/fl_network/client2.csr -CA /etc/ssl/fl_network/ca.crt -CAkey /etc/ssl/fl_network/ca.key -CAcreateserial -out /etc/ssl/fl_network/client2.crt -days 730 -extensions req_ext -extfile /etc/ssl/fl_network/certificate.conf
```

### 2. Transfer Certificates to FL Server and Clients Using SCP

#### Transfer to FL Server

On the CA server, use SCP to transfer the necessary files to the FL server:

```bash
scp /etc/ssl/fl_network/ca.crt ubuntu@FL_SERVER_IP:/etc/ssl/fl_network/
scp /etc/ssl/fl_network/server.crt ubuntu@FL_SERVER_IP:/etc/ssl/fl_network/
scp /etc/ssl/fl_network/server.key ubuntu@FL_SERVER_IP:/etc/ssl/fl_network/
```

#### Transfer to FL Clients

On the CA server, use SCP to transfer the necessary files to each FL client:

For `client1`:

```bash
scp /etc/ssl/fl_network/ca.crt ubuntu@CLIENT1_IP:/etc/ssl/fl_network/
scp /etc/ssl/fl_network/client1.crt ubuntu@CLIENT1_IP:/etc/ssl/fl_network/
scp /etc/ssl/fl_network/client1.key ubuntu@CLIENT1_IP:/etc/ssl/fl_network/
```

For `client2`:

```bash
scp /etc/ssl/fl_network/ca.crt ubuntu@CLIENT2_IP:/etc/ssl/fl_network/
scp /etc/ssl/fl_network/client2.crt ubuntu@CLIENT2_IP:/etc/ssl/fl_network/
scp /etc/ssl/fl_network/client2.key ubuntu@CLIENT2_IP:/etc/ssl/fl_network/
```

### 3. Deploy Server Code on FL Server EC2 Instance

**Server Code (`server.py`):**

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

strategy = SaveModelStrategy(min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)

fl.server.start_server(
    server_address="0.0.0.0:9092",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=3),
    certificates=(
        Path("/etc/ssl/fl_network/ca.crt").read_bytes(),
        Path("/etc/ssl/fl_network/server.crt").read_bytes(),
        Path("/etc/ssl/fl_network/server.key").read_bytes(),
    ),
)
```

### 4. Deploy Client Code on FL Client EC2 Instances

**Client Code (`client.py`):**

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
        root_certificates=open("/etc/ssl/fl_network/ca.crt", "rb").read(),
    )

if __name__ == "__main__":
    main()
```

### 5. Set Permissions

Ensure the necessary permissions are set correctly on both the server and clients:

```bash
sudo chown -R ubuntu:ubuntu /etc/ssl/fl_network
sudo chmod -R 755 /etc/ssl/fl_network
```

### 6. Verify Certificates

Verify the server certificate from a client:

```bash
openssl s_client -connect scriptchain.co:9092 -CAfile /etc/ssl/fl_network/ca.crt
```

## Summary

These instructions guide you through setting up a secure federated learning environment with mutual TLS authentication using the Flower framework. Ensure that each client is correctly configured to use its own certificate for authentication. This setup will use certificates with a validity period of 2 years.

By following these steps, you can deploy a secure federated learning setup on AWS EC2 with the Flower framework, using mTLS to ensure secure communication between the server and clients.
