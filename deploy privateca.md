# Flower Federated Learning with TLS Security

This repository provides a step-by-step guide to setting up a basic Flower federated learning system with TLS security on AWS EC2 instances. The setup involves three EC2 instances: one for a private Certificate Authority (CA) server, one for the Flower server, and one for the Flower client.

## Prerequisites

- Three EC2 instances: one for CA server, one for Flower server, and one for Flower client.
- Python environment set up on both the Flower server and client instances.
- Necessary packages installed: `flwr`, `numpy`, `logging`, `pathlib`.

## Directory Structure

```
.
├── ca_server
│   ├── generate.sh
│   ├── fl_server.conf
│   ├── fl_client.conf
│   ├── ca.key
│   ├── ca.crt
│   ├── fl_server.key
│   ├── fl_server.csr
│   ├── fl_server.crt
│   ├── fl_client.key
│   ├── fl_client.csr
│   ├── fl_client.crt
├── flower_server
│   ├── basic_server.py
│   ├── certificates
│   │   ├── ca.crt
│   │   ├── fl_server.key
│   │   ├── fl_server.crt
├── flower_client
│   ├── basic_client.py
│   ├── certificates
│   │   ├── ca.crt
│   ├── keys
│   │   ├── client_credentials_1
│   │   ├── client_credentials_1.pub
```

## Step-by-Step Instructions

### 1. Setting Up the Private CA Server

**On the CA Server Instance:**

1. **Install OpenSSL**:
    ```bash
    sudo apt update
    sudo apt install openssl
    ```

2. **Generate CA Key and Certificate**:
    ```bash
    mkdir -p ~/ca
    cd ~/ca
    openssl genrsa -out ca.key 4096
    openssl req -new -x509 -key ca.key -sha256 -subj "/C=US/ST=New York/O=ScriptChain/OU=AI/CN=PrivateCA" -days 365 -out ca.crt
    ```

### 2. Generating and Signing Certificates for the Flower Server

**On the CA Server Instance:**

1. **Create a Configuration File for the Flower Server**:
    ```bash
    nano fl_server.conf
    ```

    **Configuration File Content**:
    ```ini
    [req]
    default_bits = 2048
    encrypt_key = no
    default_keyfile = FLServer.key
    default_md = sha1
    prompt = no
    req_extensions = req_ext
    distinguished_name = dn

    [dn]
    C = US
    ST = New York
    O = ScriptChain
    OU = AI
    CN = FLServer

    [req_ext]
    basicConstraints=CA:FALSE
    subjectAltName = @alt_names
    subjectKeyIdentifier = hash

    [alt_names]
    DNS.1 = localhost
    IP.1 = ::1
    IP.2 = 127.0.0.1
    IP.3 = <FL_SERVER_PRIVATE_IP>
    IP.4 = <FL_SERVER_PUBLIC_IP>
    ```

2. **Generate Server Key and CSR**:
    ```bash
    openssl genrsa -out fl_server.key 2048
    openssl req -new -key fl_server.key -out fl_server.csr -config fl_server.conf
    ```

3. **Sign the Server Certificate with the CA**:
    ```bash
    openssl x509 -req -in fl_server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out fl_server.crt -days 365 -sha256 -extfile fl_server.conf -extensions req_ext
    ```

4. **Transfer the Certificates to the Flower Server**:
    ```bash
    scp ca.crt fl_server.key fl_server.crt ubuntu@<FL_SERVER_PUBLIC_IP>:/home/ubuntu/
    ```

### 3. Setting Up the Flower Server

**On the Flower Server Instance:**

1. **Install Dependencies**:
    ```bash
    sudo apt update
    sudo apt install python3-pip
    pip3 install flwr numpy
    ```

2. **Move Certificates to Secure Directory**:
    ```bash
    sudo mkdir -p /etc/ssl/flower
    sudo mv ~/ca.crt /etc/ssl/flower/
    sudo mv ~/fl_server.key /etc/ssl/flower/
    sudo mv ~/fl_server.crt /etc/ssl/flower/
    ```

3. **Create the Server Script**:
    ```bash
    nano basic_server.py
    ```

    **Server Script Content**:
    ```python
    import flwr as fl
    from pathlib import Path
    import logging

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    def main():
        logger.info("Starting Flower server...")

        # Create a Flower server with a simple strategy
        strategy = fl.server.strategy.FedAvg(min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1)

        # Start Flower server (SSL-enabled)
        fl.server.start_server(
            server_address="0.0.0.0:9092",
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=strategy,
            certificates=(
                Path("/etc/ssl/flower/ca.crt").read_bytes(),
                Path("/etc/ssl/flower/fl_server.crt").read_bytes(),
                Path("/etc/ssl/flower/fl_server.key").read_bytes(),
            ),
        )
        logger.info("Flower server started successfully.")

    if __name__ == "__main__":
        main()
    ```

4. **Run the Server Script**:
    ```bash
    python3 basic_server.py
    ```

### 4. Generating and Signing Certificates for the Flower Client

**On the CA Server Instance:**

1. **Create a Configuration File for the Flower Client**:
    ```bash
    nano fl_client.conf
    ```

    **Configuration File Content**:
    ```ini
    [req]
    default_bits = 2048
    encrypt_key = no
    default_keyfile = FLClient.key
    default_md = sha1
    prompt = no
    req_extensions = req_ext
    distinguished_name = dn

    [dn]
    C = US
    ST = New York
    O = ScriptChain
    OU = AI
    CN = FLClient

    [req_ext]
    basicConstraints=CA:FALSE
    subjectAltName = @alt_names
    subjectKeyIdentifier = hash

    [alt_names]
    DNS.1 = localhost
    IP.1 = ::1
    IP.2 = 127.0.0.1
    IP.3 = <FL_CLIENT_PRIVATE_IP>
    IP.4 = <FL_CLIENT_PUBLIC_IP>
    ```

2. **Generate Client Key and CSR**:
    ```bash
    openssl genrsa -out fl_client.key 2048
    openssl req -new -key fl_client.key -out fl_client.csr -config fl_client.conf
    ```

3. **Sign the Client Certificate with the CA**:
    ```bash
    openssl x509 -req -in fl_client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out fl_client.crt -days 365 -sha256 -extfile fl_client.conf -extensions req_ext
    ```

4. **Transfer the Certificates to the Flower Client**:
    ```bash
    scp ca.crt fl_client.key fl_client.crt ubuntu@<FL_CLIENT_PUBLIC_IP>:/home/ubuntu/

### 5. Setting Up the Flower Client

**On the Flower Client Instance:**

1. **Install Dependencies**:
    ```bash
    sudo apt update
    sudo apt install python3-pip
    pip3 install flwr numpy
    ```

2. **Move Certificates to Secure Directory**:
    ```bash
    sudo mkdir -p /etc/ssl/flower
    sudo mv ~/ca.crt /etc/ssl/flower/
    sudo mv ~/fl_client.key /etc/ssl/flower/
    sudo mv ~/fl_client.crt /etc/ssl/flower/
    ```

3. **Create the Client Script**:
    ```bash
    nano basic_client.py
    ```

    **Client Script Content**:
    ```python
    import numpy as np
    import flwr as fl
    import logging
    from pathlib import Path
    from typing import Dict, Tuple

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    class BasicClient(fl.client.NumPyClient):
        def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> fl.common.NDArrays:
            logger.info("Client: get_parameters")
            return np.array([0.1], dtype=np.float32)

        def fit(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[fl.common.NDArrays, int, Dict[str, fl.common.Scalar]]:
            logger.info("Client: fit")
            # Perform a mock "training" step
            new_parameters = np.array([0.2], dtype=np.float32)
            num_examples_train = 1  # Mock number of training examples
            metrics = {"accuracy": 0.9}  # Mock metric
            return new_parameters, num_examples_train, metrics

        def evaluate(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
            logger.info("Client: evaluate")
            # Perform a mock "evaluation" step
            loss = 0.1  # Mock loss
            num_examples_test = 1  # Mock number of test examples
            metrics = {"accuracy": 0.9}  # Mock metric
            return loss, num_examples_test, metrics

    def main():
        # Initialize client
        client = BasicClient()
        flower_client = client.to_client()

        # Start Flower client
        fl.client.start_client(
            server_address="172.31.24.144:9092",
            client=flower_client,
            root_certificates=Path("/etc/ssl/flower/ca.crt").read_bytes(),
        )

    if __name__ == "__main__":
        main()
    ```

4. **Run the Client Script**:
    ```bash
    python3 basic_client.py
    ```

### Verifying TLS Connection

To verify that the TLS connection is secure and working properly, you can use `openssl` to connect to the Flower server and check the certificates.

**On the Flower Client Instance:**

1. **Test TLS Connection**:
    ```bash
    openssl s_client -connect 172.31.24.144:9092 -CAfile /etc/ssl/flower/ca.crt
    ```

The output should indicate a successful TLS handshake with the correct server certificate details. If the verification is successful, you will see `Verify return code: 0 (ok)`.

### Summary

You have now successfully set up a Flower federated learning system with TLS security on AWS EC2 instances. The steps included setting up a private CA server to generate and sign certificates, configuring the Flower server with these certificates, and finally, setting up a Flower client to securely communicate with the Flower server.
