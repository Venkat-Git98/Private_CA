# Federated Learning Setup with TLS on AWS EC2

This guide provides step-by-step instructions to set up a secure federated learning environment on AWS EC2 instances using Flower framework with TLS encryption.

## Prerequisites

- AWS account with EC2 instances set up
  - One EC2 instance for the FL server
  - One EC2 instance for the FL client
  - One EC2 instance for the private CA server
- Domain name pointing to the public IP of the FL server (e.g., `scriptchain.co`)
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

1. **Generate CA Key and Certificate**

   ```bash
   openssl genpkey -algorithm RSA -out /etc/ssl/flower/ca.key -pkeyopt rsa_keygen_bits:2048
   openssl req -new -x509 -key /etc/ssl/flower/ca.key -out /etc/ssl/flower/ca.crt -days 365 -subj "/C=US/ST=New York/O=ScriptChain/OU=AI/CN=PrivateCA"
   ```

2. **Generate Server Key, CSR, and Certificate**

   ```bash
   openssl genpkey -algorithm RSA -out /etc/ssl/flower/server.key -pkeyopt rsa_keygen_bits:2048
   openssl req -new -key /etc/ssl/flower/server.key -out /etc/ssl/flower/server.csr -config /etc/ssl/flower/certificate.conf
   openssl x509 -req -in /etc/ssl/flower/server.csr -CA /etc/ssl/flower/ca.crt -CAkey /etc/ssl/flower/ca.key -CAcreateserial -out /etc/ssl/flower/server.crt -days 365 -extensions req_ext -extfile /etc/ssl/flower/certificate.conf
   ```

3. **Generate Client Key, CSR, and Certificate**

   ```bash
   openssl genpkey -algorithm RSA -out /etc/ssl/flower/client.key -pkeyopt rsa_keygen_bits:2048
   openssl req -new -key /etc/ssl/flower/client.key -out /etc/ssl/flower/client.csr -subj "/C=US/ST=New York/O=ScriptChain/OU=AI/CN=FLClient"
   openssl x509 -req -in /etc/ssl/flower/client.csr -CA /etc/ssl/flower/ca.crt -CAkey /etc/ssl/flower/ca.key -CAcreateserial -out /etc/ssl/flower/client.crt -days 365
   ```

### 2. Deploy Server Code on FL Server EC2 Instance

1. **Server Script (`server.py`)**

   ```python
   import logging
   import flwr as fl
   import numpy as np
   from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
   import grpc

   # Configure logging
   logging.basicConfig(level=logging.DEBUG)

   class SaveModelStrategy(fl.server.strategy.FedAvg):
       def aggregate_fit(self, server_round, results, failures):
           aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
           if aggregated_parameters is not None:
               print(f"Saving round {server_round} aggregated parameters...")
               aggregated_tensors = [torch.tensor(ndarray) for ndarray in parameters_to_ndarrays(aggregated_parameters)]
               state_dict = dict(zip(model.state_dict().keys(), aggregated_tensors))
               torch.save(state_dict, f"./weights/round-{server_round}-weights.pt")
           return aggregated_parameters, aggregated_metrics

   def main():
       with open('/etc/ssl/flower/server.crt', 'rb') as f:
           server_cert = f.read()
       with open('/etc/ssl/flower/server.key', 'rb') as f:
           server_key = f.read()
       with open('/etc/ssl/flower/ca.crt', 'rb') as f:
           ca_cert = f.read()

       server_credentials = grpc.ssl_server_credentials(
           [(server_key, server_cert)],
           root_certificates=ca_cert,
           require_client_auth=True
       )

       strategy = SaveModelStrategy(
           fraction_fit=0.3,
           fraction_evaluate=0.2,
           min_fit_clients=3,
           min_evaluate_clients=2,
           min_available_clients=10,
           initial_parameters=ndarrays_to_parameters([np.zeros(1)]),
       )

       fl.server.start_server(
           server_address="0.0.0.0:9092",
           config=fl.server.ServerConfig(num_rounds=3),
           strategy=strategy,
           certificates=(ca_cert, server_cert, server_key),
       )

   if __name__ == "__main__":
       main()
   ```

2. **Run Server Script**

   ```bash
   python3 /path/to/server.py
   ```

### 3. Deploy Client Code on FL Client EC2 Instance

1. **Client Script (`client.py`)**

   ```python
   import grpc
   import logging
   import flwr as fl
   import numpy as np

   # Configure logging
   logging.basicConfig(level=logging.DEBUG)

   class SimpleClient(fl.client.NumPyClient):
       def get_parameters(self):
           return [np.zeros(1)]

       def fit(self, parameters, config):
           new_parameters = np.array([0.2], dtype=np.float32)
           num_examples_train = 1
           metrics = {"accuracy": 0.9}
           return new_parameters, num_examples_train, metrics

       def evaluate(self, parameters, config):
           loss = 0.1
           num_examples_test = 1
           metrics = {"accuracy": 0.95}
           return loss, num_examples_test, metrics

   def main():
       with open('/etc/ssl/flower/ca.crt', 'rb') as f:
           ca_cert = f.read()
       with open('/etc/ssl/flower/client.crt', 'rb') as f:
           client_cert = f.read()
       with open('/etc/ssl/flower/client.key', 'rb') as f:
           client_key = f.read()

       credentials = grpc.ssl_channel_credentials(
           root_certificates=ca_cert,
           private_key=client_key,
           certificate_chain=client_cert
       )

       client = SimpleClient()

       fl.client.start_client(
           server_address="scriptchain.co:9092",
           client=client,
           root_certificates=ca_cert
       )

   if __name__ == "__main__":
       main()
   ```

2. **Run Client Script**

   ```bash
   python3 /path/to/client.py
   ```

## Verification

1. **Verify SSL/TLS Connection**

   ```bash
   openssl s_client -connect scriptchain.co:9092 -CAfile /etc/ssl/flower/ca.crt
   ```

   Ensure the output shows `Verification: OK`.

2. **Check Logs**

   Ensure both server and client logs indicate successful connection and federated learning rounds are executed. Look for `[ROUND 1]` and subsequent round logs.

By following these instructions, you will set up a secure federated learning environment on AWS EC2 instances with TLS encryption.
