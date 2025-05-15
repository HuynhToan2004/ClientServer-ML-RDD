from trainer import SparkConfig, Trainer
from models.pac import PACModel
import socket
import json
import numpy as np
import sys



def test(TCP_IP, TCP_PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    print(f"[Receiver] Connected to {TCP_IP}:{TCP_PORT}")

    buffer = ""
    try:
        while True:
            data = s.recv(4096).decode()  # nhận một phần
            if not data:
                break

            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)  # tách dòng JSON
                try:
                    payload = json.loads(line)
                    print(f"[Receiver] Received batch of {len(payload)} samples")

                    labels = [v["label"] for v in payload.values()]
                    print(f"[Receiver] Labels: {labels[:10]}...")

                except json.JSONDecodeError as e:
                    print(f"[Receiver] Failed to decode JSON: {e}")

    except KeyboardInterrupt:
        print("\n[Receiver] Interrupted by user.")
    finally:
        s.close()
        print("[Receiver] Socket closed.")

if __name__ == "__main__":

    spark_config = SparkConfig()
    pacmodel = PACModel()
    trainer = Trainer(pacmodel, "train", spark_config)


    trainer.train()
    # trainer.predict()