# consumer.py
# Listens to the Kafka topic and saves every transaction to PostgreSQL.
# Uses confluent-kafka for reliability on Python 3.13.

import json
from datetime import datetime
from confluent_kafka import Consumer, KafkaError
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

load_dotenv()


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def insert_transaction(engine, txn: dict):
    sql = text("""
        INSERT INTO raw_transactions (
            transaction_id, transaction_type, sender_phone, receiver_phone,
            sender_name, receiver_name, amount, sender_balance_before,
            sender_balance_after, location, device_fingerprint, timestamp
        ) VALUES (
            :transaction_id, :transaction_type, :sender_phone, :receiver_phone,
            :sender_name, :receiver_name, :amount, :sender_balance_before,
            :sender_balance_after, :location, :device_fingerprint, :timestamp
        )
        ON CONFLICT (transaction_id) DO NOTHING
    """)
    with engine.connect() as conn:
        conn.execute(sql, txn)
        conn.commit()


def main():
    print("Starting M-Pesa transaction consumer...")
    print("Connecting to PostgreSQL...")

    engine = get_db_engine()
    print("PostgreSQL connected.")

    # confluent-kafka Consumer config dict.auto.offset.reset=earliest: if no prior history, start from the first message ever in the topic.
    # enable.auto.commit=True: automatically tell Kafka which messages are have processed so they don't reprocess them on restart.
    consumer = Consumer({
        'bootstrap.servers': os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        'group.id': 'fraud-detection-consumer',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True
    })

    consumer.subscribe([os.getenv("KAFKA_TOPIC", "mpesa_transactions")])
    print("Kafka connected. Listening for transactions...\n")

    stored = 0
    failed = 0

    try:
        while True:
            # poll(1.0) waits up to 1 second for a new message.
            # Returns None if nothing arrived in that second and just loops again.
            msg = consumer.poll(1.0)

            if msg is None:
                continue  # No message yet, keep waiting

            if msg.error():
                # KafkaError._PARTITION_EOF means we've read all existing messages and are now waiting for new ones.
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"[KAFKA ERROR] {msg.error()}")
                    continue

            # Decode the message bytes back into a Python dict
            txn = json.loads(msg.value().decode("utf-8"))

            try:
                insert_transaction(engine, txn)
                stored += 1
                print(f"[STORED {stored}] {txn['transaction_id']} | "
                      f"{txn['transaction_type']} | KES {float(txn['amount']):,.2f} | "
                      f"{txn['sender_name']} → {txn['receiver_name']}")

            except Exception as e:
                failed += 1
                print(f"[FAILED {failed}] {txn.get('transaction_id', 'unknown')} | Error: {e}")

    except KeyboardInterrupt:
        print(f"\nConsumer stopped.")
        print(f"Total stored: {stored} | Total failed: {failed}")
        consumer.close()


if __name__ == "__main__":
    main()