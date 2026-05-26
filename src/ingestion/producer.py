# producer.py
# Generates realistic M-Pesa transactions and sends them to Kafka.

import json
import random
import time
import uuid
from datetime import datetime
from confluent_kafka import Producer
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ingestion.schema import MpesaTransaction

load_dotenv()

# Realistic data pools

CUSTOMER_NAMES = [
    "Michael Kinyua", "Amina Wanjiku", "Peter Otieno", "Grace Njeri", "Stacy Wairimu", "Florence Hamisi",
    "Samuel Kamau", "Fatuma Achieng", "David Mwangi", "Joyce Waithera", "Abdulah Hassan",
    "James Kipchoge", "Mary Wambui", "Hassan Abdi", "Lilian Adhiambo", "Timothy Mutiso",
    "Kevin Mutua", "Esther Chebet", "Moses Odhiambo", "Purity Karimi", "Christopher Njoroge",
    "Vincent Omondi", "Beatrice Nkirote", "Emmanuel Waweru", "Agnes Mumbi", "Christine Ndungu",
    "Dennis Kiptoo", "Alice Wanjiru", "Brian Mwangi", "Carolyn Njeri", "Anthony Kimani", "Faith Wambui", "Joseph Mwangi", "Francisca Achieng", "Mark Kiprono", "Ruth Njeri", "Eric Mwangi", "Jane Wairimu", "Simon Karanja", "Nancy Wambui", "Paul Mwangi", "Catherine Njeri", "Andrew Kipchoge", "Rose Wanjiku", "Daniel Mwangi", "Martha Njeri", "Kevin Otieno", "Alice Wairimu", "Michael Mwangi", "Grace Achieng", "Samuel Kamau", "Fatuma Achieng", "David Mwangi", "Joyce Waithera", "Abdul Majid", "Jimmy Kiprop", "Maryanne Mwendwa", "Valentine Wambui", "Lilian Adhiambo", "Timothy Mutiso", "Kevin Mutua"
]

CUSTOMER_NUMBER = [f"07{random.randint(10,99)}{random.randint(100000,999999)}"
                 for _ in range(200)]

LOCATIONS = [
    "Nairobi CBD", "Westlands", "Kibera", "Kasarani", "Embakasi", "Kiambu", "Limuru", "Juja",
    "Thika", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Lamu", "Githunguri", "Narok", "Busia", "Embu",
    "Machakos", "Nyeri", "Meru", "Kakamega", "Garissa", "Homabay", "Kitale", "Kisii", "Vihiga", "Bungoma", "Murang'a", "Kericho", "Kajiado", "Siaya", "Ahero", "Chuka", "Turkana", "Marsabit", "Kapenguria", "Kitui", "Lodwar", "Ruiru", "Runda", "Karen", "Syokimau", "Ngong'", "Dandora", "Mathare", "Githurai", "Kahawa Sukari", "Kitusuru", "Kileleshwa", "Naivasha", "Nanyuki", "Nyahururu",
    "Moyale", "Iten", "Kapsabet", "Bomet", "Maralal"
]

TRANSACTION_TYPES = [
    "Send Money", "Buy Goods", "Pay Bill", "Withdraw",
    "Pochi la Biashara", "Airtime Purchase", "Lipa na Mpesa",
]

TYPE_WEIGHTS = [0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05]

AMOUNT_RANGES = {
    "Send Money":        (50, 70000),
    "Buy Goods":         (20, 15000),
    "Pay Bill":          (100, 50000),
    "Withdraw":          (100, 70000),
    "Pochi la Biashara": (10, 5000),
    "Airtime Purchase":  (5, 1000),
    "Lipa na Mpesa":     (50, 30000),
}


def delivery_report(err, msg):
    """
    Callback function for confluent-kafka calls this after every send attempt.
    If err is None, the message was delivered successfully.
    """
    if err is not None:
        print(f"[DELIVERY FAILED] {err}")


def generate_transaction():
    txn_type = random.choices(TRANSACTION_TYPES, weights=TYPE_WEIGHTS)[0]
    amount = round(random.uniform(*AMOUNT_RANGES[txn_type]), 2)
    balance_before = round(random.uniform(amount, 150000), 2)
    balance_after = round(balance_before - amount, 2)

    sender_name = random.choice(CUSTOMER_NAMES)
    receiver_name = random.choice(CUSTOMER_NAMES)
    while receiver_name == sender_name:
        receiver_name = random.choice(CUSTOMER_NAMES)

    return MpesaTransaction(
        transaction_id=f"TXN-{uuid.uuid4().hex[:12].upper()}",
        transaction_type=txn_type,
        sender_phone=f"07{random.randint(10,99)}{random.randint(100000,999999)}",
        receiver_phone=f"07{random.randint(10,99)}{random.randint(100000,999999)}",
        sender_name=sender_name,
        receiver_name=receiver_name,
        amount=amount,
        sender_balance_before=balance_before,
        sender_balance_after=balance_after,
        location=random.choice(LOCATIONS),
        device_fingerprint=f"DEV-{uuid.uuid4().hex[:12].upper()}",
        timestamp=datetime.now().isoformat()
    )


def get_send_interval():
    hour = datetime.now().hour
    if 0 <= hour < 6:
        return random.uniform(8, 15)
    elif hour in [7, 8, 17, 18]:
        return random.uniform(1, 3)
    else:
        return random.uniform(3, 6)


def main():
    print("Starting M-Pesa transaction producer...")

    # confluent-kafka uses a config dict instead of keyword arguments
    producer = Producer({
        'bootstrap.servers': os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    })

    topic = os.getenv("KAFKA_TOPIC", "mpesa_transactions")
    count = 0

    try:
        while True:
            txn = generate_transaction()

            # produce() sends the message. We JSON-encode it to a string, then encode to bytes coz Kafka only transmits bytes.
            # on_delivery=delivery_report wires up our callback above.
            producer.produce(
                topic,
                value=json.dumps(txn.to_dict()).encode("utf-8"),
                on_delivery=delivery_report
            )

            # poll(0) checks for delivery callbacks without blocking. Without this, delivery_report would never be called.
            producer.poll(0)

            count += 1
            print(f"[{count}] Sent: {txn.transaction_id} | "
                  f"{txn.transaction_type} | KES {txn.amount:,.2f} | "
                  f"{txn.sender_name} → {txn.receiver_name} | "
                  f"{txn.location}")

            time.sleep(get_send_interval())

    except KeyboardInterrupt:
        print(f"\nFlushing remaining messages...")
        # flush() waits until all sent messages are confirmed delivered
        producer.flush()
        print(f"Producer stopped. Total transactions sent: {count}")


if __name__ == "__main__":
    main()