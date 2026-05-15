# schema.py defines the structure of one M-Pesa transaction as a Python dataclass.
# Both producer.py and consumer.py import this to stay in sync.

from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class MpesaTransaction:
    transaction_id: str        # Unique ID like "TXN-abc123"
    transaction_type: str      # e.g. "Send Money", "Buy Goods"
    sender_phone: str          # e.g. "0712345678"
    receiver_phone: str
    sender_name: str           # e.g. "Beatrice Njai"
    receiver_name: str         
    amount: float              # amount in KES
    sender_balance_before: float   # Sender's balance before transaction
    sender_balance_after: float    # Sender's balance after transaction
    location: str              # Where the transaction happened e.g. "Nairobi CBD"
    device_fingerprint: str    # Unique device ID which helps detect account takeover.
    timestamp: str             # When the transaction happened (ISO format string)

    def to_dict(self):
        # Converts this dataclass into a plain Python dictionary.
        # Kafka can only send strings/bytes, not Python objects — so we
        # convert to dict first, then JSON-encode it in producer.py.
        return asdict(self)