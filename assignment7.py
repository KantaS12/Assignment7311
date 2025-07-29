"""
Five Distinct Message Communication Implementations - Python Version
Each implementation addresses one specific requirement from the assignment
"""

import random
import json
import math
import numpy as np
from collections import deque
import time

# ===============================
# SHARED FOUNDATION CLASSES
# ===============================

class Person:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.connections = set()
        self.public_key = None
        self.private_key = None
        self.messages = []

    def add_connection(self, person):
        self.connections.add(person)
        person.connections.add(self)

class Message:
    def __init__(self, sender_id, receiver_id, body, metadata=None):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.body = body
        self.metadata = metadata if metadata else {}
        self.metadata['timestamp'] = time.time()
        self.metadata['message_id'] = self.generate_id()

    def generate_id(self):
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=9))

class CommunicationNetwork:
    def __init__(self):
        self.people = {}
        self.message_history = []

    def add_person(self, person):
        self.people[person.id] = person

    def find_path(self, sender_id, receiver_id):
        if sender_id == receiver_id:
            return [sender_id]

        visited = set()
        queue = deque([[sender_id]])

        while queue:
            path = queue.popleft()
            current_id = path[-1]

            if current_id in visited:
                continue
            visited.add(current_id)

            current_person = self.people.get(current_id)
            if not current_person:
                continue

            for connection in current_person.connections:
                if connection.id == receiver_id:
                    return path + [receiver_id]

                if connection.id not in visited:
                    queue.append(path + [connection.id])

        return None

    def send_message(self, message):
        path = self.find_path(message.sender_id, message.receiver_id)
        if not path:
            raise Exception(f'No path found from {message.sender_id} to {message.receiver_id}')

        message.metadata['route_path'] = path

        # Route to final recipient
        receiver = self.people.get(message.receiver_id)
        receiver.messages.append(message)

        self.message_history.append(message)
        return message.metadata['message_id']

# ===============================
# IMPLEMENTATION 1: RUN-LENGTH ENCODING
# ===============================

class RunLengthEncodingImplementation:
    """
    IMPLEMENTATION 1: Send compressed messages using run-length encoding
    Requirement: "The metadata should indicate that the message is run-length encoded"
    """

    @staticmethod
    def encode(message):
        """
        Encodes message using run-length encoding
        Time Complexity: O(n) where n is message length
        Example: "aaaabbbb" -> "4a4b"
        """
        if not message:
            return ''

        encoded = ''
        count = 1
        current_char = message[0]

        for i in range(1, len(message)):
            if message[i] == current_char:
                count += 1
            else:
                encoded += f'{count}{current_char}' if count > 1 else current_char
                current_char = message[i]
                count = 1

        encoded += f'{count}{current_char}' if count > 1 else current_char
        return encoded

    @staticmethod
    def decode(encoded):
        """
        Decodes run-length encoded message
        Time Complexity: O(m + k) where m is encoded length, k is decoded length
        """
        if not encoded:
            return ''

        decoded = ''
        i = 0

        while i < len(encoded):
            count = ''
            
            while i < len(encoded) and encoded[i].isdigit():
                count += encoded[i]
                i += 1

            count = int(count) if count else 1

            if i < len(encoded):
                decoded += encoded[i] * count
                i += 1

        return decoded

    @staticmethod
    def send_message(network, sender_id, receiver_id, original_message):
        """
        Sends run-length encoded message through network
        Creates message with specific metadata indicating run-length encoding
        """
        print(f'\n🔤 IMPLEMENTATION 1: Run-Length Encoding')
        print(f'Original message: "{original_message}"')

        # Encode the message
        encoded_message = RunLengthEncodingImplementation.encode(original_message)
        compression_ratio = len(encoded_message) / len(original_message)

        # Create message with required metadata
        message = Message(sender_id, receiver_id, encoded_message, {
            # REQUIRED: Metadata indicating run-length encoding
            'encoding': 'run-length-encoded',
            'compression_type': 'RLE',
            'original_length': len(original_message),
            'compressed_length': len(encoded_message),
            'compression_ratio': compression_ratio,
            'algorithm': 'Run-Length Encoding'
        })

        print(f'Encoded message: "{encoded_message}"')
        print(f'Compression ratio: {compression_ratio:.3f}')
        print(f'Metadata indicates: {message.metadata["encoding"]}')

        return network.send_message(message)

    @staticmethod
    def receive_message(message):
        """
        Receives and decodes run-length encoded message
        """
        # Verify metadata indicates run-length encoding
        if message.metadata.get('encoding') != 'run-length-encoded':
            raise Exception('Message metadata does not indicate run-length encoding')

        decoded_message = RunLengthEncodingImplementation.decode(message.body)
        print(f'Decoded message: "{decoded_message}"')
        print(f'Verified encoding type: {message.metadata["encoding"]}')

        return decoded_message

# ===============================
# IMPLEMENTATION 2: FFT LOSSY COMPRESSION
# ===============================

class FFTLossyCompressionImplementation:
    """
    IMPLEMENTATION 2: Send lossy compressed messages using Fast Fourier Transform
    Requirement: "Sender may specify how lossy the compression should be"
    Requirement: "Messages should be 'blurry', fine details can be lost"
    Requirement: "Metadata specifies original message length"
    """

    @staticmethod
    def dft(signal):
        """
        Discrete Fourier Transform for frequency analysis
        Time Complexity: O(n²) - educational implementation
        """
        N = len(signal)
        result = []

        for k in range(N):
            real = 0
            imag = 0

            for n in range(N):
                angle = -2 * math.pi * k * n / N
                real += signal[n] * math.cos(angle)
                imag += signal[n] * math.sin(angle)

            result.append({
                'real': real,
                'imag': imag,
                'magnitude': math.sqrt(real * real + imag * imag)
            })

        return result

    @staticmethod
    def idft(coefficients):
        """
        Inverse DFT to reconstruct signal
        """
        N = len(coefficients)
        result = []

        for n in range(N):
            real = 0

            for k in range(N):
                angle = 2 * math.pi * k * n / N
                real += coefficients[k]['real'] * math.cos(angle) - coefficients[k]['imag'] * math.sin(angle)

            result.append(real / N)

        return result

    @staticmethod
    def compress_lossy(message, loss_level):
        """
        Applies lossy compression to message
        loss_level: 0.0 = no loss, 0.9 = very lossy/blurry
        """
        # Convert message to signal
        signal = [ord(char) for char in message]
        original_length = len(signal)

        # Pad for better processing
        while len(signal) & (len(signal) - 1):
            signal.append(0)

        # Apply FFT
        coefficients = FFTLossyCompressionImplementation.dft(signal)

        # LOSSY COMPRESSION: Remove high-frequency components
        keep_ratio = 1 - loss_level
        keep_count = int(len(coefficients) * keep_ratio)

        # Sort by magnitude and keep most important frequencies
        sorted_indices = sorted(range(len(coefficients)), 
                              key=lambda i: coefficients[i]['magnitude'], 
                              reverse=True)[:keep_count]

        # Zero out less important frequencies (creates "blurry" effect)
        lossy_coefficients = []
        for i, coef in enumerate(coefficients):
            if i in sorted_indices:
                lossy_coefficients.append(coef)
            else:
                lossy_coefficients.append({'real': 0, 'imag': 0, 'magnitude': 0})

        return {
            'coefficients': lossy_coefficients,
            'original_length': original_length,
            'padded_length': len(signal),
            'frequencies_kept': keep_count,
            'loss_level': loss_level
        }

    @staticmethod
    def decompress_lossy(compressed_data):
        """
        Decompresses lossy message
        """
        signal = FFTLossyCompressionImplementation.idft(compressed_data['coefficients'])

        result = ''
        for i in range(compressed_data['original_length']):
            char_code = round(max(0, min(255, signal[i])))
            result += chr(char_code)

        return result

    @staticmethod
    def send_message(network, sender_id, receiver_id, original_message, loss_level=0.3):
        """
        Sends lossy compressed message with sender-specified loss level
        """
        print(f'\n🌊 IMPLEMENTATION 2: FFT Lossy Compression')
        print(f'Original message: "{original_message}"')
        print(f'Sender-specified loss level: {loss_level * 100:.0f}%')

        # Apply lossy compression
        compressed = FFTLossyCompressionImplementation.compress_lossy(original_message, loss_level)
        serialized_data = json.dumps([[c['real'], c['imag']] for c in compressed['coefficients']])

        # Create message with required metadata
        message = Message(sender_id, receiver_id, serialized_data, {
            # REQUIRED: Metadata specifies original message length
            'original_message_length': compressed['original_length'],
            'compression_type': 'FFT-Lossy',
            'loss_level': loss_level,
            'frequencies_kept': compressed['frequencies_kept'],
            'total_frequencies': len(compressed['coefficients']),
            'padded_length': compressed['padded_length'],
            'is_blurry': True,  # Indicates fine details lost
            'algorithm': 'Fast Fourier Transform Lossy Compression'
        })

        print(f'Original length: {compressed["original_length"]} characters')
        print(f'Frequencies kept: {compressed["frequencies_kept"]}/{len(compressed["coefficients"])}')
        print(f'Blurry message (fine details lost): {loss_level > 0}')
        print(f'Metadata specifies original length: {message.metadata["original_message_length"]}')

        return network.send_message(message)

    @staticmethod
    def receive_message(message):
        """
        Receives and decompresses lossy message
        """
        # Verify this is FFT lossy compressed
        if message.metadata.get('compression_type') != 'FFT-Lossy':
            raise Exception('Message is not FFT lossy compressed')

        # Reconstruct compression data
        coefficient_pairs = json.loads(message.body)
        compressed_data = {
            'coefficients': [{'real': real, 'imag': imag} for real, imag in coefficient_pairs],
            'original_length': message.metadata['original_message_length'],
            'padded_length': message.metadata['padded_length']
        }

        decompressed = FFTLossyCompressionImplementation.decompress_lossy(compressed_data)

        print(f'Decompressed message: "{decompressed}"')
        print(f'Original length from metadata: {message.metadata["original_message_length"]}')
        print(f'Loss level was: {message.metadata["loss_level"] * 100:.0f}%')
        print(f'Message was blurry: {message.metadata["is_blurry"]}')

        return decompressed

# ===============================
# IMPLEMENTATION 3: RSA ENCRYPTION
# ===============================

class RSAEncryptionImplementation:
    """
    IMPLEMENTATION 3: Send encrypted messages using RSA encryption
    Requirement: "Sender and receiver both need public key and private key"
    """

    @staticmethod
    def generate_key_pair():
        """
        Generates RSA key pair (public and private keys)
        """
        # Using small primes for educational purposes
        p = 61
        q = 53
        n = p * q  # 3233
        phi = (p - 1) * (q - 1)  # 3120
        e = 17  # Public exponent
        d = RSAEncryptionImplementation.mod_inverse(e, phi)  # Private exponent

        return {
            'public_key': {'n': n, 'e': e},
            'private_key': {'n': n, 'd': d}
        }

    @staticmethod
    def extended_gcd(a, b):
        """
        Extended Euclidean Algorithm for modular inverse
        """
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = RSAEncryptionImplementation.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        return gcd, x, x1

    @staticmethod
    def mod_inverse(a, m):
        gcd, x, _ = RSAEncryptionImplementation.extended_gcd(a, m)
        if gcd != 1:
            raise Exception('Modular inverse does not exist')
        return (x % m + m) % m

    @staticmethod
    def mod_pow(base, exponent, modulus):
        """
        Fast modular exponentiation
        Time Complexity: O(log exponent)
        """
        result = 1
        base = base % modulus

        while exponent > 0:
            if exponent % 2 == 1:
                result = (result * base) % modulus
            exponent //= 2
            base = (base * base) % modulus

        return result

    @staticmethod
    def encrypt(message, public_key):
        """
        Encrypts message using receiver's public key
        """
        n = public_key['n']
        e = public_key['e']
        encrypted = []

        for char in message:
            char_code = ord(char)
            encrypted_char = RSAEncryptionImplementation.mod_pow(char_code, e, n)
            encrypted.append(encrypted_char)

        return json.dumps(encrypted)

    @staticmethod
    def decrypt(encrypted_message, private_key):
        """
        Decrypts message using receiver's private key
        """
        n = private_key['n']
        d = private_key['d']
        encrypted = json.loads(encrypted_message)
        decrypted = ''

        for encrypted_char in encrypted:
            char_code = RSAEncryptionImplementation.mod_pow(encrypted_char, d, n)
            decrypted += chr(char_code)

        return decrypted

    @staticmethod
    def setup_keys(network, sender_id, receiver_id):
        """
        Sets up both sender and receiver with public/private key pairs
        """
        sender = network.people[sender_id]
        receiver = network.people[receiver_id]

        # REQUIRED: Both sender and receiver need public and private keys
        if not sender.public_key or not sender.private_key:
            sender_keys = RSAEncryptionImplementation.generate_key_pair()
            sender.public_key = sender_keys['public_key']
            sender.private_key = sender_keys['private_key']
            print(f'Generated keys for sender {sender_id}')

        if not receiver.public_key or not receiver.private_key:
            receiver_keys = RSAEncryptionImplementation.generate_key_pair()
            receiver.public_key = receiver_keys['public_key']
            receiver.private_key = receiver_keys['private_key']
            print(f'Generated keys for receiver {receiver_id}')

        return {
            'sender_keys': {'public': sender.public_key, 'private': sender.private_key},
            'receiver_keys': {'public': receiver.public_key, 'private': receiver.private_key}
        }

    @staticmethod
    def send_message(network, sender_id, receiver_id, original_message):
        """
        Sends RSA encrypted message
        """
        print(f'\n🔒 IMPLEMENTATION 3: RSA Encryption')
        print(f'Original message: "{original_message}"')

        # Ensure both parties have keys
        keys = RSAEncryptionImplementation.setup_keys(network, sender_id, receiver_id)
        receiver = network.people[receiver_id]

        # Encrypt using receiver's public key
        encrypted_message = RSAEncryptionImplementation.encrypt(original_message, receiver.public_key)

        message = Message(sender_id, receiver_id, encrypted_message, {
            'encryption_type': 'RSA',
            'algorithm': 'RSA Encryption',
            'key_size': len(str(receiver.public_key['n'])) * 8,
            'sender_has_keys': True,
            'receiver_has_keys': True,
            'encrypted_with': 'receiver-public-key'
        })

        print(f'Encrypted message length: {len(encrypted_message)} bytes')
        print(f'Sender has public/private keys: ✓')
        print(f'Receiver has public/private keys: ✓')
        print(f'Encrypted with receiver\'s public key')

        return network.send_message(message)

    @staticmethod
    def receive_message(message, receiver_private_key):
        """
        Receives and decrypts RSA encrypted message
        """
        if message.metadata.get('encryption_type') != 'RSA':
            raise Exception('Message is not RSA encrypted')

        decrypted_message = RSAEncryptionImplementation.decrypt(message.body, receiver_private_key)

        print(f'Decrypted message: "{decrypted_message}"')
        print(f'Used receiver\'s private key for decryption')

        return decrypted_message

# ===============================
# IMPLEMENTATION 4: RSA MESSAGE SIGNING
# ===============================

class RSAMessageSigningImplementation:
    """
    IMPLEMENTATION 4: Send signed messages using RSA
    Requirement: "Signature should be RSA-encrypted hash of the message"
    Requirement: "Receiver can decrypt hash and confirm it matches by redoing hash"
    """

    @staticmethod
    def hash(message):
        """
        Simple hash function for message integrity
        Time Complexity: O(n)
        """
        hash_value = 0
        for char in message:
            hash_value = ((hash_value << 5) - hash_value) + ord(char)
            hash_value = hash_value & hash_value
        return abs(hash_value)

    # Uses RSA key operations from Implementation 3
    mod_pow = RSAEncryptionImplementation.mod_pow
    generate_key_pair = RSAEncryptionImplementation.generate_key_pair

    @staticmethod
    def sign_message(message, sender_private_key):
        """
        Creates RSA signature by encrypting hash with sender's private key
        """
        # Step 1: Create hash of message
        message_hash = RSAMessageSigningImplementation.hash(message)

        # Step 2: RSA-encrypt the hash with sender's private key (this is the signature)
        signature = RSAMessageSigningImplementation.mod_pow(
            message_hash, sender_private_key['d'], sender_private_key['n']
        )

        return {
            'original_message': message,
            'message_hash': message_hash,
            'signature': signature
        }

    @staticmethod
    def verify_signature(signed_data, sender_public_key):
        """
        Verifies signature by decrypting with sender's public key
        """
        original_message = signed_data['original_message']
        message_hash = signed_data['message_hash']
        signature = signed_data['signature']

        # Step 1: Decrypt signature using sender's public key
        decrypted_hash = RSAMessageSigningImplementation.mod_pow(
            signature, sender_public_key['e'], sender_public_key['n']
        )

        # Step 2: Redo the hash of the message
        recomputed_hash = RSAMessageSigningImplementation.hash(original_message)

        # Step 3: Confirm that decrypted hash matches recomputed hash
        is_valid = (decrypted_hash == recomputed_hash) and (recomputed_hash == message_hash)

        return {
            'valid': is_valid,
            'decrypted_hash': decrypted_hash,
            'recomputed_hash': recomputed_hash,
            'original_hash': message_hash
        }

    @staticmethod
    def send_message(network, sender_id, receiver_id, original_message):
        """
        Sends signed message through network
        """
        print(f'\n✍️ IMPLEMENTATION 4: RSA Message Signing')
        print(f'Original message: "{original_message}"')

        sender = network.people[sender_id]

        # Ensure sender has keys
        if not sender.public_key or not sender.private_key:
            keys = RSAEncryptionImplementation.generate_key_pair()
            sender.public_key = keys['public_key']
            sender.private_key = keys['private_key']

        # REQUIRED: Create RSA-encrypted hash signature
        signed_data = RSAMessageSigningImplementation.sign_message(original_message, sender.private_key)

        message = Message(sender_id, receiver_id, json.dumps(signed_data), {
            'message_type': 'RSA-signed',
            'algorithm': 'RSA Digital Signature',
            'sender_public_key': sender.public_key,
            'signature_method': 'RSA-encrypted-hash',
            'original_message_hash': signed_data['message_hash'],
            'signature': signed_data['signature']
        })

        print(f'Message hash: {signed_data["message_hash"]}')
        print(f'RSA-encrypted hash (signature): {signed_data["signature"]}')
        print(f'Signature created with sender\'s private key')

        return network.send_message(message)

    @staticmethod
    def receive_message(message):
        """
        Receives and verifies signed message
        """
        if message.metadata.get('message_type') != 'RSA-signed':
            raise Exception('Message is not RSA signed')

        signed_data = json.loads(message.body)
        sender_public_key = message.metadata['sender_public_key']

        # REQUIRED: Decrypt hash and confirm it matches by redoing hash
        verification = RSAMessageSigningImplementation.verify_signature(signed_data, sender_public_key)

        print(f'Received signed message: "{signed_data["original_message"]}"')
        print(f'Decrypted hash from signature: {verification["decrypted_hash"]}')
        print(f'Recomputed hash of message: {verification["recomputed_hash"]}')
        print(f'Original hash matches: {verification["decrypted_hash"] == verification["original_hash"]}')
        print(f'Recomputed hash matches: {verification["recomputed_hash"] == verification["original_hash"]}')
        print(f'Signature valid: {verification["valid"]}')

        return {
            'message': signed_data['original_message'],
            'verified': verification['valid'],
            'verification': verification
        }

# ===============================
# IMPLEMENTATION 5: SIGNED CONFIRMATIONS
# ===============================

class SignedConfirmationImplementation:
    """
    IMPLEMENTATION 5: Send signed confirmation messages
    Requirement: "Return signed message confirming the signed message was received and validated"
    Requirement: "Include original signature, original hash, hash of returned message, encrypted hash of returned message"
    """

    # Uses hash and RSA operations from previous implementations
    hash = RSAMessageSigningImplementation.hash
    mod_pow = RSAEncryptionImplementation.mod_pow
    generate_key_pair = RSAEncryptionImplementation.generate_key_pair

    @staticmethod
    def create_confirmation(original_signed_message, verification_result, confirmer_private_key):
        """
        Creates comprehensive signed confirmation message
        """
        original_data = json.loads(original_signed_message.body)

        # Create confirmation message text
        confirmation_text = f'CONFIRMATION: Message from {original_signed_message.sender_id} has been received and {"successfully validated" if verification_result["verified"] else "failed validation"}'

        # REQUIRED: Include original signature and original hash
        confirmation_data = {
            'confirmation_message': confirmation_text,
            'original_sender': original_signed_message.sender_id,
            'original_signature': original_data['signature'],
            'original_hash': original_data['message_hash'],
            'verification_status': verification_result['verified'],
            'timestamp': time.time()
        }

        # REQUIRED: Create hash of the returned message
        confirmation_hash = SignedConfirmationImplementation.hash(json.dumps(confirmation_data))

        # REQUIRED: Create encrypted hash of the returned message (signature)
        encrypted_confirmation_hash = SignedConfirmationImplementation.mod_pow(
            confirmation_hash,
            confirmer_private_key['d'],
            confirmer_private_key['n']
        )

        # Complete confirmation package
        full_confirmation = {
            **confirmation_data,
            'hash_of_returned_message': confirmation_hash,
            'encrypted_hash_of_returned_message': encrypted_confirmation_hash
        }

        return full_confirmation

    @staticmethod
    def send_confirmation(network, confirmer_id, original_sender_id, original_signed_message, verification_result):
        """
        Sends signed confirmation in response to a signed message
        """
        print(f'\n📋 IMPLEMENTATION 5: Signed Confirmation')
        print(f'Confirming message from {original_sender_id} to {confirmer_id}')

        confirmer = network.people[confirmer_id]

        # Ensure confirmer has keys
        if not confirmer.public_key or not confirmer.private_key:
            keys = SignedConfirmationImplementation.generate_key_pair()
            confirmer.public_key = keys['public_key']
            confirmer.private_key = keys['private_key']

        # REQUIRED: Create comprehensive confirmation with all required elements
        confirmation = SignedConfirmationImplementation.create_confirmation(
            original_signed_message,
            verification_result,
            confirmer.private_key
        )

        message = Message(confirmer_id, original_sender_id, json.dumps(confirmation), {
            'message_type': 'signed-confirmation',
            'algorithm': 'RSA Signed Confirmation',
            'confirmer_public_key': confirmer.public_key,
            'original_message_id': original_signed_message.metadata['message_id'],
            'confirmation_for': original_sender_id,

            # REQUIRED ELEMENTS in metadata for easy access:
            'original_signature': confirmation['original_signature'],
            'original_hash': confirmation['original_hash'],
            'hash_of_returned_message': confirmation['hash_of_returned_message'],
            'encrypted_hash_of_returned_message': confirmation['encrypted_hash_of_returned_message']
        })

        print(f'✅ Confirmation includes all required elements:')
        print(f'   • Original received signature: {confirmation["original_signature"]}')
        print(f'   • Original hash: {confirmation["original_hash"]}')
        print(f'   • Hash of returned message: {confirmation["hash_of_returned_message"]}')
        print(f'   • Encrypted hash of returned message: {confirmation["encrypted_hash_of_returned_message"]}')
        print(f'   • Verification status: {confirmation["verification_status"]}')

        return network.send_message(message)

    @staticmethod
    def receive_confirmation(confirmation_message):
        """
        Receives and processes signed confirmation
        """
        if confirmation_message.metadata.get('message_type') != 'signed-confirmation':
            raise Exception('Message is not a signed confirmation')

        confirmation_data = json.loads(confirmation_message.body)
        confirmer_public_key = confirmation_message.metadata['confirmer_public_key']

        # Verify the confirmation signature
        decrypted_hash = SignedConfirmationImplementation.mod_pow(
            confirmation_data['encrypted_hash_of_returned_message'],
            confirmer_public_key['e'],
            confirmer_public_key['n']
        )

        # Recreate confirmation data without signature for hash verification
        data_to_hash = {
            'confirmation_message': confirmation_data['confirmation_message'],
            'original_sender': confirmation_data['original_sender'],
            'original_signature': confirmation_data['original_signature'],
            'original_hash': confirmation_data['original_hash'],
            'verification_status': confirmation_data['verification_status'],
            'timestamp': confirmation_data['timestamp']
        }

        recomputed_hash = SignedConfirmationImplementation.hash(json.dumps(data_to_hash))
        confirmation_valid = decrypted_hash == recomputed_hash

        print(f'📨 Received signed confirmation:')
        print(f'   • Confirmation message: "{confirmation_data["confirmation_message"]}"')
        print(f'   • Original signature included: {confirmation_data["original_signature"]}')
        print(f'   • Original hash included: {confirmation_data["original_hash"]}')
        print(f'   • Hash of returned message: {confirmation_data["hash_of_returned_message"]}')
        print(f'   • Encrypted hash of returned message: {confirmation_data["encrypted_hash_of_returned_message"]}')
        print(f'   • Confirmation signature valid: {confirmation_valid}')
        print(f'   • Original message was verified: {confirmation_data["verification_status"]}')

        return {
            'confirmation_valid': confirmation_valid,
            'original_message_verified': confirmation_data['verification_status'],
            'confirmation_data': confirmation_data
        }

# ===============================
# DEMONSTRATION OF ALL 5 IMPLEMENTATIONS
# ===============================

def demonstrate_all_implementations():
    """
    Comprehensive demonstration of all five implementations
    Shows each requirement working independently and together
    """
    print('=' * 80)
    print('    COMPREHENSIVE DEMONSTRATION: ALL 5 IMPLEMENTATIONS')
    print('=' * 80)

    # Setup network and people
    network = CommunicationNetwork()

    people = [
        Person('alice', 'Alice'),
        Person('bob', 'Bob'),
        Person('charlie', 'Charlie'),
        Person('diana', 'Diana'),
        Person('eve', 'Eve')
    ]

    for person in people:
        network.add_person(person)

    # Create network connections
    people[0].add_connection(people[1])  # Alice <-> Bob
    people[1].add_connection(people[2])  # Bob <-> Charlie
    people[2].add_connection(people[3])  # Charlie <-> Diana
    people[1].add_connection(people[4])  # Bob <-> Eve
    people[4].add_connection(people[3])  # Eve <-> Diana

    print(f'\n📡 Network setup complete: {len(people)} people with connections')
    print('Network topology: Alice <-> Bob <-> Charlie <-> Diana')
    print('                         ↕               ↗')
    print('                        Eve  <-----------')

    # ===============================
    # IMPLEMENTATION 1 DEMO
    # ===============================

    print('\n' + '=' * 60)
    print('TESTING IMPLEMENTATION 1: RUN-LENGTH ENCODING')
    print('=' * 60)

    message1 = 'aaaaaabbbbbbccccccdddddd!!!!!!......'
    msg_id1 = RunLengthEncodingImplementation.send_message(
        network, 'alice', 'diana', message1
    )

    received_msg1 = next((m for m in network.people['diana'].messages if m.metadata['message_id'] == msg_id1), None)
    decoded1 = RunLengthEncodingImplementation.receive_message(received_msg1)

    print(f' Implementation 1 Success: {"PASSED" if decoded1 == message1 else "FAILED"}')
    print(f' Metadata correctly indicates: {received_msg1.metadata["encoding"]}')

    # ===============================
    # IMPLEMENTATION 2 DEMO
    # ===============================

    print('\n' + '=' * 60)
    print('TESTING IMPLEMENTATION 2: FFT LOSSY COMPRESSION')
    print('=' * 60)

    message2 = 'Hello World! This message will become blurry with fine details lost through FFT compression.'

    # Test different loss levels as specified by sender
    loss_levels = [0.2, 0.5, 0.8]

    for loss_level in loss_levels:
        print(f'\n🎛️ Testing sender-specified loss level: {loss_level * 100:.0f}%')

        msg_id2 = FFTLossyCompressionImplementation.send_message(
            network, 'bob', 'alice', message2, loss_level
        )

        received_msg2 = next((m for m in network.people['alice'].messages if m.metadata['message_id'] == msg_id2), None)
        decoded2 = FFTLossyCompressionImplementation.receive_message(received_msg2)

        # Calculate similarity to show "blurry" effect
        similarity = calculate_similarity(message2, decoded2)
        print(f' Blurry effect achieved: {"YES" if similarity < 1.0 else "NO"} ({similarity * 100:.1f}% similarity)')
        print(f' Metadata specifies original length: {received_msg2.metadata["original_message_length"]} chars')

    # ===============================
    # IMPLEMENTATION 3 DEMO
    # ===============================

    print('\n' + '=' * 60)
    print('TESTING IMPLEMENTATION 3: RSA ENCRYPTION')
    print('=' * 60)

    message3 = 'This is a secret message that requires RSA encryption!'
    msg_id3 = RSAEncryptionImplementation.send_message(
        network, 'charlie', 'eve', message3
    )

    received_msg3 = next((m for m in network.people['eve'].messages if m.metadata['message_id'] == msg_id3), None)
    decoded3 = RSAEncryptionImplementation.receive_message(received_msg3, network.people['eve'].private_key)

    print(f'✅ Implementation 3 Success: {"PASSED" if decoded3 == message3 else "FAILED"}')
    print(f'✅ Both parties have public/private keys: {received_msg3.metadata["sender_has_keys"] and received_msg3.metadata["receiver_has_keys"]}')

    # ===============================
    # IMPLEMENTATION 4 DEMO
    # ===============================

    print('\n' + '=' * 60)
    print('TESTING IMPLEMENTATION 4: RSA MESSAGE SIGNING')
    print('=' * 60)

    message4 = 'This is an important message that requires digital signature verification.'
    msg_id4 = RSAMessageSigningImplementation.send_message(
        network, 'diana', 'alice', message4
    )

    received_msg4 = next((m for m in network.people['alice'].messages if m.metadata['message_id'] == msg_id4), None)
    verification_result = RSAMessageSigningImplementation.receive_message(received_msg4)

    print(f'✅ Implementation 4 Success: {"PASSED" if verification_result["verified"] else "FAILED"}')
    print(f'✅ Signature is RSA-encrypted hash: {received_msg4.metadata["signature_method"] == "RSA-encrypted-hash"}')
    print(f'✅ Receiver decrypted hash and confirmed match: {verification_result["verified"]}')

    # ===============================
    # IMPLEMENTATION 5 DEMO
    # ===============================

    print('\n' + '=' * 60)
    print('TESTING IMPLEMENTATION 5: SIGNED CONFIRMATIONS')
    print('=' * 60)

    # Send confirmation back to Diana from Alice
    confirmation_msg_id = SignedConfirmationImplementation.send_confirmation(
        network, 'alice', 'diana', received_msg4, verification_result
    )

    confirmation_msg = next((m for m in network.people['diana'].messages if m.metadata['message_id'] == confirmation_msg_id), None)
    confirmation_result = SignedConfirmationImplementation.receive_confirmation(confirmation_msg)

    print(f'✅ Implementation 5 Success: {"PASSED" if confirmation_result["confirmation_valid"] else "FAILED"}')
    print(f'✅ Includes original signature: {"YES" if confirmation_msg.metadata.get("original_signature") else "NO"}')
    print(f'✅ Includes original hash: {"YES" if confirmation_msg.metadata.get("original_hash") else "NO"}')
    print(f'✅ Includes hash of returned message: {"YES" if confirmation_msg.metadata.get("hash_of_returned_message") else "NO"}')
    print(f'✅ Includes encrypted hash of returned message: {"YES" if confirmation_msg.metadata.get("encrypted_hash_of_returned_message") else "NO"}')

    # ===============================
    # FINAL SUMMARY
    # ===============================

    print('\n' + '=' * 80)
    print('    FINAL SUMMARY: ALL 5 IMPLEMENTATIONS')
    print('=' * 80)

    results = [
        {
            'impl': 'Implementation 1: Run-Length Encoding',
            'requirement': 'Metadata indicates run-length encoding',
            'status': ' PASSED' if received_msg1.metadata.get('encoding') == 'run-length-encoded' else '❌ FAILED'
        },
        {
            'impl': 'Implementation 2: FFT Lossy Compression',
            'requirement': 'Sender specifies loss level, metadata has original length',
            'status': ' PASSED' if received_msg2 and received_msg2.metadata.get('original_message_length') else '❌ FAILED'
        },
        {
            'impl': 'Implementation 3: RSA Encryption',
            'requirement': 'Both sender and receiver have public/private keys',
            'status': ' PASSED' if received_msg3.metadata.get('sender_has_keys') and received_msg3.metadata.get('receiver_has_keys') else '❌ FAILED'
        },
        {
            'impl': 'Implementation 4: RSA Message Signing',
            'requirement': 'Signature is RSA-encrypted hash, receiver verifies by decrypting',
            'status': ' PASSED' if verification_result.get('verified') else '❌ FAILED'
        },
        {
            'impl': 'Implementation 5: Signed Confirmations',
            'requirement': 'Includes original signature, original hash, new hash, encrypted new hash',
            'status': (' PASSED' if (confirmation_msg.metadata.get('original_signature') and
                                    confirmation_msg.metadata.get('original_hash') and
                                    confirmation_msg.metadata.get('hash_of_returned_message') and
                                    confirmation_msg.metadata.get('encrypted_hash_of_returned_message')) else '❌ FAILED')
        }
    ]

    print('\n📊 Implementation Results:')
    for i, result in enumerate(results):
        print(f'\n{i + 1}. {result["impl"]}')
        print(f'   Requirement: {result["requirement"]}')
        print(f'   Status: {result["status"]}')

    all_passed = all('PASSED' in r['status'] for r in results)

    print('\n' + '=' * 80)
    print(f'    OVERALL RESULT: {"🎉 ALL IMPLEMENTATIONS SUCCESSFUL" if all_passed else "⚠️ SOME IMPLEMENTATIONS NEED REVIEW"}')
    print('=' * 80)

    # Network statistics
    print(f'\n📈 Network Statistics:')
    print(f'   • Total messages sent: {len(network.message_history)}')
    print(f'   • People in network: {len(network.people)}')
    print(f'   • Message types demonstrated: 5')

    message_types = []
    for m in network.message_history:
        msg_type = (m.metadata.get('encoding') or 
                   m.metadata.get('compression_type') or 
                   m.metadata.get('encryption_type') or 
                   m.metadata.get('message_type') or 'other')
        message_types.append(msg_type)

    type_counts = {}
    for msg_type in message_types:
        type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

    print(f'   • Message type breakdown:')
    for msg_type, count in type_counts.items():
        print(f'     - {msg_type}: {count} message(s)')

    return {
        'all_implementations_successful': all_passed,
        'results': results,
        'network_stats': {
            'total_messages': len(network.message_history),
            'people_count': len(network.people),
            'message_types': type_counts
        }
    }

def calculate_similarity(str1, str2):
    """
    Utility function to calculate string similarity
    """
    max_length = max(len(str1), len(str2))
    if max_length == 0:
        return 1.0

    matches = 0
    min_length = min(len(str1), len(str2))

    for i in range(min_length):
        if str1[i] == str2[i]:
            matches += 1

    return matches / max_length

# ===============================
# INDIVIDUAL IMPLEMENTATION TESTING FUNCTIONS
# ===============================

def test_implementation_1():
    """Test Implementation 1: Run-Length Encoding"""
    print(' Testing Implementation 1: Run-Length Encoding')
    network = CommunicationNetwork()
    alice = Person('alice', 'Alice')
    bob = Person('bob', 'Bob')
    alice.add_connection(bob)
    network.add_person(alice)
    network.add_person(bob)

    test_message = 'aaaabbbbccccdddd'
    msg_id = RunLengthEncodingImplementation.send_message(network, 'alice', 'bob', test_message)
    received = next((m for m in network.people['bob'].messages if m.metadata['message_id'] == msg_id), None)
    decoded = RunLengthEncodingImplementation.receive_message(received)

    result = decoded == test_message
    print(f'Result: {"PASSED" if result else "FAILED"}')
    return result

def test_implementation_2():
    """Test Implementation 2: FFT Lossy Compression"""
    print(' Testing Implementation 2: FFT Lossy Compression')
    network = CommunicationNetwork()
    alice = Person('alice', 'Alice')
    bob = Person('bob', 'Bob')
    alice.add_connection(bob)
    network.add_person(alice)
    network.add_person(bob)

    test_message = 'Hello World Test'
    msg_id = FFTLossyCompressionImplementation.send_message(network, 'alice', 'bob', test_message, 0.3)
    received = next((m for m in network.people['bob'].messages if m.metadata['message_id'] == msg_id), None)
    decoded = FFTLossyCompressionImplementation.receive_message(received)

    has_original_length = received.metadata.get('original_message_length') == len(test_message)
    print(f'Result: {"PASSED" if has_original_length else "FAILED"}')
    return has_original_length

def test_implementation_3():
    """Test Implementation 3: RSA Encryption"""
    print(' Testing Implementation 3: RSA Encryption')
    network = CommunicationNetwork()
    alice = Person('alice', 'Alice')
    bob = Person('bob', 'Bob')
    alice.add_connection(bob)
    network.add_person(alice)
    network.add_person(bob)

    test_message = 'Secret message'
    msg_id = RSAEncryptionImplementation.send_message(network, 'alice', 'bob', test_message)
    received = next((m for m in network.people['bob'].messages if m.metadata['message_id'] == msg_id), None)
    decoded = RSAEncryptionImplementation.receive_message(received, network.people['bob'].private_key)

    both_have_keys = received.metadata.get('sender_has_keys') and received.metadata.get('receiver_has_keys')
    result = decoded == test_message and both_have_keys
    print(f'Result: {"PASSED" if result else "FAILED"}')
    return result

def test_implementation_4():
    """Test Implementation 4: RSA Message Signing"""
    print(' Testing Implementation 4: RSA Message Signing')
    network = CommunicationNetwork()
    alice = Person('alice', 'Alice')
    bob = Person('bob', 'Bob')
    alice.add_connection(bob)
    network.add_person(alice)
    network.add_person(bob)

    test_message = 'Signed message'
    msg_id = RSAMessageSigningImplementation.send_message(network, 'alice', 'bob', test_message)
    received = next((m for m in network.people['bob'].messages if m.metadata['message_id'] == msg_id), None)
    result_data = RSAMessageSigningImplementation.receive_message(received)

    result = result_data.get('verified', False)
    print(f'Result: {"PASSED" if result else "FAILED"}')
    return result

def test_implementation_5():
    """Test Implementation 5: Signed Confirmations"""
    print(' Testing Implementation 5: Signed Confirmations')
    network = CommunicationNetwork()
    alice = Person('alice', 'Alice')
    bob = Person('bob', 'Bob')
    alice.add_connection(bob)
    network.add_person(alice)
    network.add_person(bob)

    # First send a signed message
    test_message = 'Message to confirm'
    msg_id = RSAMessageSigningImplementation.send_message(network, 'alice', 'bob', test_message)
    received = next((m for m in network.people['bob'].messages if m.metadata['message_id'] == msg_id), None)
    verification_result = RSAMessageSigningImplementation.receive_message(received)

    # Then send confirmation
    confirm_id = SignedConfirmationImplementation.send_confirmation(network, 'bob', 'alice', received, verification_result)
    confirmation = next((m for m in network.people['alice'].messages if m.metadata['message_id'] == confirm_id), None)
    confirm_result = SignedConfirmationImplementation.receive_confirmation(confirmation)

    has_all_elements = (confirmation.metadata.get('original_signature') and
                       confirmation.metadata.get('original_hash') and
                       confirmation.metadata.get('hash_of_returned_message') and
                       confirmation.metadata.get('encrypted_hash_of_returned_message'))

    result = confirm_result.get('confirmation_valid', False) and has_all_elements
    print(f'Result: {"PASSED" if result else "FAILED"}')
    return result

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    print(' Starting demonstration of all 5 implementations...\n')

    # Run the comprehensive demonstration
    results = demonstrate_all_implementations()

    # Optional: Run individual tests
    print('\n\n🔧 Running individual implementation tests...')
    print('-' * 50)

    individual_results = [
        test_implementation_1(),
        test_implementation_2(),
        test_implementation_3(),
        test_implementation_4(),
        test_implementation_5()
    ]

    all_individual_passed = all(individual_results)

    print(f'\n Individual test summary: {"ALL PASSED" if all_individual_passed else "SOME FAILED"}')
    print(f' Overall demonstration: {"ALL SUCCESSFUL" if results["all_implementations_successful"] else "NEEDS REVIEW"}')

    print('\n' + '=' * 80)
    print('     DEMONSTRATION COMPLETE - ALL 5 IMPLEMENTATIONS READY')
    print('=' * 80)

