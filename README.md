============================================================
TESTING UPDATED RLE FUNCTION STANDALONE
============================================================

🧪 Testing: Basic repetitive pattern
   Input: "aaaabbbcccc"
   Encoded: "4a3b4c"
   Compression: 11 → 6 chars
   Decoded: "aaaabbbcccc"
   ✅ Decode match: PASSED
   ✅ Length validation: PASSED

🧪 Testing: No repetition
   Input: "abcdef"
   Encoded: "1a1b1c1d1e1f"
   Compression: 6 → 12 chars
   Decoded: "abcdef"
   ✅ Decode match: PASSED
   ✅ Length validation: PASSED

🧪 Testing: Long repetition (>9 chars)
   Input: "aaaaaaaaaa"
   Encoded: "9a1a"
   Compression: 10 → 4 chars
   Decoded: "aaaaaaaaaa"
   ✅ Decode match: PASSED
   ✅ Length validation: PASSED

🧪 Testing: Multiple short repetitions
   Input: "aabbccddee"
   Encoded: "2a2b2c2d2e"
   Compression: 10 → 10 chars
   Decoded: "aabbccddee"
   ✅ Decode match: PASSED
   ✅ Length validation: PASSED

🧪 Testing: Single character
   Input: "a"
   Encoded: "1a"
   Compression: 1 → 2 chars
   Decoded: "a"
   ✅ Decode match: PASSED
   ✅ Length validation: PASSED

🧪 Testing: Repeated patterns
   Input: "aaabbbaaaccc"
   Encoded: "3a3b3a3c"
   Compression: 12 → 8 chars
   Decoded: "aaabbbaaaccc"
   ✅ Decode match: PASSED
   ✅ Length validation: PASSED

🧪 Testing: Mixed content
   Input: "hello world!!!"
   Encoded: "1h1e2l1o1 1w1o1r1l1d3!"
   Compression: 14 → 22 chars
   Decoded: "hello world!!!"
   ✅ Decode match: PASSED
   ✅ Length validation: PASSED

📊 Overall RLE Function Test: ✅ ALL PASSED
================================================================================
    COMPREHENSIVE DEMONSTRATION: ALL 5 IMPLEMENTATIONS
================================================================================

📡 Network setup complete: 5 people with connections
Network topology: Alice <-> Bob <-> Charlie <-> Diana
                         ↕               ↗
                        Eve  <-----------

============================================================
TESTING IMPLEMENTATION 1: UPDATED RUN-LENGTH ENCODING
============================================================
Testing message with repetitive patterns: "aaaaaabbbbbbccccccdddddd!!!!!!......"

🔤 IMPLEMENTATION 1: Run-Length Encoding (Updated)
Original message: "aaaaaabbbbbbccccccdddddd!!!!!!......"
Encoded message: "6a6b6c6d6!6."
Compression ratio: 0.333
Original length: 36
Compressed length: 12
Metadata indicates: run-length-encoded
Decoded message: "aaaaaabbbbbbccccccdddddd!!!!!!......"
Verified encoding type: run-length-encoded
Length validation: PASSED
✅ Implementation 1 Success: PASSED
✅ Metadata correctly indicates: run-length-encoded
✅ Length validation: PASSED

🧪 Testing RLE edge cases:
   Test 1: "aaaa"
   Result: PASSED - Encoded: "4a"
   Test 2: "abcd"
   Result: PASSED - Encoded: "1a1b1c1d"
   Test 3: "aaabbbccc"
   Result: PASSED - Encoded: "3a3b3c"
   Test 4: "a"
   Result: PASSED - Encoded: "1a"

============================================================
TESTING IMPLEMENTATION 2: FFT LOSSY COMPRESSION
============================================================

🎛️ Testing sender-specified loss level: 20%

🌊 IMPLEMENTATION 2: FFT Lossy Compression
Original message: "Hello World! This message will become blurry with fine details lost through FFT compression."
Sender-specified loss level: 20%
Original length: 92 characters
Frequencies kept: 102/128
Blurry message (fine details lost): True
Metadata specifies original length: 92
Decompressed message: "Ehkii)Upwp`(Kgiq%p`ptbjd!ylhk!bdftd_ ch}vuxukqh#ehs`adridmt%mrlt!rgvjphnEIXeomqxevq`ml)"
Original length from metadata: 92
Loss level was: 20%
Message was blurry: True
✅ Blurry effect achieved: YES (8.7% similarity)
✅ Metadata specifies original length: 92 chars

🎛️ Testing sender-specified loss level: 50%

🌊 IMPLEMENTATION 2: FFT Lossy Compression
Original message: "Hello World! This message will become blurry with fine details lost through FFT compression."
Sender-specified loss level: 50%
Original length: 92 characters
Frequencies kept: 64/128
Blurry message (fine details lost): True
Metadata specifies original length: 92
Decompressed message: "Ungo`1]Y~cg3Fbji"fZgynhb"wqqf)]gvu\['ewtxns)nYjj.ZhrS*fectftn)fp&ppno]pZ)ALZ^fpgq`vnhmy""
Original length from metadata: 92
Loss level was: 50%
Message was blurry: True
✅ Blurry effect achieved: YES (3.3% similarity)
✅ Metadata specifies original length: 92 chars

🎛️ Testing sender-specified loss level: 80%

🌊 IMPLEMENTATION 2: FFT Lossy Compression
Original message: "Hello World! This message will become blurry with fine details lost through FFT compression."
Sender-specified loss level: 80%
Original length: 92 characters
Frequencies kept: 25/128
Blurry message (fine details lost): True
Metadata specifies original length: 92
Decompressed message: "-Zrn^C\fjjOF+7[ceIRedqbaUPsumJ;ZdzraR9Zx
                                                               vPZZck_`CH`edCMbl}miPBdu
iU_ZmmpiCFEMG6QYhon~hipj_,"
Original length from metadata: 92
Loss level was: 80%
Message was blurry: True
✅ Blurry effect achieved: YES (1.1% similarity)
✅ Metadata specifies original length: 92 chars

============================================================
TESTING IMPLEMENTATION 3: RSA ENCRYPTION
============================================================

🔒 IMPLEMENTATION 3: RSA Encryption
Original message: "This is a secret message that requires RSA encryption!"
Generated keys for sender charlie
Generated keys for receiver eve
Encrypted message length: 316 bytes
Sender has public/private keys: ✓
Receiver has public/private keys: ✓
Encrypted with receiver's public key
Decrypted message: "This is a secret message that requires RSA encryption!"
Used receiver's private key for decryption
✅ Implementation 3 Success: PASSED
✅ Both parties have public/private keys: True

============================================================
TESTING IMPLEMENTATION 4: RSA MESSAGE SIGNING
============================================================

✍️ IMPLEMENTATION 4: RSA Message Signing (Fixed)
Original message: "This is an important message that requires digital signature verification."
Generated RSA keys for sender diana
Message hash: 750
RSA-encrypted hash (signature): 1799
Signature created with sender's private key
Key modulus (n): 3233
Received signed message: "This is an important message that requires digital signature verification."
Original hash in message: 750
Decrypted hash from signature: 750
Recomputed hash of message: 750
Decrypted matches recomputed: True
Recomputed matches original: True
✅ Signature valid: True
✅ Implementation 4 Success: PASSED
✅ Signature is RSA-encrypted hash: True
✅ Receiver decrypted hash and confirmed match: True

============================================================
TESTING IMPLEMENTATION 5: SIGNED CONFIRMATIONS
============================================================

📋 IMPLEMENTATION 5: Signed Confirmation (Fixed)
Confirming message from diana to alice
Generated RSA keys for confirmer alice
✅ Confirmation includes all required elements:
   • Original received signature: 1799
   • Original hash: 750
   • Hash of returned message: 2804
   • Encrypted hash of returned message: 3060
   • Verification status: True
📨 Received signed confirmation:
   • Confirmation message: "CONFIRMATION: Message from diana has been received and successfully validated"
   • Original signature included: 1799
   • Original hash included: 750
   • Hash of returned message: 2804
   • Encrypted hash of returned message: 3060
   • Decrypted hash: 2804
   • Recomputed hash: 2804
   ✅ Confirmation signature valid: True
   ✅ Original message was verified: True
✅ Implementation 5 Success: PASSED
✅ Includes original signature: YES
✅ Includes original hash: YES
✅ Includes hash of returned message: YES
✅ Includes encrypted hash of returned message: YES

================================================================================
    FINAL SUMMARY: ALL 5 IMPLEMENTATIONS (UPDATED)
================================================================================

📊 Implementation Results:

1. Implementation 1: Updated Run-Length Encoding
   Requirement: Metadata indicates run-length encoding, improved algorithm with validation
   Status: ✅ PASSED

2. Implementation 2: FFT Lossy Compression
   Requirement: Sender specifies loss level, metadata has original length
   Status: ✅ PASSED

3. Implementation 3: RSA Encryption
   Requirement: Both sender and receiver have public/private keys
   Status: ✅ PASSED

4. Implementation 4: RSA Message Signing
   Requirement: Signature is RSA-encrypted hash, receiver verifies by decrypting
   Status: ✅ PASSED

5. Implementation 5: Signed Confirmations
   Requirement: Includes original signature, original hash, new hash, encrypted new hash
   Status: ✅ PASSED

================================================================================
    OVERALL RESULT: 🎉 ALL IMPLEMENTATIONS SUCCESSFUL
================================================================================

📈 Network Statistics:
   • Total messages sent: 7
   • People in network: 5
   • Message types demonstrated: 5
   • Message type breakdown:
     - run-length-encoded: 1 message(s)
     - FFT-Lossy: 3 message(s)
     - RSA: 1 message(s)
     - RSA-signed: 1 message(s)
     - signed-confirmation: 1 message(s)

🔤 Updated RLE Implementation Features:
   • Max consecutive character limit: 9 (prevents overflow)
   • Validation: Length verification enabled
   • Error handling: TypeError and ValueError exceptions
   • Metadata integration: Enhanced with validation flags


🔧 Running individual implementation tests...
--------------------------------------------------
🔤 Testing Implementation 1: Updated Run-Length Encoding

🔤 IMPLEMENTATION 1: Run-Length Encoding (Updated)
Original message: "aaaabbbbccccdddd"
Encoded message: "4a4b4c4d"
Compression ratio: 0.500
Original length: 16
Compressed length: 8
Metadata indicates: run-length-encoded
Decoded message: "aaaabbbbccccdddd"
Verified encoding type: run-length-encoded
Length validation: PASSED
   Result: PASSED
   Length validation: PASSED
🌊 Testing Implementation 2: FFT Lossy Compression

🌊 IMPLEMENTATION 2: FFT Lossy Compression
Original message: "Hello World Test"
Sender-specified loss level: 30%
Original length: 16 characters
Frequencies kept: 11/16
Blurry message (fine details lost): True
Metadata specifies original length: 16
Decompressed message: "K_fdp&Rtvxb!Zfqi"
Original length from metadata: 16
Loss level was: 30%
Message was blurry: True
   Result: PASSED
🔒 Testing Implementation 3: RSA Encryption

🔒 IMPLEMENTATION 3: RSA Encryption
Original message: "Secret message"
Generated keys for sender alice
Generated keys for receiver bob
Encrypted message length: 82 bytes
Sender has public/private keys: ✓
Receiver has public/private keys: ✓
Encrypted with receiver's public key
Decrypted message: "Secret message"
Used receiver's private key for decryption
   Result: PASSED
✍️ Testing Implementation 4: RSA Message Signing

✍️ IMPLEMENTATION 4: RSA Message Signing (Fixed)
Original message: "Signed message"
Generated RSA keys for sender alice
Message hash: 1029
RSA-encrypted hash (signature): 2830
Signature created with sender's private key
Key modulus (n): 3233
Received signed message: "Signed message"
Original hash in message: 1029
Decrypted hash from signature: 1029
Recomputed hash of message: 1029
Decrypted matches recomputed: True
Recomputed matches original: True
✅ Signature valid: True
   Result: PASSED
📋 Testing Implementation 5: Signed Confirmations

✍️ IMPLEMENTATION 4: RSA Message Signing (Fixed)
Original message: "Message to confirm"
Generated RSA keys for sender alice
Message hash: 572
RSA-encrypted hash (signature): 521
Signature created with sender's private key
Key modulus (n): 3233
Received signed message: "Message to confirm"
Original hash in message: 572
Decrypted hash from signature: 572
Recomputed hash of message: 572
Decrypted matches recomputed: True
Recomputed matches original: True
✅ Signature valid: True

📋 IMPLEMENTATION 5: Signed Confirmation (Fixed)
Confirming message from alice to bob
Generated RSA keys for confirmer bob
✅ Confirmation includes all required elements:
   • Original received signature: 521
   • Original hash: 572
   • Hash of returned message: 90
   • Encrypted hash of returned message: 2339
   • Verification status: True
📨 Received signed confirmation:
   • Confirmation message: "CONFIRMATION: Message from alice has been received and successfully validated"
   • Original signature included: 521
   • Original hash included: 572
   • Hash of returned message: 90
   • Encrypted hash of returned message: 2339
   • Decrypted hash: 90
   • Recomputed hash: 90
   ✅ Confirmation signature valid: True
   ✅ Original message was verified: True
   Result: PASSED

📋 Individual test summary: ✅ ALL PASSED
📋 Overall demonstration: ✅ ALL SUCCESSFUL

================================================================================
     DEMONSTRATION COMPLETE - ALL 5 IMPLEMENTATIONS READY
     (Updated with Enhanced RLE Algorithm)
================================================================================

========================================
EXAMPLE USAGE OF RLE FUNCTION:
========================================
Encoded result: {'metadata': {'encoding': 'RLE', 'original_length': 11, 'compressed_length': 6}, 'message_body': '4a3b4c'}
Decoded result: "aaaabbbcccc"
