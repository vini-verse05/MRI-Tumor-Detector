# aes_encryption.py
# Purpose: Encrypt and decrypt medical images using AES-256

import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

BASE_DIR = '/tmp/mri_app'
ENCRYPTED_DIR = os.path.join(BASE_DIR, 'encrypted_images')
os.makedirs(ENCRYPTED_DIR, exist_ok=True)

# 32 bytes = 256-bit AES key from environment variable
SECRET_KEY = os.environ.get('AES_SECRET_KEY', '').encode()

if not SECRET_KEY:
    raise ValueError('AES_SECRET_KEY environment variable not set! Must be exactly 32 bytes.')

assert len(SECRET_KEY) == 32, f'AES-256 requires exactly 32 bytes! Got {len(SECRET_KEY)} bytes.'


def encrypt_image(input_path, encrypted_filename=None):
    """
    Encrypt an image file using AES-256-CBC.

    Returns:
        output_path: Path where encrypted file was saved
    """

    # Read image as raw bytes
    with open(input_path, 'rb') as f:
        image_bytes = f.read()

    # Generate a fresh random IV (16 bytes) for each encryption
    iv = get_random_bytes(16)

    # Create AES cipher in CBC mode
    cipher = AES.new(SECRET_KEY, AES.MODE_CBC, iv)

    # Encrypt with PKCS7 padding
    ciphertext = cipher.encrypt(pad(image_bytes, AES.block_size))

    # Determine output filename
    if encrypted_filename is None:
        base = os.path.basename(input_path)
        encrypted_filename = base + '.enc'

    output_path = os.path.join(ENCRYPTED_DIR, encrypted_filename)

    # Save: IV (16 bytes) prepended to ciphertext
    with open(output_path, 'wb') as f:
        f.write(iv + ciphertext)

    print(f'Encrypted: {input_path} → {output_path}')
    return output_path


def decrypt_image(encrypted_path, output_path):
    """
    Decrypt an AES-256-CBC encrypted image file.

    Returns:
        output_path if successful
    """

    with open(encrypted_path, 'rb') as f:
        data = f.read()

    # First 16 bytes = IV, rest = ciphertext
    iv         = data[:16]
    ciphertext = data[16:]

    cipher      = AES.new(SECRET_KEY, AES.MODE_CBC, iv)
    image_bytes = unpad(cipher.decrypt(ciphertext), AES.block_size)

    with open(output_path, 'wb') as f:
        f.write(image_bytes)

    print(f'Decrypted: {encrypted_path} → {output_path}')
    return output_path


def decrypt_to_bytes(encrypted_path):
    """
    Decrypt directly to bytes (no temp file needed).
    Used by Flask app to keep decrypted data only in memory.
    """
    with open(encrypted_path, 'rb') as f:
        data = f.read()

    iv         = data[:16]
    ciphertext = data[16:]

    cipher = AES.new(SECRET_KEY, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size)


# ── Quick Test ──────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        enc_path = encrypt_image(test_img)
        dec_path = 'decrypted_test.jpg'
        decrypt_image(enc_path, dec_path)
        print(f'Test complete! Check {dec_path}')
    else:
        print('Usage: python aes_encryption.py path/to/image.jpg')