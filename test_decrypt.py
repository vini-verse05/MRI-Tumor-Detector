from security.aes_encryption import decrypt_image

# Replace with your encrypted filename
encrypted_file = "encrypted_images/1c1fdbac_Te-pi_33.jpg.enc"

output_file = "decrypted_test.jpg"

decrypt_image(encrypted_file, output_file)

print("Decrypted image saved as:", output_file)