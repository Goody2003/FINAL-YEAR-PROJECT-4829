from cryptography.fernet import Fernet

# Generate a key
key = Fernet.generate_key()

# Save the key to a file for later use
with open("file_key.key", "wb") as key_file:
    key_file.write(key)

print("Key saved as 'file_key.key'")
