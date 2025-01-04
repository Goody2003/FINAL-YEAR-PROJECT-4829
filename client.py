import socket  # Importing the socket module for network communication
import hashlib  # Importing hashlib for secure hash functions
import os  # Importing os for interacting with the operating system
import base64  # Importing base64 for encoding and decoding binary data
from cryptography.fernet import Fernet  # Importing Fernet for symmetric encryption
import pandas as pd  # Importing pandas for data manipulation and analysis
import joblib  # Importing joblib for saving and loading Python objects
from sklearn.preprocessing import LabelEncoder  # Importing LabelEncoder for encoding categorical features
import psutil  # Importing psutil for system and process utilities
import customtkinter as ctk  # Importing customtkinter for custom GUI components
from tkinter.filedialog import askopenfilename  # Importing file dialog for file selection
from tkinter import messagebox  # Importing messagebox for alert dialogs
import threading  # Importing threading for concurrent execution
import time  # Importing time for time-related functions

# Load the encryption key
with open("file_key.key", "rb") as key_file:  # Open the key file in read-binary mode
    key = key_file.read()  # Read the encryption key

cipher = Fernet(key)  # Create a Fernet cipher object using the loaded key

# Load pre-trained models (Random Forest and Gradient Boosting)
rf_model = joblib.load('rf_model.pkl')  # Load the Random Forest model
gb_model = joblib.load('gb_model.pkl')  # Load the Gradient Boosting model

def calculate_checksum(filename):
    """Calculate the SHA-256 checksum of the file."""
    sha256 = hashlib.sha256()  # Create a new SHA-256 hash object
    with open(filename, 'rb') as f:  # Open the file in read-binary mode
        while chunk := f.read(1024):  # Read the file in chunks of 1024 bytes
            sha256.update(chunk)  # Update the hash object with the chunk
    return sha256.hexdigest()  # Return the hex representation of the checksum

def gather_input_data(filename):
    """Gather input data for anomaly prediction."""
    log_level_x = "INFO"  # Log level for start message
    log_level_y = "INFO"  # Log level for completion message
    message_x = "File transfer started"  # Start message
    message_y = "File transfer completed successfully"  # Completion message
    file_size = os.path.getsize(filename)  # Get the size of the file
    cpu_usage = psutil.cpu_percent(interval=1)  # Get current CPU usage percentage
    memory_usage = psutil.virtual_memory().percent  # Get current memory usage percentage
    disk_usage = psutil.disk_usage('/').percent  # Get current disk usage percentage
    server_status = "ACTIVE"  # Current server status
    response_time = 0.1  # Sample response time
    failure_code = 0  # failure code 
    failure_code = 1  # Initialize failure code to 1 For errors anomaly

    # Read file content to check for anomalies
    with open(filename, 'rb') as f:  # Open the file in read-binary mode
        file_content = f.read()  # Read the entire file content
        if b"ANOMALY DETECTED" in file_content:  # Check for anomaly detection
            server_status = "FAILURE"  # Update server status on anomaly detection
            response_time = 5.0  # Update response time on failure
            message_y = "File transfer failed due to an error."  # Update failure message
            failure_code = 1  # Set failure code to indicate an error

    # Return a dictionary containing various metrics related to the file processing
    return {
        'log_level_x': log_level_x,  # Log level for start message
        'message_x': message_x,  # Start message
        'file_name': filename,  # File name being processed
        'file_size': file_size,  # Size of the file
        'server_status': server_status,  # Current server status
        'response_time': response_time,  # Simulated response time
        'log_level_y': log_level_y,  # Log level for completion message
        'message_y': message_y,  # Completion message
        'cpu_usage': cpu_usage,  # Current CPU usage
        'memory_usage': memory_usage,  # Current memory usage
        'disk_usage': disk_usage,  # Current disk usage
        'failure_code': failure_code  # Sample failure code
    }

def preprocess_data(input_data):
    """Preprocess input data for model prediction."""
    df = pd.DataFrame([input_data])  # Create a DataFrame from the input data
    le = LabelEncoder()  # Create a LabelEncoder instance
    for column in df.columns:  # Iterate over each column in the DataFrame
        if df[column].dtype == 'object':  # Check if the column is of type object
            df[column] = le.fit_transform(df[column])  # Encode the categorical column
    return df  # Return the processed DataFrame

def predict_anomaly(input_data):
    """Predict anomalies using the pre-trained models."""
    processed_data = preprocess_data(input_data)  # Preprocess the input data
    rf_prediction = rf_model.predict(processed_data)  # Get prediction from Random Forest model
    gb_prediction = gb_model.predict(processed_data)  # Get prediction from Gradient Boosting model
    return rf_prediction, gb_prediction  # Return both predictions

class FileTransferApp:
    """Main application class for the file transfer client."""
    def __init__(self, root):
        self.root = root  # Store the root window
        self.root.title("Client Dashboard")  # Set the window title
        ctk.set_appearance_mode("System")  # Set the appearance mode for the GUI
        ctk.set_default_color_theme("blue")  # Set the default color theme for the GUI

        # File Selection Frame
        self.file_frame = ctk.CTkFrame(root)  # Create a frame for file selection
        self.file_frame.pack(padx=10, pady=10, fill="x")  # Pack the frame

        # Button to select a file
        self.select_button = ctk.CTkButton(self.file_frame, text="Select File", command=self.select_file)  # Button for file selection
        self.select_button.pack()  # Pack the button

        # Label to display file details
        self.file_details_label = ctk.CTkLabel(self.file_frame, text="File Details: Not selected")  # Label for displaying selected file details
        self.file_details_label.pack()  # Pack the label

        # Transfer Status Frame
        self.status_frame = ctk.CTkFrame(root)  # Create a frame for transfer status
        self.status_frame.pack(padx=10, pady=10, fill="x")  # Pack the frame

        # Label for transfer status
        self.status_label = ctk.CTkLabel(self.status_frame, text="Transfer Status")  # Label for transfer status
        self.status_label.pack()  # Pack the label

        # Textbox to display status messages
        self.status_textbox = ctk.CTkTextbox(self.status_frame, height=5, state="disabled")  # Textbox for transfer status messages
        self.status_textbox.pack(fill="x")  # Pack the textbox

        # Prediction Results Frame
        self.prediction_frame = ctk.CTkFrame(root)  # Create a frame for prediction results
        self.prediction_frame.pack(padx=10, pady=10, fill="x")  # Pack the frame

        # Label for prediction results
        self.prediction_label = ctk.CTkLabel(self.prediction_frame, text="Prediction Results")  # Label for prediction results
        self.prediction_label.pack()  # Pack the label

        # Label to display the actual prediction result
        self.prediction_result = ctk.CTkLabel(self.prediction_frame, text="No prediction yet.")  # Label for displaying prediction result
        self.prediction_result.pack()  # Pack the label

        # System Resources Frame
        self.system_frame = ctk.CTkFrame(root)  # Create a frame for system resource display
        self.system_frame.pack(padx=10, pady=10, fill="x")  # Pack the frame

        # Labels for system resource usage
        self.cpu_label = ctk.CTkLabel(self.system_frame, text="CPU Usage: ")  # Label for CPU usage
        self.cpu_label.pack()  # Pack the label

        self.memory_label = ctk.CTkLabel(self.system_frame, text="Memory Usage: ")  # Label for memory usage
        self.memory_label.pack()  # Pack the label

        self.disk_label = ctk.CTkLabel(self.system_frame, text="Disk Usage: ")  # Label for disk usage
        self.disk_label.pack()  # Pack the label

        # Initialize file path variable
        self.file_path = None  # Variable to store the selected file path

        # Buttons for transfer and client start
        self.start_transfer_button = ctk.CTkButton(self.root, text="Start Transfer", command=self.start_transfer)  # Button to start file transfer
        self.start_transfer_button.pack(padx=10, pady=10)  # Pack the button

        self.start_client_button = ctk.CTkButton(self.root, text="Start Client", command=self.start_client)  # Button to start the client
        self.start_client_button.pack(padx=10, pady=10)  # Pack the button

        # Start system resource update loop in a separate thread
        threading.Thread(target=self.update_system_resource_loop, daemon=True).start()  # Start resource updating thread

    def select_file(self):
        """Select a file for transfer."""
        file_path = askopenfilename(title="Select a file to send")  # Open file dialog to select a file
        if file_path:  # If a file is selected
            self.file_path = file_path  # Store the file path
            filename = os.path.basename(file_path)  # Get the base filename
            file_size = os.path.getsize(file_path)  # Get the size of the file
            file_type = filename.split('.')[-1]  # Get the file type/extension
            checksum = calculate_checksum(file_path)  # Calculate the file checksum
            # Format file details for display
            file_details = f"Filename: {filename}\nSize: {file_size} bytes\nType: {file_type}\nChecksum: {checksum}"
            self.file_details_label.configure(text=file_details)  # Update the label with file details
        else:
            messagebox.showwarning("No file selected", "Please select a file.")  # Show a warning if no file is selected

    def update_status(self, message):
        """Update the transfer status in the GUI."""
        self.status_textbox.configure(state="normal")  # Enable the textbox for editing
        self.status_textbox.insert("end", message + "\n")  # Insert the new status message
        self.status_textbox.configure(state="disabled")  # Disable the textbox after updating

    def update_prediction_result(self, prediction):
        """Update the prediction result in the GUI."""
        self.prediction_result.configure(text=f"Prediction: {prediction}")  # Update the prediction result label

    def update_system_resources(self, cpu, memory, disk):
        """Update the system resource usage in the GUI."""
        self.cpu_label.configure(text=f"CPU Usage: {cpu}%")  # Update CPU usage label
        self.memory_label.configure(text=f"Memory Usage: {memory}%")  # Update memory usage label
        self.disk_label.configure(text=f"Disk Usage: {disk}%")  # Update disk usage label

    def start_transfer(self):
        """Start the file transfer process."""
        if not self.file_path:  # Check if a file has been selected
            messagebox.showwarning("No file selected", "Please select a file before transferring.")  # Show warning if no file
            return  # Exit the method if no file is selected
        threading.Thread(target=self.transfer_file).start()  # Start file transfer in a new thread

    def transfer_file(self):
        """Transfer the file to the server."""
        try:
            input_data = gather_input_data(self.file_path)  # Gather input data for anomaly detection
            rf_prediction, gb_prediction = predict_anomaly(input_data)  # Predict anomalies using the models
            self.update_prediction_result(f"RF: {rf_prediction[0]}, GB: {gb_prediction[0]}")  # Update prediction results in the GUI

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # Create a socket connection
                s.connect(("127.0.0.1", 65432))  # Connect to the server

                original_filename = os.path.basename(self.file_path)  # Get the base filename
                s.sendall(original_filename.encode('utf-8') + b'\n')  # Send the filename to the server

                chunk_size = 3072  # Define the chunk size for file transfer
                with open(self.file_path, 'rb') as f:  # Open the file in read-binary mode
                    while chunk := f.read(chunk_size):  # Read the file in chunks
                        encrypted_chunk = cipher.encrypt(chunk)  # Encrypt the chunk
                        encoded_chunk = base64.b64encode(encrypted_chunk)  # Encode the encrypted chunk
                        s.sendall(encoded_chunk)  # Send the encoded chunk
                        print(f"Sent chunk size: {len(encoded_chunk)}")  # Log the size of the sent chunk

                s.sendall(b"TRANSFER_COMPLETE")  # Signal the end of the transfer
                print("Transfer complete signal sent.")  # Log transfer completion

            checksum = calculate_checksum(self.file_path)  # Calculate checksum of the sent file
            self.update_status(f"File sent successfully. Checksum: {checksum}")  # Update status with success message

        except socket.error as e:
            self.update_status(f"Socket error: {str(e)}")  # Log socket errors
        except Exception as e:
            self.update_status(f"Error during transfer: {str(e)}")  # Log any other errors

    def start_client(self):
        """Start the client to receive the server response."""
        threading.Thread(target=self.run_client).start()  # Start the client in a new thread

    def run_client(self):
        """Receive response from the server."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # Create a socket connection
                s.connect(("127.0.0.1", 65432))  # Connect to the server
                response = s.recv(1024)  # Receive response from the server
                self.update_status(response.decode())  # Update status with the received message
        except Exception as e:
            self.update_status(f"Error receiving: {str(e)}")  # Log any errors during reception

    def update_system_resource_loop(self):
        """Continuously update system resource usage in the GUI."""
        while True:
            cpu = psutil.cpu_percent(interval=1)  # Get current CPU usage
            memory = psutil.virtual_memory().percent  # Get current memory usage
            disk = psutil.disk_usage('/').percent  # Get current disk usage
            self.update_system_resources(cpu, memory, disk)  # Update the GUI with resource usage
            time.sleep(1)  # Pause for a second before the next update

if __name__ == "__main__":
    root = ctk.CTk()  # Create the main window
    app = FileTransferApp(root)  # Initialize the FileTransferApp instance
    root.mainloop()  # Run the GUI event loop