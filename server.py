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
import threading  # Importing threading for concurrent execution
import requests  # Importing requests for making HTTP requests (For ESP communication)
import time  # Importing time for time-related functions

# Load the encryption key
with open("file_key.key", "rb") as key_file:  # Open the key file in read-binary mode
    key = key_file.read()  # Read the encryption key

cipher = Fernet(key)  # Create a Fernet cipher object using the loaded key

# Load pre-trained models (Random Forest and Gradient Boosting)
rf_model = joblib.load('rf_model.pkl')  # Load the Random Forest model
gb_model = joblib.load('gb_model.pkl')  # Load the Gradient Boosting model

# ESP Node MCU Configuration
ESP_IP = "192.168.107.142"  # ESP Node MCU's IP address
ESP_PORT = 80  # Default HTTP port for communication with ESP


def notify_esp(status):
    """Send success or failure notification to the ESP Node MCU."""
    try:
        url = f"http://{ESP_IP}:{ESP_PORT}/{status}"  # URL for notification
        response = requests.get(url, timeout=5)  # Make an HTTP GET request
        if response.status_code == 200:  # Check if the notification was successful
            print(f"ESP notified successfully with status: {status}")
        else:
            print(f"Failed to notify ESP. HTTP status code: {response.status_code}")
    except Exception as e:
        print(f"Error notifying ESP: {e}")  # Log any errors that occur during notification


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
        'failure_code': failure_code  #  failure code
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


class ServerDashboard:
    """Server GUI Dashboard for monitoring file transfers and system resources."""
    def __init__(self, root):
        self.root = root  # Store the root window
        self.root.title("Server Dashboard 4829 Goodness Ekong")  #  window title

        ctk.set_appearance_mode("System")  # Set appearance mode for the GUI
        ctk.set_default_color_theme("blue")  # Set default color theme

        # Create frames for organizing the GUI layout
        self.connection_frame = ctk.CTkFrame(root)  # Frame for connection status
        self.connection_frame.pack(padx=10, pady=10, fill="x")  # Pack the frame

        self.incoming_frame = ctk.CTkFrame(root)  # Frame for incoming transfers
        self.incoming_frame.pack(padx=10, pady=10, fill="x")  # Pack the frame

        self.prediction_frame = ctk.CTkFrame(root)  # Frame for prediction results
        self.prediction_frame.pack(padx=10, pady=10, fill="x")  # Pack the frame

        self.system_frame = ctk.CTkFrame(root)  # Frame for system resource display
        self.system_frame.pack(padx=10, pady=10, fill="x")  # Pack the frame

        self.processed_frame = ctk.CTkFrame(root)  # Frame for processed files display
        self.processed_frame.pack(padx=10, pady=10, fill="x")  # Pack the frame

        # Connection Status Display
        self.connection_label = ctk.CTkLabel(self.connection_frame, text="Connection Status")  # Label for connection status
        self.connection_label.pack()  # Pack the label

        self.connection_textbox = ctk.CTkTextbox(self.connection_frame, height=2)  # Textbox for connection status
        self.connection_textbox.pack(fill="x")  # Pack the textbox

        # Incoming Transfers Display
        self.incoming_label = ctk.CTkLabel(self.incoming_frame, text="Incoming Transfers")  # Label for incoming transfers
        self.incoming_label.pack()  # Pack the label

        self.incoming_textbox = ctk.CTkTextbox(self.incoming_frame, height=5)  # Textbox for incoming file details
        self.incoming_textbox.pack(fill="x")  # Pack the textbox

        # Prediction Results Display
        self.prediction_label = ctk.CTkLabel(self.prediction_frame, text="Prediction Results")  # Label for prediction results
        self.prediction_label.pack()  # Pack the label

        self.prediction_result = ctk.CTkLabel(self.prediction_frame, text="No prediction yet.")  # Label for prediction result
        self.prediction_result.pack()  # Pack the label

        # System Resource Display
        self.system_label = ctk.CTkLabel(self.system_frame, text="System Resources")  # Label for system resources
        self.system_label.pack()  # Pack the label

        self.cpu_label = ctk.CTkLabel(self.system_frame, text="CPU Usage: ")  # Label for CPU usage display
        self.cpu_label.pack()  # Pack the label

        self.memory_label = ctk.CTkLabel(self.system_frame, text="Memory Usage: ")  # Label for memory usage display
        self.memory_label.pack()  # Pack the label

        self.disk_label = ctk.CTkLabel(self.system_frame, text="Disk Usage: ")  # Label for disk usage display
        self.disk_label.pack()  # Pack the label

        # Processed Files List
        self.processed_label = ctk.CTkLabel(self.processed_frame, text="Processed Files")  # Label for processed files
        self.processed_label.pack()  # Pack the label

        self.processed_textbox = ctk.CTkTextbox(self.processed_frame, height=5)  # Textbox for displaying processed files
        self.processed_textbox.pack(fill="x")  # Pack the textbox

    def update_incoming_file(self, filename, file_size, sender_ip, checksum):
        """Update the incoming file display with new file details."""
        self.root.after(0, lambda: self.incoming_textbox.insert("end", f"{filename} - {file_size} bytes - {sender_ip} - {checksum}\n"))

    def update_prediction_result(self, prediction):
        """Update the prediction result display."""
        self.root.after(0, lambda: self.prediction_result.configure(text=prediction))

    def update_system_resources(self, cpu, memory, disk):
        """Update the system resource display on the dashboard."""
        self.root.after(0, lambda: self.cpu_label.configure(text=f"CPU Usage: {cpu}%"))
        self.root.after(0, lambda: self.memory_label.configure(text=f"Memory Usage: {memory}%"))
        self.root.after(0, lambda: self.disk_label.configure(text=f"Disk Usage: {disk}%"))

    def add_processed_file(self, filename):
        """Add a processed file to the processed files list."""
        self.root.after(0, lambda: self.processed_textbox.insert("end", f"{filename}\n"))

    def update_error_message(self, message):
        """Display an error message in the incoming transfers area."""
        self.root.after(0, lambda: self.incoming_textbox.insert("end", f"Error: {message}\n"))

    def update_connection_status(self, addr):
        """Update the connection status display."""
        self.root.after(0, lambda: self.connection_textbox.insert("end", f"Connected by {addr}\n"))


def handle_client(conn, addr, buffer_size=2048):
    """Handle incoming client connections and process file transfer."""
    try:
        # Update GUI with connection status
        dashboard.update_connection_status(addr)  # Update the GUI with the client's address

        # Get the original filename from the client
        original_filename = conn.recv(1024).decode('utf-8').strip()  # Receive the filename
        print(f"Receiving file: {original_filename} from {addr}")  # Log the receiving file info
        file_path = original_filename  # Store the file path
        data_buffer = bytearray()  # Initialize a buffer for incoming data

        while True:
            data = conn.recv(buffer_size)  # Receive data from the client
            if b"TRANSFER_COMPLETE" in data:  # Check for completion signal
                # Remove the "TRANSFER_COMPLETE" signal from the buffer
                data = data.replace(b"TRANSFER_COMPLETE", b"")
                data_buffer.extend(data)  # Add data to the buffer
                break
            elif data:
                data_buffer.extend(data)  # Add data to the buffer if it's valid
            else:
                break

        # If data was received, process the file
        if len(data_buffer) > 0:
            try:
                # Decrypt data and save to file
                decrypted_data = cipher.decrypt(base64.urlsafe_b64decode(data_buffer))  # Decrypt the data
                with open(file_path, 'wb') as f:  # Write the decrypted data to a file
                    f.write(decrypted_data)

                # Validate checksum
                checksum = calculate_checksum(file_path)  # Calculate the file checksum
                dashboard.update_incoming_file(file_path, os.path.getsize(file_path), addr[0], checksum)  # Update GUI with file info

                # Gather input data and predict anomalies
                input_data = gather_input_data(file_path)  # Gather data for prediction
                rf_pred, gb_pred = predict_anomaly(input_data)  # Predict anomalies
                prediction = f"RF Prediction: {rf_pred[0]}, GB Prediction: {gb_pred[0]}"  # Format prediction results
                dashboard.update_prediction_result(prediction)  # Update GUI with prediction result

                # Notify ESP of success
                notify_esp("SUCCESS")  # Notify ESP about successful processing

                # Send a response file back to the client
                response_file = "response_file.txt"  # Define response file name
                with open(response_file, 'w') as f:  # Write response to the file
                    f.write("Server processed your file successfully.")
                conn.sendall(b"TRANSFER_COMPLETE")  # Notify client of transfer completion
                dashboard.add_processed_file(file_path)  # Update processed files list
            except Exception as e:
                print(f"Error during file processing: {e}")  # Log processing error
                notify_esp("FAILURE")  # Notify ESP of failure
                dashboard.update_error_message(f"Error: {e}")  # Update GUI with error message
        else:
            print("No data received from client.")  # Log if no data received
            notify_esp("FAILURE")  # Notify ESP of failure
    except Exception as e:
        print(f"Error handling client {addr}: {e}")  # Log any errors during handling
        notify_esp("FAILURE")  # Notify ESP of failure
    finally:
        conn.close()  # Always close the connection
        print(f"Connection with {addr} closed.")  # Log closing of connection


def update_system_resources():
    """Periodic function to update system resource usage on the dashboard."""
    while True:
        # Retrieve current system resource usage
        cpu_usage = psutil.cpu_percent(interval=1)  # Get CPU usage
        memory_usage = psutil.virtual_memory().percent  # Get memory usage
        disk_usage = psutil.disk_usage('/').percent  # Get disk usage
        
        # Update the GUI with the latest resource usage
        dashboard.update_system_resources(cpu_usage, memory_usage, disk_usage)  # Update GUI with resource usage
        time.sleep(5)  # Delay before the next update (every 5 seconds)


def run_server(host='127.0.0.1', port=65432, buffer_size=2048):
    """Start the server to accept incoming connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a TCP socket
    server_socket.bind((host, port))  # Bind the socket to the host and port
    server_socket.listen(5)  # Listen for incoming connections
    print(f"Server is listening on {host}:{port}...")  # Log server status

    while True:
        conn, addr = server_socket.accept()  # Accept a new connection
        print(f"Connected by {addr}")  # Log connection info
        # Handle client in a new thread for concurrency
        threading.Thread(target=handle_client, args=(conn, addr, buffer_size), daemon=True).start()


def start_bidirectional(host='127.0.0.1', port=65432, buffer_size=2048):
    """Start the GUI and server simultaneously."""
    root = ctk.CTk()  # Create the main window
    global dashboard  # Declare global dashboard instance
    dashboard = ServerDashboard(root)  # Initialize the ServerDashboard instance

    # Start the server in a separate thread
    threading.Thread(target=run_server, args=(host, port, buffer_size), daemon=True).start()

    # Start the system resource monitoring in a separate thread
    threading.Thread(target=update_system_resources, daemon=True).start()

    # Start the GUI main loop
    root.mainloop()  # Run the GUI event loop


if __name__ == "__main__":
    start_bidirectional()  # Start the application