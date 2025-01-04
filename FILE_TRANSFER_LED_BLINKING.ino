#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h> // For HTTP server

// network credentials
const char* ssid = "GG";
const char* password = "Goodness19121";

ESP8266WebServer server(80); // HTTP server running on port 80

// GPIO pins for LEDs
const int greenLED = D1; // GPIO5
const int redLED = D2;   // GPIO4

void setup() {
  // Initialize LEDs
  pinMode(greenLED, OUTPUT);
  pinMode(redLED, OUTPUT);
  digitalWrite(greenLED, LOW);
  digitalWrite(redLED, LOW);
  
  // Initialize Serial Monitor
  Serial.begin(115200);
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(700);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi");
  Serial.print("NodeMCU IP Address: ");
  Serial.println(WiFi.localIP());

  // Define HTTP routes
  server.on("/SUCCESS", handleSuccess);  // Route for success
  server.on("/FAILURE", handleFailure);  // Route for failure
  server.onNotFound(handleNotFound);     // Handle invalid routes

  // Start the server
  server.begin();
  Serial.println("HTTP server started");
}

// Handle "/SUCCESS" endpoint
void handleSuccess() {
  digitalWrite(greenLED, HIGH);  // Turn on green LED
  digitalWrite(redLED, LOW);     // Turn off red LED
  server.send(200, "text/plain", "Green LED ON: SUCCESS");
}

// Handle "/FAILURE" endpoint
void handleFailure() {
  digitalWrite(greenLED, LOW);   // Turn off green LED
  digitalWrite(redLED, HIGH);    // Turn on red LED
  server.send(200, "text/plain", "Red LED ON: FAILURE");
}

// Handle invalid routes
void handleNotFound() {
  server.send(404, "text/plain", "Endpoint Not Found");
}

void loop() {
  server.handleClient(); // Handle incoming HTTP requests
}
