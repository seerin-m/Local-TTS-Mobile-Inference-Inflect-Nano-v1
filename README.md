# Inflect-Nano TTS

This project demonstrates running inference for the **Inflect-Nano-v1** TTS model. It is an end-to-end Text-to-Speech (TTS) solution split into two parts:
1. **Python Server (`inflect-phone`)**: Handles text phonemization locally on your machine.
2. **Android App (`InflectTTS`)**: Runs the ONNX-based acoustic models to generate and play speech on an Android device.

## Demo Videos

### Demo 1
<video src="Demo1.mp4" controls width="100%"></video>

### Demo 2
<video src="Demo2.mp4" controls width="100%"></video>

## Prerequisites
* Python 3.13+
* Android Studio
* A physical Android device (for running the app)

---

## Setup Instructions

### 1. Start the Phonemization Server

1. Clone the repository:
   ```bash
   git clone https://github.com/seerin-m/Local-TTS-Mobile-Inference-Inflect-Nano-v1.git
   ```
2. Navigate into the repository directory:
   ```bash
   cd Local-TTS-Mobile-Inference-Inflect-Nano-v1
   ```
3. Navigate into the Python server folder:
   ```bash
   cd inflect-phone
   ```
4. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the server:
   ```bash
   uvicorn phoneme_server:app --host 0.0.0.0 --port 8765
   ```
   *(Keep this terminal window open while using the Android app).*

### 2. Configure Windows Firewall (Important!)
To allow your Android device to communicate with the Python server running on your Windows laptop, you must open port `8765`.

1. Open **Command Prompt** as an **Administrator**.
2. Run the following command:
   ```cmd
   netsh advfirewall firewall add rule name="TTS Server" dir=in action=allow protocol=TCP localport=8765
   ```

### 3. Setup the Android App

1. Open **Android Studio**.
2. Select **Open** and choose the `InflectTTS` folder from this repository.
3. Wait for Android Studio to sync the Gradle project.

#### Connecting the Real Mobile Device
To use the app on a physical Android device, the app needs to talk to the local Python server:
1. **Same Network:** Connect both your Android phone and your laptop to the **same Wi-Fi network**.
2. **Find Laptop IP:** Open a terminal/CMD on your laptop and run `ipconfig` (Windows) or `ifconfig` (Mac/Linux) to find your laptop's local IPv4 Address (e.g., `192.168.1.15`).
3. **Update the App:** In Android Studio, navigate to:
   `app/src/main/java/com/mobilerun/inflecttts/MainActivity.java`
4. Locate the `LAPTOP_IP` variable near the top of the file:
   ```java
   private static final String LAPTOP_IP = "your laptop ip";
   ```
5. Replace `"your laptop ip"` with your actual IPv4 address:
   ```java
   private static final String LAPTOP_IP = "192.168.1.15"; // Example
   ```

### 4. Build and Run
1. Connect your Android device to your laptop via USB (ensure **USB Debugging** is enabled in your phone's Developer Options).
2. Select your physical device from the target drop-down menu at the top of Android Studio.
3. Click the **Run** button (▶) or press `Shift + F10` to build and launch the app.
4. Type text into the app, press **Speak**, and the app will generate the speech!
