import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

if not ports:
    print("No devices found.")
else:
    print("Connected devices:")
    for port, desc, hwid in ports:
        print(f"Port: {port}, Description: {desc}, HWID: {hwid}")