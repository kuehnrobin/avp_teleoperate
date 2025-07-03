#!/usr/bin/env python3

import sys
import time

try:
    from dynamixel_sdk import *
except ImportError:
    print("ERROR: dynamixel_sdk not found")
    sys.exit(1)

# Configuration
PROTOCOL_VERSION = 2.0
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 2000000
DXL_ID_1 = 1  # Pitch servo
DXL_ID_2 = 2  # Yaw servo

# Control table addresses
ADDR_PRESENT_POSITION = 132
ADDR_HARDWARE_ERROR_STATUS = 70

def read_current_positions():
    """Read current positions of both servos"""
    print("=== Reading Current Servo Positions ===")
    
    # Initialize port and packet handler
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if not port_handler.openPort():
        print("✗ Failed to open port")
        return
    
    # Set baudrate
    if not port_handler.setBaudRate(BAUDRATE):
        print("✗ Failed to set baudrate")
        port_handler.closePort()
        return
    
    try:
        # Read positions for both servos
        for servo_id in [DXL_ID_1, DXL_ID_2]:
            servo_name = "Pitch" if servo_id == DXL_ID_1 else "Yaw"
            
            # Check hardware error status first
            error_status, comm_result, error = packet_handler.read1ByteTxRx(
                port_handler, servo_id, ADDR_HARDWARE_ERROR_STATUS)
            
            if comm_result == COMM_SUCCESS and error == 0:
                if error_status != 0:
                    print(f"Servo {servo_id} ({servo_name}): Hardware Error Status = {error_status}")
                    continue
            
            # Read position
            position, comm_result, error = packet_handler.read4ByteTxRx(
                port_handler, servo_id, ADDR_PRESENT_POSITION)
            
            if comm_result == COMM_SUCCESS and error == 0:
                # Convert from Dynamixel units to degrees
                position_deg = position * 360.0 / 4096.0
                
                print(f"Servo {servo_id} ({servo_name}):")
                print(f"  Position: {position} units")
                print(f"  Position: {position_deg:.2f}°")
                print(f"  Radians: {position_deg * 3.14159/180:.6f}")
                print()
            else:
                print(f"✗ Failed to read position for Servo {servo_id}: {packet_handler.getTxRxResult(comm_result)}")
    
    finally:
        port_handler.closePort()

if __name__ == "__main__":
    read_current_positions()
