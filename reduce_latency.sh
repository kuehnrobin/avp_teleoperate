#!/bin/bash
# This script reduces the latency of the FTDI USB-to-serial converter to 1ms.
# This is a common optimization for Dynamixel servos.

DEVICE_PATH="/sys/bus/usb-serial/devices/ttyUSB0"

if [ -e "$DEVICE_PATH" ]; then
    echo "Setting latency_timer for $DEVICE_PATH to 1"
    echo 1 | sudo tee "$DEVICE_PATH/latency_timer"
else
    echo "Device not found at $DEVICE_PATH"
    echo "Please check the device path and try again."
    echo "You can find the correct path by running: ls /sys/bus/usb-serial/devices/"
fi
