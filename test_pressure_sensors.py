#!/usr/bin/env python3
"""
Test script to access Unitree hand pressure sensor data
"""
import numpy as np
import time
import argparse
import threading
import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, PressSensorState_

# Topic names for pressure sensor data
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

class PressureSensorTest:
    def __init__(self, networkInterface='enxa0cec8616f27'):
        """
        Initialize pressure sensor test
        
        Args:
            networkInterface: Network interface for CycloneDDS
        """
        print("Initializing Pressure Sensor Test...")
        
        # Initialize DDS
        ChannelFactoryInitialize(0, networkInterface)
        
        # Initialize subscribers for hand states (which contain pressure sensor data)
        self.left_hand_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.left_hand_subscriber.Init()
        
        self.right_hand_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.right_hand_subscriber.Init()
        
        # Data storage
        self.left_pressure_data = np.zeros(12)
        self.left_temperature_data = np.zeros(12)
        self.right_pressure_data = np.zeros(12)
        self.right_temperature_data = np.zeros(12)
        
        self.data_received = False
        self.running = False
        
        # Start subscriber thread
        self.subscriber_thread = threading.Thread(target=self._subscribe_pressure_data)
        self.subscriber_thread.daemon = True
        
        print("Pressure Sensor Test initialized successfully!")
        
    def _subscribe_pressure_data(self):
        """
        Background thread to continuously read pressure sensor data
        """
        print("Starting pressure sensor data subscription...")
        
        while self.running:
            try:
                # Read left hand state
                left_msg = self.left_hand_subscriber.Read()
                if left_msg is not None and len(left_msg.press_sensor_state) > 0:
                    press_sensor = left_msg.press_sensor_state[0]  # Get first pressure sensor state
                    self.left_pressure_data = np.array(press_sensor.pressure[:12])
                    self.left_temperature_data = np.array(press_sensor.temperature[:12])
                    self.data_received = True
                
                # Read right hand state
                right_msg = self.right_hand_subscriber.Read()
                if right_msg is not None and len(right_msg.press_sensor_state) > 0:
                    press_sensor = right_msg.press_sensor_state[0]  # Get first pressure sensor state
                    self.right_pressure_data = np.array(press_sensor.pressure[:12])
                    self.right_temperature_data = np.array(press_sensor.temperature[:12])
                    self.data_received = True
                    
            except Exception as e:
                print(f"Error reading pressure data: {e}")
                
            time.sleep(0.001)  # 1ms sleep to avoid busy waiting
            
    def start(self):
        """Start the pressure sensor test"""
        self.running = True
        self.subscriber_thread.start()
        
        # Wait for initial data
        print("Waiting for pressure sensor data...")
        timeout = 10.0  # 10 second timeout
        start_time = time.time()
        
        while not self.data_received and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if not self.data_received:
            print("Warning: No pressure sensor data received within timeout!")
            return False
            
        print("Pressure sensor data received successfully!")
        return True
        
    def stop(self):
        """Stop the pressure sensor test"""
        self.running = False
        if self.subscriber_thread.is_alive():
            self.subscriber_thread.join(timeout=1.0)
            
    def print_data(self):
        """Print current pressure sensor data"""
        if not self.data_received:
            print("No data available yet...")
            return
            
        print("\n" + "="*60)
        print("PRESSURE SENSOR DATA")
        print("="*60)
        
        # Left hand data
        print("\nLEFT HAND:")

        print(f"Pressures:     {[f'{p:.3f}' for p in self.left_pressure_data]}")
        print(f"Temperatures:  {[f'{t:.3f}' for t in self.left_temperature_data]}")
        print(f"Max Pressure:  {np.max(self.left_pressure_data):.3f}")
        print(f"Avg Pressure:  {np.mean(self.left_pressure_data):.3f}")
        
        # Right hand data
        print("\nRIGHT HAND:")
        print(f"Pressures:     {[f'{p:.3f}' for p in self.right_pressure_data]}")
        print(f"Temperatures:  {[f'{t:.3f}' for t in self.right_temperature_data]}")
        print(f"Max Pressure:  {np.max(self.right_pressure_data):.3f}")
        print(f"Avg Pressure:  {np.mean(self.right_pressure_data):.3f}")
        
    def get_pressure_data(self):
        """
        Get current pressure data
        
        Returns:
            tuple: (left_pressure, left_temp, right_pressure, right_temp)
        """
        return (self.left_pressure_data.copy(), 
                self.left_temperature_data.copy(),
                self.right_pressure_data.copy(), 
                self.right_temperature_data.copy())


def main():
    parser = argparse.ArgumentParser(description='Test Unitree hand pressure sensors')
    parser.add_argument('--cyclonedds_uri', type=str, default='enxa0cec8616f27', 
                       help='Network interface for CycloneDDS (default: enxa0cec8616f27)')
    parser.add_argument('--frequency', type=float, default=10.0,
                       help='Display frequency in Hz (default: 10.0)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Test duration in seconds (default: run indefinitely)')
    
    args = parser.parse_args()
    
    print(f"Starting pressure sensor test with args: {args}")
    
    # Create test instance
    pressure_test = PressureSensorTest(networkInterface=args.cyclonedds_uri)
    
    try:
        # Start the test
        if not pressure_test.start():
            print("Failed to start pressure sensor test!")
            return
            
        print(f"\nPressure sensor test running at {args.frequency} Hz")
        print("Press Ctrl+C to stop")
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            # Print data at specified frequency
            pressure_test.print_data()
            frame_count += 1
            
            # Check duration limit
            if args.duration and (time.time() - start_time) >= args.duration:
                print(f"\nTest completed after {args.duration} seconds")
                break
                
            # Sleep to maintain frequency
            time.sleep(1.0 / args.frequency)
            
            # Print statistics every 50 frames
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                actual_freq = frame_count / elapsed if elapsed > 0 else 0
                print(f"\nStatistics: {frame_count} frames, {elapsed:.1f}s elapsed, {actual_freq:.1f} Hz actual")
                
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping...")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        pressure_test.stop()
        print("Pressure sensor test stopped")


if __name__ == "__main__":
    main()
