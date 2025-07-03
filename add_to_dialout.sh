#!/bin/bash
# Add the current user to the dialout group to grant serial port access.

USER=$(whoami)

echo "Adding user '$USER' to the 'dialout' group..."
sudo usermod -a -G dialout $USER

echo "Done. Please log out and log back in for the changes to take effect."
