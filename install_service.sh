#!/bin/bash
# Install Jarvis as a systemd service for auto-start on boot.
# Run with: sudo ./install_service.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="${SUDO_USER:-$USER}"
SERVICE_FILE="$SCRIPT_DIR/jarvis.service"
SYSTEMD_DIR="/etc/systemd/system"

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo ./install_service.sh"
    exit 1
fi

# Create a copy with correct paths, user, and XDG_RUNTIME_DIR (for PipeWire)
SERVICE_UID=$(id -u "$SERVICE_USER")
sed -e "s|/home/pi/Documents/project_jarvis|$SCRIPT_DIR|g" \
    -e "s|^User=.*|User=$SERVICE_USER|" \
    -e "s|^Group=.*|Group=$(id -gn "$SERVICE_USER")|" \
    -e "s|/run/user/1000|/run/user/$SERVICE_UID|" \
    "$SERVICE_FILE" > "$SYSTEMD_DIR/jarvis.service"

echo "Installed jarvis.service to $SYSTEMD_DIR"
echo "  User: $SERVICE_USER (for speaker/audio)"
echo "  WorkingDirectory: $SCRIPT_DIR"
echo ""
echo "Enable and start:"
echo "  sudo systemctl enable jarvis"
echo "  sudo systemctl start jarvis"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status jarvis   # Check status"
echo "  journalctl -u jarvis -f        # Follow logs"
