#!/bin/bash
# Set CPU governor to performance mode for better responsiveness
# Run with sudo if needed

echo "Setting CPU governor to performance mode..."

for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -f "$cpu" ]; then
        echo "performance" | sudo tee "$cpu" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ $(basename $(dirname $(dirname $cpu))): performance"
        else
            echo "✗ $(basename $(dirname $(dirname $cpu))): failed (may need sudo)"
        fi
    fi
done

echo ""
echo "Current CPU frequencies:"
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    if [ -f "$cpu" ]; then
        freq=$(cat "$cpu")
        freq_mhz=$((freq / 1000))
        echo "  $(basename $(dirname $(dirname $cpu))): ${freq_mhz} MHz"
    fi
done
