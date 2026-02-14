#!/usr/bin/env python3
"""Test Kasa lights directly - turn on, then off. Loads credentials from .env"""

import asyncio
import os
import re

# Load .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                        val = val[1:-1]
                    os.environ[key] = val

from kasa import Discover, Credentials, Module

async def _turn_on_bulb(dev):
    """Turn on a single bulb. LB130 color bulbs use set_hsv; LB100 uses turn_on + brightness."""
    await dev.turn_on()
    try:
        if Module.Light in dev.modules:
            light = dev.modules[Module.Light]
            # LB130 color bulb: set_hsv(0,0,100) = white at full brightness - more reliable
            if light.has_feature("hsv"):
                await light.set_hsv(0, 0, 100)
            elif light.has_feature("brightness"):
                await light.set_brightness(100)
    except Exception:
        pass
    await dev.update()

async def _control_device(ip: str, creds, on: bool):
    """Control a single device with a fresh connection - avoids shared-state issues."""
    dev = await Discover.discover_single(ip, credentials=creds)
    await dev.update()
    if on:
        await _turn_on_bulb(dev)
    else:
        await dev.turn_off()
    return dev.alias

async def main():
    username = os.environ.get("KASA_USERNAME", "").strip()
    password = os.environ.get("KASA_PASSWORD", "").strip()
    creds = Credentials(username, password) if username and password else None

    print("Discovering Kasa devices...")
    found = await Discover.discover(credentials=creds)
    if not found:
        print("No devices found.")
        return

    # Collect (ip, alias, model) - control Desk (LB130) FIRST since it was failing
    devices = [(ip, dev.alias or ip, dev.model) for ip, dev in found.items()]
    devices.sort(key=lambda x: (0 if "130" in str(x[2]) else 1, x[1]))  # LB130 before LB100
    for ip, alias, model in devices:
        print(f"  {alias} ({model}) at {ip}")

    # Control each device with a FRESH connection - avoids second device failing
    # Desk (LB130) first since it was the problematic one
    print("\nTurning lights ON (fresh connection per device)...")
    for i, (ip, alias, model) in enumerate(devices):
        try:
            await _control_device(ip, creds, on=True)
            print(f"  {alias}: on")
        except Exception as e:
            print(f"  {alias}: error - {e}, retrying in 2s...")
            await asyncio.sleep(2)
            try:
                await _control_device(ip, creds, on=True)
                print(f"  {alias}: on (retry ok)")
            except Exception as e2:
                print(f"  {alias}: retry failed - {e2}")
        if i < len(devices) - 1:
            await asyncio.sleep(2)

    print("\n(Lights should be ON now - check them!) Waiting 5 seconds...")
    await asyncio.sleep(5)

    print("\nTurning lights OFF...")
    for i, (ip, alias, model) in enumerate(devices):
        try:
            await _control_device(ip, creds, on=False)
            print(f"  {alias}: off")
        except Exception as e:
            print(f"  {alias}: error - {e}")
        if i < len(devices) - 1:
            await asyncio.sleep(2)

    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
