#!/usr/bin/env python3
import os
import time
import socket
import select
import subprocess
import sys

NUM_NODES = 31
BASE_PORT = 5000

PCE_BIN = "/home/dj-tst/build/pce-0.2.2/build/bin/pce-macplus"
DISKS_BASE = "/home/dj-tst/build/pce-0.2.2/disks"

#ram {{ address = 0; size = 4096K; default = 0x00 }}
# Note: Removed the serial 1 (Printer) port. Port 0 acts as our single bidirectional "MIDI" cable.
CFG_TEMPLATE = """\
path = "rom"
path = {pce_bin}
path = "-."
memtest = 0
system {{ model = "mac-plus" }}
cpu {{ model = "68000"; speed = 0 }}
ram {{ address = 0; size = 2560K; default = 0x00 }}
rom {{ file = "mac-plus.rom"; address = 0x400000; size = 256K; default = 0xff }}
rom {{ address = 0xf80000; size = 256K; file = "macplus-pcex.rom"; default = 0xff }}
terminal {{ driver = "x11"; scale = 1; aspect_x = 3; aspect_y = 2 }}
sound {{ lowpass = 8000; driver = "sdl:wav=speaker.wav:lowpass=0:wavfilter=0" }}
keyboard {{ model = 0; intl = 0; keypad_motion = 0 }}
adb {{ mouse = true; keyboard = true; keypad_motion = false }}
rtc {{ file = "pram-mac-plus.dat"; romdisk = 0 }}
sony {{ enable = 1; insert_delay = 2 }}
scsi {{
    device {{ id = 6; drive = 128; vendor = "PCE     "; product = "PCEDISK         " }}
    device {{ id = 4; drive = 129 }}
}}

serial {{
    port = 0
    multichar = 1
    driver = "tcp:host=127.0.0.1:port={tcp_port}:server=1"
}}


disk {{ drive = 128; type = "auto"; file = "{disks_base}/node{folder_id}/disk1.hfs"; optional = 0 }}
disk {{ drive = 129; type = "auto"; file = "{disks_base}/node{folder_id}/disk2.img"; optional = 0 }}
"""

def main():
    print(f"[*] Generating configs for {NUM_NODES} Mac Plus nodes...")
    processes = []
    
    # 1. Generate configs and launch emulators
    for node_id in range(NUM_NODES):
        folder_id = node_id + 1
        tcp_port = BASE_PORT + node_id
        
        cfg = CFG_TEMPLATE.format(
            folder_id=folder_id,
            tcp_port=tcp_port,
            disks_base=DISKS_BASE,
            pce_bin=PCE_BIN
        )
        
        cfg_name = f"pce_node_{node_id}.cfg"
        with open(cfg_name, "w") as f:
            f.write(cfg)
            
        print(f"[*] Booting Node {node_id} on Port {tcp_port}...")
        p = subprocess.Popen([PCE_BIN, "-r", "-c", cfg_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        time.sleep(0.2) # Slight stagger so X11 doesn't choke

    print("[*] All emulators launched. Waiting 5 seconds for Mac OS to boot...")
    time.sleep(5)

    print("[*] Connecting Python Ring Router to emulator sockets...")
    sockets = []
    
    # 2. Connect to all listening PCE emulators
    for i in range(NUM_NODES):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = BASE_PORT + i
        connected = False
        while not connected:
            try:
                s.connect(('127.0.0.1', port))
                s.setblocking(False)
                connected = True
            except ConnectionRefusedError:
                time.sleep(0.5)
        sockets.append(s)
        print(f"    -> Connected to Node {i} (Port {port})")

    print("\n[+] RING NETWORK ESTABLISHED!")
    print("[+] Press Ctrl+C in this terminal to kill the cluster.\n")

    # 3. Route the Ring Traffic
    try:
        while True:
            # Wait until at least one socket has data to read
            readable, _, _ = select.select(sockets, [], [])
            
            for s in readable:
                idx = sockets.index(s)
                next_idx = (idx + 1) % NUM_NODES
                
                try:
                    data = s.recv(8192)
                    if not data:
                        raise ConnectionResetError
                        
                    # Push bytes to the next node in the ring
                    sockets[next_idx].sendall(data)
                    
                except (ConnectionResetError, BrokenPipeError):
                    print(f"[!] Node {idx} disconnected. Tearing down cluster.")
                    raise KeyboardInterrupt
                    
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        for s in sockets:
            s.close()
        for p in processes:
            p.terminate()
            p.wait()
        sys.exit(0)

if __name__ == "__main__":
    main()
