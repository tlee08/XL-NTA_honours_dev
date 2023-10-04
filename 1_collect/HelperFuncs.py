import os
import sys
import logging
import socket

from threading import Thread
from subprocess import Popen, PIPE

import time
import json
import struct
import numpy as np
import regex as re
from psutil import Process
import traceback

from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service


"""
**************************************************************************************************
            CONSTANTS
**************************************************************************************************
"""


MAC_HEADER = "mac_is"
START_CAPTURE = "start_capture"
END_CAPTURE = "end_capture"
IF_RECONNECTED = "if_reconnected"
URL_HEADER = "url_is"
NAV_STARTED = "nav_started"
NAV_ENDED = "nav_ended"
VIDEO_STARTED = "video_started"
VIDEO_ENDED = "video_ended"
METADATA_HEADER = "metadata_is"

MESSAGE_HEADER_STRUCT = ">Q"
MESSAGE_HEADER_SIZE = struct.calcsize(MESSAGE_HEADER_STRUCT)

CAPTURES_DIR = "CAPTURES"
CSI_DIR = "csi"
WLAN_DIR = "wlan"
PCAP_EXT = ".pcap"
METADATA_EXT = ".json"
METADATA_DIR = "metadata"
COUNTING_FILE = "counting.txt"

LOGGING_DIR = "LOGGING"
LOGGING_EXT = ".log"

INTERFERING_PROCS = [
    "wpa_action",
    "wpa_supplicant",
    "wpa_cli",
    "dhclient",
    "ifplugd",
    "dhcdbd",
    "dhcpcd",
    "udhcpc",
    "NetworkManager",
    "knetworkmanager",
    "avahi-autoipd",
    "avahi-daemon",
    "wlassistant",
    "wifibox",
    "net_applet",
    "wicd-daemon",
    "wicd-client",
    "iwd",
]

"""
**************************************************************************************************
            GET NETWORK INTERFACE INFO FUNCS
**************************************************************************************************
"""


def NMGetDevStatus():
    return evalSubprocess("nmcli dev status")[0]


def NMGetWiFiStatus():
    return evalSubprocess("nmcli dev wifi")[0]


def getIWInfo(ifname):
    return evalSubprocess(f"sudo iw {ifname} info")[0]


def getIfconfig(ifname):
    return evalSubprocess(f"sudo ifconfig {ifname}")[0]


def getIwconfig(ifname):
    return evalSubprocess(f"sudo iwconfig {ifname}")[0]


def getMACAddr(ifname):
    output = evalSubprocess(f"sudo iw {ifname} info")[0]
    pattern = r"addr\s+(\w+:\w+:\w+:\w+:\w+:\w+)"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return None


def getIPAddr(ifname):
    output = evalSubprocess(f"sudo ifconfig {ifname}")[0]
    pattern = r"inet\s+(\d+\.\d+\.\d+\.\d+)"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return None


def getSSID(ifname):
    output = evalSubprocess(f"sudo iw {ifname} info")[0]
    pattern = r"ssid\s+(\w+)"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return None


def getBSSID(ifname):
    output = evalSubprocess(f"sudo iw dev {ifname} info")[0]
    pattern = r"BSS\s+(\w{2}:\w{2}:\w{2}:\w{2}:\w{2}:\w{2})"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return None


def getChannel(ifname):
    output = evalSubprocess(f"sudo iw {ifname} info")[0]
    pattern = r"channel\s+(\d+)\s+\(\d+\s+MHz\)"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return None


def getBandwidth(ifname):
    output = evalSubprocess(f"sudo iw {ifname} info")[0]
    pattern = r"width:\s+(\d+)\s+MHz"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return None


"""
**************************************************************************************************
            SOCKET PROGRAMMING FUNCS
**************************************************************************************************
"""


def recv(sock):
    header = recvall(sock, MESSAGE_HEADER_SIZE)
    if header == b"":
        return ""
    header_len = struct.unpack(MESSAGE_HEADER_STRUCT, header)[0]
    message = recvall(sock, header_len).decode("utf-8")
    # logging.info(f"RECEIVED SIZE: {header_len}\n")
    # logging.info(f"RECEIVED HEADER: {header}\n")
    logging.info(f"RECEIVED: {message}\n")
    return message


def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return b""
        data += packet
    return data


def send(sock, message):
    header = struct.pack(MESSAGE_HEADER_STRUCT, len(message))
    sock.sendall(header)
    sock.sendall(message.encode("utf-8"))
    # logging.info(f"SENT SIZE: {len(message)}\n")
    # logging.info(f"SENT HEADER: {header}\n")
    logging.info(f"SENT: {message}\n")


def startSocket(ip, port):
    logging.info("Setting up and starting socket...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))
    logging.info(f"I am: {sock.getsockname()}\n")
    return sock


def makeConnection(sock, ip, port):
    logging.info("Connecting client to server...")
    sock.connect((ip, port))
    logging.info(f"Connected to: {sock.getpeername()}\n")
    return sock


def listenConnection(sock, n):
    logging.info("Listening for incoming connections...")
    sock.listen(n)


def acceptConnection(sock):
    logging.info("Waiting for client connections...")
    client_sock, client_addr = sock.accept()
    logging.info(f"Connected to: {client_sock.getpeername()}\n")
    return client_sock


def closeSocket(sock):
    sock.close()
    logging.info("Socket closed!\n")


"""
**************************************************************************************************
            PROCESSING FUNCS (THREADS, SUBPROCESSES)
**************************************************************************************************
"""


def runThread(func, args):
    logging.info(f"Starting thread: {func.__name__}...")
    t = Thread(target=func, args=args)
    t.start()
    return t


def runSubprocess(command: str):
    command_ls = command.split()
    return Popen(
        command_ls, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True, close_fds=False
    )


def evalSubprocess(command):
    p = runSubprocess(command)
    try:
        return p.communicate()
    except Exception as e:
        logging.warning(f"The process failed: {e}")
        return "", f"ERROR: {e}"


def seeSubprocess(command):
    def _seeSubprocess(process):
        for line in process.stdout:
            print(line, end="")

    p = runSubprocess(command)
    t = Thread(target=_seeSubprocess, args=(p,), daemon=True)
    t.start()
    return p


def termProcess(pid):
    p = Process(pid)
    for c in p.children(recursive=True):
        evalSubprocess(f"sudo kill -15 {c.pid}")


"""
**************************************************************************************************
            LOGGING FUNCS
**************************************************************************************************
"""


def makeLogger(fp):
    # Preparing logger file
    if not os.path.isdir(LOGGING_DIR):
        os.makedirs(LOGGING_DIR)
    fp = os.path.join(LOGGING_DIR, f"{fp}{LOGGING_EXT}")
    print(fp)
    # Configuring logger
    format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        filename=fp,
        filemode="a",
        format=format,
        # datefmt="%Y-%m-%d %H:%M:%S.%f",
    )
    # Setting up logger to also write to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(format))
    console_handler.setStream(sys.stdout)
    logging.getLogger().addHandler(console_handler)
    return


"""
**************************************************************************************************
            GENERATE CLIENT HELPER FUNCS (TIME)
**************************************************************************************************
"""


def getCurrentTime():
    return time.time()


def waitTime(sleepSecs):
    sleepSecs_whole = int(sleepSecs)
    sleepSecs_remainder = sleepSecs - sleepSecs_whole
    for i in np.arange(sleepSecs_whole):
        print(". ", end="", flush=True)
        time.sleep(1)
    if sleepSecs_remainder:
        print(". ")
        time.sleep(sleepSecs_remainder)
    else:
        print()


"""
**************************************************************************************************
            CAPTURE SERVER HELPER FUNCS (FILE GETTING AND SAVING)
**************************************************************************************************
"""


def generateFilenameID(root_dir):
    """
    Reads the number stored in the "counting.txt" file in the root_dir.
    """
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    fp = os.path.join(root_dir, COUNTING_FILE)
    id = 0
    if not os.path.isfile(fp):
        with open(fp, "w") as f:
            f.write(str(id))
    with open(fp, "r") as f:
        id = int(f.read())
    with open(fp, "w") as f:
        f.write(str(id + 1))
    return id


def generateFilename(root_dir, id, ext):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    return os.path.join(root_dir, f"cap_{id}{ext}")


"""
**************************************************************************************************
            HANDLING METADATA FUNCS
**************************************************************************************************
"""


def metadataInit(**kwargs):
    metadata = {
        "mac": str(),
        "ip": str(),
        "ssid": str(),
        "bssid": str(),
        "channel": str(),
        "bandwidth": str(),
        "url": str(),
        "type": str(),
        "if_info": str(),
        "proposed_cap_start": float(),
        "proposed_cap_end": float(),
        "reconnected_time": float(),
        "nav_start": float(),
        "nav_end": float(),
        "vid_dur": float(),
        "vid_start": float(),
        "vid_end": float(),
    }
    for k, v in kwargs.items():
        metadata[k] = v
    return metadata


def metadataUpdate(metadata, **kwargs):
    for k, v in kwargs.items():
        metadata[k] = v
    return metadata


def metadataPack(metadata_dict):
    return json.dumps(metadata_dict)


def metadataUnpack(metadata_str):
    return json.loads(metadata_str)


"""
**************************************************************************************************
            PLACEHOLDER FUNCS
**************************************************************************************************
"""


def recvMessages(sock):
    while True:
        message = recv(sock)
        if message == "":
            break
        print(message)
    return


def sendMessages(sock):
    while True:
        try:
            message = input("Your message: ")
            send(sock, message)
        except Exception as e:
            break
    return


def ping(url):
    return seeSubprocess(f"ping {url}")


def tcpdump(ifname, fp=None, bpf=None, flags=None):
    command = f"tcpdump {bpf} -i {ifname} {flags}"
    if not fp is None:
        command += f" -w {fp}"
    return seeSubprocess(command)
