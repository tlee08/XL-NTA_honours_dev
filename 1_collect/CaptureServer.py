from HelperFuncs import *


"""
**************************************************************************************************
            SETTING NETWORK INTERFACES TO MONITOR
**************************************************************************************************
"""


def setupProcs():
    """
    Stopping network capture interfering processes
    """
    output = evalSubprocess("ps -A -o pid= -o comm=")[0]
    for line in output.splitlines():
        pid, comm = line.strip().split(maxsplit=1)
        if comm in INTERFERING_PROCS:
            evalSubprocess(f"sudo kill -19 {pid}")


def setupCSI(phyname, ifname, mac_addr=None):
    # f"sudo iw dev mon0 del" # TRY RESETTING mon0
    # Making a base64 encoded parameter string to configure the wlan extractor
    command = f"mcp -c 11/20 -C 1 -N 1"
    if mac_addr:
        command += f" -m {mac_addr}"
    csi_config = evalSubprocess(command)[0]
    evalSubprocess(f"nexutil -I{ifname} -s500 -b -l34 -v{csi_config}")
    evalSubprocess(f"sudo iw phy {phyname} interface add mon0 type monitor")
    evalSubprocess(f"sudo ifconfig mon0 up")
    logging.info(getIWInfo(ifname))


def setupWLAN(ifname, channel, bdwth, txpw):
    """
    NOTE: the interface to listen to is f"{ifname}mon"
    """
    ifmon = f"{ifname}mon"
    # From https://github.com/morrownr/Monitor_Mode
    evalSubprocess(f"sudo ip link set dev {ifname} down")
    evalSubprocess(f"sudo iw dev {ifname} set monitor none")
    evalSubprocess(f"sudo ip link set dev {ifname} name {ifmon}")
    evalSubprocess(f"sudo ip link set dev {ifmon} up")
    evalSubprocess(f"sudo iw dev {ifmon} set channel {channel} {bdwth}")
    evalSubprocess(f"sudo iw dev {ifmon} set txpower fixed {txpw}")
    logging.info(getIWInfo(ifmon))


def setupCapture(mac_addr=None):
    # Stopping processes that interfere with capture
    setupProcs()
    # Setting up CSI capture
    phyname_csi = "phy0"
    ifname_csi = "wlan0"
    setupCSI(phyname_csi, ifname_csi, mac_addr)
    # Setting up WLAN capture
    ifname_wlan = "wlan1"
    channel_wlan = 11
    bdwth_wlan = "HT20"
    txpw = 1700
    setupWLAN(ifname_wlan, channel_wlan, bdwth_wlan, txpw)


"""
**************************************************************************************************
            RESETTING NETWORK INTERFACES TO MANAGED
**************************************************************************************************
"""


def resetProcs():
    output = evalSubprocess("ps -A -o pid= -o comm=")[0]
    for line in output.splitlines():
        pid, comm = line.strip().split(maxsplit=1)
        if comm in INTERFERING_PROCS:
            evalSubprocess(f"sudo kill -18 {pid}")


def resetCSI(phyname, ifname):
    pass


def resetWLAN(ifname):
    # From https://github.com/morrownr/Monitor_Mode
    ifmon = f"{ifname}mon"
    evalSubprocess(f"sudo ip link set dev {ifmon} down")
    evalSubprocess(f"sudo iw {ifmon} set type managed")
    evalSubprocess(f"sudo ip link set dev {ifmon} name {ifname}")
    evalSubprocess(f"sudo ip link set dev {ifname} up")
    # Connecting to network
    # evalSubprocess(f"nmcli dev connect {ifname}")
    # evalSubprocess(f"sudo ifmetric {ifname} 0")
    return


def resetCapture():
    # Resetting CSI interface
    phyname_csi = "phy0"
    ifname_csi = "wlan0"
    resetCSI(phyname_csi, ifname_csi)
    # Resetting WLAN interface
    ifname_wlan = "wlan1"
    resetWLAN(ifname_wlan)
    # Starting the stopped network processes again
    resetProcs()
    return


"""
**************************************************************************************************
            CAPTURING NETWORK TRAFFIC
**************************************************************************************************
"""


def captureCSI(root_dir, id, ifname):
    # Getting filename
    root_dir = os.path.join(root_dir, CSI_DIR)
    fp = generateFilename(root_dir, id, PCAP_EXT)
    # Starting capture
    command = f"sudo tcpdump dst port 5500 -i {ifname} -w {fp} -vv"
    return runSubprocess(command)


def captureWLAN(root_dir, id, ifname, mac_addr=None):
    ifmon = f"{ifname}mon"
    # Getting filename
    root_dir = os.path.join(root_dir, WLAN_DIR)
    fp = generateFilename(root_dir, id, PCAP_EXT)
    # Starting capture
    command = f"sudo tcpdump -i {ifmon} -w {fp} -vv"
    if mac_addr:
        command += f" ether host {mac_addr}"
    return runSubprocess(command)


def startCapture(root_dir, id, mac_addr=None):
    procs = []
    # Running CSI collection
    ifname_csi = "wlan0"
    p = captureCSI(
        root_dir,
        id,
        ifname_csi,
    )
    procs.append(p)
    # Running WLAN collection
    ifname_wlan = "wlan1"
    p = captureWLAN(
        root_dir,
        id,
        ifname_wlan,
        mac_addr,
    )
    procs.append(p)
    return procs


def stopCapture(procs):
    for p in procs:
        try:
            termProcess(p.pid)
            p.communicate()
        except Exception as e:
            logging.warning(f"Could not stop process {p.pid} - {e}")
    return []


"""
**************************************************************************************************
            SAVING CAPTURE METADATA FROM VICTIM
**************************************************************************************************
"""


def saveMetadata(root_dir, id, metadata_str):
    # Getting filename
    root_dir = os.path.join(root_dir, METADATA_DIR)
    fp = generateFilename(root_dir, id, METADATA_EXT)
    # Saving metadata
    metadata_dict = metadataUnpack(metadata_str)
    with open(fp, "w") as f:
        json.dump(metadata_dict, f, indent=2)
    return


"""
**************************************************************************************************
            RUNNING SINGLE EXPERIMENT
**************************************************************************************************
"""


def serverCaptureTraffic(sock):
    mac_addr = None
    root_dir = CAPTURES_DIR
    id = -1
    procs = []
    logging.info("Starting Capture Traffic...")
    while True:
        message = recv(sock)
        if message == MAC_HEADER:
            mac_addr = recv(sock)
            setupCapture(mac_addr)
        elif message == URL_HEADER:
            url = recv(sock)
            root_dir = os.path.join(CAPTURES_DIR, os.path.split(url)[1])
            id = generateFilenameID(root_dir)
        elif message == START_CAPTURE:
            procs = startCapture(root_dir, id, mac_addr)
        elif message == END_CAPTURE:
            procs = stopCapture(procs)
        elif message == METADATA_HEADER:
            metadata = recv(sock)
            saveMetadata(root_dir, id, metadata)
        elif message == "":
            procs = stopCapture(procs)
            closeSocket(sock)
            return


def run_server():
    makeLogger("CaptureServer")
    # # NOTE: NEED TO CHANGE TO ETH IP - PORT CAN STAY THE SAME
    server_sock = startSocket("10.0.2.2", 8080)
    listenConnection(server_sock, 1)
    while True:
        client_sock = acceptConnection(server_sock)
        serverCaptureTraffic(client_sock)


if __name__ == "__main__":
    try:
        run_server()
    except Exception as e:
        print(e)
    finally:
        resetCapture()
