from HelperFuncs import *

VIDEOS_LS = [
    # First batch: news
    # r"https://www.youtube.com/watch?v=t634q_Voeto",
    # r"https://www.youtube.com/watch?v=yve6qo6eowU",
    # r"https://www.youtube.com/watch?v=3InbMow9IYo",
    # Second batch: fireplace
    # r"https://www.youtube.com/watch?v=mkWKZWMokdI",
    # r"https://www.youtube.com/watch?v=A3gUpodXMv0",
    # r"https://www.youtube.com/watch?v=w_oGIbFjiCo",
    # Third batch: kaleidoscope
    r"https://www.youtube.com/watch?v=NSW5u1RTxEA",
    r"https://www.youtube.com/watch?v=gxxqdrrpgZc",
    r"https://www.youtube.com/watch?v=t6jlhqNxRYk",
]
ITERS = 100
VID_DUR = 180

"""
**************************************************************************************************
            MODIFY NETWORK INTERFACES/SETTINGS
**************************************************************************************************
"""


def NMRestart():
    evalSubprocess("systemctl restart NetworkManager")
    waitTime(3)


def NMConnectNew(ssid, pwd, ifname):
    output = evalSubprocess(
        f"nmcli dev wifi connect {ssid} password {pwd} ifname {ifname}"
    )[0]
    if f"Device '{ifname}' successfully activated" in output:
        logging.info("Sucessfully connected to network.")
        evalSubprocess(f"sudo ifmetric {ifname} 0")
        return True
    else:
        logging.info("Not connected to network.")
        return False


def NMConnectExisting(ssid, ifname):
    output = evalSubprocess(f"nmcli connection up {ssid} ifname {ifname}")[0]
    if f"Connection successfully activated" in output:
        logging.info(f"Sucessfully connected {ifname} to network {ssid}.")
        evalSubprocess(f"sudo ifmetric {ifname} 0")
        return True
    else:
        logging.info("Not connected to network.")
        return False


"""
**************************************************************************************************
            SELENIUM CHROMIUM FUNCS
**************************************************************************************************
"""


def safeFindElement(driver, by, value):
    return WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((by, value))
    )


def safeFindElements(driver, by, value):
    return WebDriverWait(driver, 10).until(
        EC.visibility_of_all_elements_located((by, value))
    )


def browserOpen():
    chrome_options = ChromeOptions()
    chrome_options.add_extension("chrome_extensions/adblock.crx")
    logging.info("Opening web window")
    chrome_service = Service("/usr/bin/chromedriver")
    driver = Chrome(service=chrome_service, options=chrome_options)
    browserCloseExtPopup(driver)
    return driver


def browserCloseExtPopup(driver):
    original_window = driver.current_window_handle
    while True:
        # Checking which is the adblocker popup
        for window in driver.window_handles:
            driver.switch_to.window(window)
            # Closing the adblocker popup and going back to original tab
            if "data:," != driver.current_url:
                driver.close()
                driver.switch_to.window(original_window)
                return
        # Waiting for popup to open
        time.sleep(1)


def browserGoToURL(driver, url):
    logging.info(f"Navigating to URL: {url}")
    driver.get(url)


def browserYTSetQuality(driver, quality):
    # Open the video settings menu
    safeFindElements(driver, By.CLASS_NAME, "ytp-settings-button")[0].click()
    # Open the quality submenu
    for i in safeFindElements(driver, By.CLASS_NAME, "ytp-menuitem-label"):
        if i.text == "Quality":
            i.click()
            break
    # Select the quality
    time.sleep(0.5)
    qualities = safeFindElements(driver, By.CSS_SELECTOR, ".ytp-menuitem-label")
    for i in qualities:
        try:
            if i.text == quality:
                i.click()
                logging.info(f"Setting YT video quality as {quality}")
                return
        except:
            pass
    qualitites_texts = ", ".join([i.text for i in qualities])
    raise ValueError(
        f"The quality, {quality}, is not a valid resolution option for this video."
        + f"Valid resolutions are: {qualitites_texts}"
    )


def browserYTIsPlaying(driver):
    play_pause_btn = safeFindElements(driver, By.CSS_SELECTOR, ".ytp-play-button")[0]
    # print(play_pause_btn.get_attribute("outerHTML"))
    attr = play_pause_btn.get_attribute("data-title-no-tooltip")
    res = attr == "Pause"  # "Play"
    logging.info(f"Checking whether the YT video is playing: (is {res})")
    return res


def browserYTPressSpace(driver):
    logging.info("Pressing the space bar on the YT video")
    safeFindElements(driver, By.ID, "movie_player")[0].send_keys(Keys.SPACE)


def browserYTPlayVideo(driver, sleep_secs):
    if not browserYTIsPlaying(driver):
        browserYTPressSpace(driver)
    logging.info(f"Playing video for {sleep_secs} seconds")
    waitTime(sleep_secs)


def browserClose(driver):
    logging.info("Closing web window")
    driver.close()


"""
**************************************************************************************************
            RUNNING INDIVIDUAL EXPERIMENTS
**************************************************************************************************
"""


def client_play_video(sock, url):
    ifname = "wlan0"
    logging.info("Starting Play Video...")
    metadata = metadataInit(
        mac=getMACAddr(ifname),
        url=url,
        vid_dur=VID_DUR,
        quality="480p",
    )
    try:
        # Sending MAC address to configure capture
        send(sock, MAC_HEADER)
        send(sock, metadata["mac"])
        # Sending URL to configure capture
        send(sock, URL_HEADER)
        send(sock, metadata["url"])
        # Sending message to start capture
        metadataUpdate(metadata, proposed_cap_start=getCurrentTime())
        send(sock, START_CAPTURE)
        # Waiting to let CaptureServer setup the capture
        waitTime(3)
        # Reconnecting to WiFi (so OEAP hanshake occurs)
        NMConnectExisting("67Marlie", ifname)
        metadataUpdate(
            metadata,
            reconnected_time=getCurrentTime(),
            if_info=getIWInfo(ifname),
            ifconfig=getIfconfig(ifname),
            iwconfig=getIwconfig(ifname),
            ip=getIPAddr(ifname),
            ssid=getSSID(ifname),
            bssid=getBSSID(ifname),
            channel=getChannel(ifname),
            bandwidth=getBandwidth(ifname),
        )
        send(sock, IF_RECONNECTED)
        # Setting up browser
        driver = browserOpen()
        metadataUpdate(metadata, nav_start=getCurrentTime())
        send(sock, NAV_STARTED)
        # Navigating to url
        browserGoToURL(driver, url)
        metadataUpdate(metadata, nav_end=getCurrentTime())
        send(sock, NAV_ENDED)
        # Playing video
        browserYTSetQuality(driver, metadata["quality"])
        metadataUpdate(metadata, vid_start=getCurrentTime())
        send(sock, VIDEO_STARTED)
        browserYTPlayVideo(driver, metadata["vid_dur"])
        # Closing browser
        browserClose(driver)
        metadataUpdate(metadata, vid_end=getCurrentTime())
        send(sock, VIDEO_ENDED)
        # Sending message to stop capture
        metadataUpdate(metadata, proposed_cap_end=getCurrentTime())
        send(sock, END_CAPTURE)
    except Exception as e:
        # Logging the error
        err = traceback.format_exc()
        logging.error(f"Failed at: {err}")
        metadataUpdate(metadata, error=err)
    finally:
        # Sending metadata for capture
        send(sock, METADATA_HEADER)
        send(sock, metadataPack(metadata))
        # Closing socket
        waitTime(2)
        closeSocket(sock)
        return


def startClientSocket(host_ip, host_port, peer_ip, peer_port):
    while True:
        try:
            sock = startSocket(host_ip, host_port)
            makeConnection(sock, peer_ip, peer_port)
            return sock
        except ConnectionRefusedError as e:
            # Connection refused
            raise e from e
        except OSError as e:
            # Address already in use
            host_port += 1
            logging.info(f"Address already in use. Trying with port {host_port}...")


def run_client():
    makeLogger("GenerateClient")
    # NOTE: NEED TO CHANGE TO ETH IPs - PORT CAN STAY THE SAME
    for i in np.arange(ITERS):
        for url in VIDEOS_LS:
            client_sock = startClientSocket("10.0.2.1", 8081, "10.0.2.2", 8080)
            client_play_video(client_sock, url)


if __name__ == "__main__":
    run_client()
