from GenerateClient import *

makeLogger("example")

ifname = "wlan0"
# url = "https://www.youtube.com/watch?v=L_LUpnjgPso"
url = "https://www.youtube.com/watch?v=t3XTxPwiHkE"

# input(">> Press enter to SETUP...")
metadata = metadataInit(
    mac=getMACAddr(ifname),
    url=url,
    vid_dur=10,
    quality="480p",
)
metadataUpdate(metadata, proposed_cap_start=getCurrentTime())

# input(">> Press enter to RECONNECT...")
NMConnectExisting("67Marlie", ifname)
# metadataUpdate(
#     metadata,
#     reconnected_time=getCurrentTime(),
#     if_info=getIWInfo(ifname),
#     ifconfig=getIfconfig(ifname),
#     iwconfig=getIwconfig(ifname),
#     ip=getIPAddr(ifname),
#     ssid=getSSID(ifname),
#     bssid=getBSSID(ifname),
#     channel=getChannel(ifname),
#     bandwidth=getBandwidth(ifname),
# )

# input(">> Press enter to START BROWSER...")
driver = browserOpen()
browserGoToURL(driver, url)
metadataUpdate(metadata, nav_start=getCurrentTime())
metadataUpdate(metadata, nav_end=getCurrentTime())
browserYTSetQuality(driver, metadata["quality"])
metadataUpdate(metadata, vid_start=getCurrentTime())
browserYTPressSpace(driver)
browserYTIsPlaying(driver)
browserYTPressSpace(driver)
browserYTIsPlaying(driver)
browserYTPlayVideo(driver, metadata["vid_dur"])
browserClose(driver)
metadataUpdate(metadata, vid_end=getCurrentTime())
metadataUpdate(metadata, proposed_cap_end=getCurrentTime())

# input(">> Press enter to SHOW METADATA...")
print(json.dumps(metadata, indent=2))
