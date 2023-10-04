# Nexmon Install Instructions

1. Flash RaspiOS 2022-01-28 on an SD card. Link:
```
https://downloads.raspberrypi.org/raspios_lite_armhf/images/raspios_lite_armhf-2022-01-28/
```
2. Enter the raspi-config application:
```zsh
sudo raspi-config
```
3. In the raspi-config application, set:
    a. **Time Zone**: `Localisation Options -> Timezone`.
    b. **WiFi Country to US**: `Localisation Options -> WLAN Country -> US`.
    c. **Expand File System**: `Advanced Options -> Expand Filesystem`.
    d. **Keyboard to US**: `Localisation Options -> Keyboard -> <current> -> Other -> English (US) -> English (US) -> The default for the keyboard layout -> no compose key`.
    e. **update tool**: `Update`.
    f. **select Network Manager**: `Advanced Options -> Network Config -> NetworkManager`.
4. Install the network manager (so we can use `nmcli`). Look here for reference https://pimylifeup.com/raspberry-pi-network-manager/:
```zsh
sudo apt -y install network-manager
```
5. Reboot:
```zsh
sudo reboot
```
6. Run the following:
```zsh
sudo apt update
sudo apt -y install automake bc bison flex gawk git libgmp3-dev libncurses5-dev libssl-dev libtool-bin make python-is-python2 qpdf tcpdump texinfo tmux
sudo reboot
```
7. Get the kernel headers with `rpi-source`:
```zsh
sudo wget https://raw.githubusercontent.com/RPi-Distro/rpi-source/master/rpi-source -O /usr/local/bin/rpi-source
sudo chmod +x /usr/local/bin/rpi-source
/usr/local/bin/rpi-source --tag-update
rpi-source
sudo reboot
```
8. Install Nexmon and Nexmon_CSI:
```zsh
sudo su
wget https://raw.githubusercontent.com/zeroby0/nexmon_csi/pi-5.10.92/install.sh -O install.sh
tmux new -c /home/pi -s nexmon 'bash install.sh | tee output.log'
```

## Helper Server (it sends instructions)

1. Start the server with:
```zsh
python broadcast_message.py
```
2. Get the following commands from the server with `wget`:
    a.
    ```zsh
    wget 10.0.1.18:8080/step6
    wget 10.0.1.18:8080/step7
    wget 10.0.1.18:8080/step8
    ```
