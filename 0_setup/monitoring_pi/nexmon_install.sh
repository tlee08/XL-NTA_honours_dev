#!/bin/sh

function setStatus () {
  echo "********** ----- Status: $1 ----- ***********"
  echo "$1" >> 'nexmoncsi_status.log'
}

case "$(uname -r)" in
    "5.10."*)
        echo "Running on kernel 5.10.y"
    ;;

    *)
        echo "This script is designed to be run on kernel version 5.10.y, but you seem to be running version $(uname -r)."
        echo "Please use Raspbian Buster Lite 2022-01-28 as indicated in the ReadMe to install Nexmon_CSI."
        exit 1
    ;;
esac

setStatus "Downloading Nexmon"
cd src
if [ -e "nexmon" ]; then
    echo "nexmon exists"
else
    echo "nexmon does not exist. Cloning repo now."
    git clone https://github.com/seemoo-lab/nexmon.git
fi
cd nexmon
git pull
NEXDIR=$(pwd)

setStatus "Building libISL"
cd $NEXDIR/buildtools/isl-0.10
autoreconf -f -i
./configure
make
make install
ln -s -f /usr/local/lib/libisl.so /usr/lib/arm-linux-gnueabihf/libisl.so.10

setStatus "Building libMPFR"
cd $NEXDIR/buildtools/mpfr-3.1.4
autoreconf -f -i
./configure
make
make install
ln -s -f /usr/local/lib/libmpfr.so /usr/lib/arm-linux-gnueabihf/libmpfr.so.4

setStatus "Setting up Build Environment"
cd $NEXDIR
source setup_env.sh
make

setStatus "Downloading Nexmon_CSI"
cd $NEXDIR/patches/bcm43455c0/7_45_189/
if [ -e "nexmon_csi" ]; then
    echo "nexmon_csi exists"
else
    echo "nexmon_csi does not exist. Cloning repo now."
    git clone https://github.com/seemoo-lab/nexmon_csi.git
fi

setStatus "Building and installing Nexmon_CSI"
cd nexmon_csi
git pull
make
make backup-firmware
make install-firmware

setStatus "Installing makecsiparams"
cd utils/makecsiparams
make
ln -s -f $PWD/makecsiparams /usr/local/bin/mcp

setStatus "Installing nexutil"
cd $NEXDIR/utilities/nexutil
make
make install

setStatus "Setting up Persistance"
cd $NEXDIR/patches/bcm43455c0/7_45_189/nexmon_csi/brcmfmac_5.10.y-nexmon
cp ./brcmfmac.ko $(modinfo brcmfmac -n)
depmod -a

setStatus "Completed"

