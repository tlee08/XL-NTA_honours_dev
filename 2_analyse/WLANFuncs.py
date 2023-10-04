# from scapy.all import *

import struct
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from glob import glob
import os
from subprocess import Popen, PIPE, run
import regex as re
import json


"""
**************************************************************************************************
            DEFINING CONSTANTS
**************************************************************************************************
"""

H5_WLAN_KEY = "wlan"

# List all possible fields
command = "tshark -G | cut -f 3"

TSHARK_FIELDS = """
-e frame
-e frame.time_epoch
-e frame.time_relative
-e frame.number
-e frame.len

-e radiotap

-e wlan_radio
-e wlan_radio.phy
-e wlan_radio.data_rate
-e wlan_radio.channel
-e wlan_radio.frequency
-e wlan_radio.signal_dbm
-e wlan_radio.duration

-e wlan
-e wlan.fc.version
-e wlan.fc.type
-e wlan.fc.subtype
-e wlan.frag
-e wlan.seq
-e wlan.fc
-e wlan.ra
-e wlan.ta
-e wlan.da
-e wlan.sa
-e wlan.bssid
-e wlan.staa

-e ip
-e ip.version
-e ip.proto
-e ip.src
-e ip.dst
-e ip.len

-e tcp
-e tcp.srcport
-e tcp.dstport
-e tcp.stream
-e tcp.len

-e udp
-e udp.srcport
-e udp.dstport
-e udp.stream
-e udp.length
"""

"""
**************************************************************************************************
            OVERALL HELPER FUNCS
**************************************************************************************************
"""


def parse_json(fp):
    with open(fp, "r") as f:
        return json.load(f)


"""
**************************************************************************************************
            PCAP TO JSON FUNCS
**************************************************************************************************
"""

# Example bash command to convert pcap to decrypted json. Can pipe stdout to a file
"""tshark -nr {fp} -o wlan.enable_decryption:TRUE -o "uat:80211_keys:\"wpa-pwd\",\"67Marlie:sayplease\"" -T json -e frame"""


# VERY IMPORTANT FUNCTION - THIS DECRYPTS AND OUTPUTS KEY RAW FEATURES AS A DICT (LIKE A JSON)
def pcap_to_json(wlan_fp, json_dir, ssid, pwd, fields):
    # Getting filepaths
    name = os.path.splitext(os.path.split(wlan_fp)[1])[0]
    json_fp = os.path.join(json_dir, f"{name}.json")
    # Constructing tshark command to read, decrypt, and get key fields from pcap file
    command = rf"""tshark
        -nr {wlan_fp}
        -o wlan.enable_decryption:TRUE
        -o uat:80211_keys:"wpa-pwd","{pwd}:{ssid}"
        -T json
        {fields}"""
    # Running tshark command and piping output to json file
    with open(json_fp, "w") as f:
        run(command.split(), stdout=f, text=True, check=True)


"""
**************************************************************************************************
            JSON TO H5 FUNCS
**************************************************************************************************
"""


def json_to_df(json_fp):
    # Reading in JSON and initial reformatting (vectorising)
    with open(json_fp) as f:
        content = json.load(f)
    data = np.vectorize(lambda i: i["_source"]["layers"])(content)
    # Making df
    df = pd.DataFrame(data.tolist())
    # Unwrapping the lists to the first element value
    df = df.applymap(lambda x: x[0] if isinstance(x, list) else x)
    return df


def col_as_num(df, col):
    if col not in df.columns:
        series = np.nan
    else:
        series = pd.to_numeric(df[col])
    return series


def col_as_str(df, col):
    if col not in df.columns:
        series = np.nan
    else:
        series = df[col]
    return series


def add_is_upstream_attr(df, mac_src):
    # mac_src defines whether it is upstream/downstream
    # frames from mac_src is upstream, frames to mac_src is downstream
    # Checking mac_src is at either sa (source) or da (destination) for ALL frames
    has_mac = (df["wlan.sa"] == mac_src) | (df["wlan.da"] == mac_src)
    if np.mean(has_mac) != 1:
        raise ValueError(
            f"The mac, {mac_src}, is not in all packets. The frame numbers without the mac address are:"
            + f"{df.index[np.logical_not(has_mac)]}"
        )
    # Determining whether the frame is upstreaming or downstreaming
    df["is_upstream"] = df["wlan.sa"] == mac_src
    return df


def add_stream_attr(df):
    # Adds column of IP stream "<source> -> <dest>"
    # All source will be the client's MAC address
    # MUST have the "is_upstream" column
    def _add_stream_attr(row):
        if row["is_upstream"] == True:
            return f"{row['ip.src']} -> {row['ip.dst']}"
        else:
            return f"{row['ip.dst']} -> {row['ip.src']}"

    df["stream"] = df.apply(_add_stream_attr, axis=1)
    return df


def df_wlan_clean(df):
    # Setting the correct column dtypes
    df["ip.src"] = col_as_str(df, "ip.src")
    df["ip.dst"] = col_as_str(df, "ip.dst")
    df["wlan_radio.channel"] = col_as_num(df, "wlan_radio.channel")
    df["frame.number"] = col_as_num(df, "frame.number")
    df["frame.len"] = col_as_num(df, "frame.len")
    df["frame.time_epoch"] = col_as_num(df, "frame.time_epoch")
    df["frame.time_relative"] = col_as_num(df, "frame.time_relative")
    df["ip.len"] = col_as_num(df, "ip.len")
    df["udp.length"] = col_as_num(df, "udp.length")
    df["udp.srcport"] = col_as_num(df, "udp.srcport")
    df["udp.dstport"] = col_as_num(df, "udp.dstport")
    df["udp.stream"] = col_as_num(df, "udp.stream")
    df["tcp.len"] = col_as_num(df, "tcp.len")
    df["tcp.stream"] = col_as_num(df, "tcp.stream")
    return df


def df_wlan_derive_attrs(df, mac_src):
    df = add_is_upstream_attr(df, mac_src)
    df = add_stream_attr(df)
    return df


def save_df_wlan(df, h5_fp):
    df.to_hdf(h5_fp, key=H5_WLAN_KEY, mode="w")


"""
**************************************************************************************************
            ANALYSING AND FILTERING STREAMS
**************************************************************************************************
"""


def list_convs(df):
    df_convs = (
        df.groupby("stream")
        .agg(
            {
                "frame.len": np.nansum,
                "ip.len": np.nansum,
                "udp.length": np.nansum,
            }
        )
        .sort_values("udp.length", ascending=False)
    )
    return df_convs


def list_convs_TODO(df):
    h5_fp = r"/Users/timothylee/Desktop/Uni/Yr5/Honours/captures/v=L_LUpnjgPso/wlan_h5/cap_21.h5"

    df = pd.read_hdf(h5_fp, key=H5_WLAN_KEY)
    df_convs = list_convs(df)

    # TODO: make a column called "to_keep", which specifies whether to keep the packet or not (based on the size of the conv stream it's from)
    to_keep = df_convs[df_convs["udp.length"] > "xxx"].index
    df_filtered = df[df["stream"] in to_keep]


"""
**************************************************************************************************
            ANALYSING AND VISUALISING DATA
**************************************************************************************************
"""


def gen_time_bins(series, interval):
    start = np.floor(series.min())
    end = np.ceil(series.max()) + interval
    bins = np.arange(start, end, interval)
    return pd.cut(series, bins=bins, include_lowest=True, labels=bins[:-1])


def wlan_cumsum_plot(h5_dir):
    """
    Plots all captures in the same wlan_h5 folder on a cumulative line graph, broken by 0.5s time increments.
    """
    fig, ax = plt.subplots()
    for fp in os.listdir(h5_dir):
        # Getting file paths
        name = os.path.splitext(fp)[0]
        print(f"{h5_dir} - {name}")
        h5_fp = os.path.join(h5_dir, f"{name}.h5")
        # Reading h5 dataframe
        df = pd.read_hdf(h5_fp, key=H5_WLAN_KEY)
        # Binning the data by 0.5 second intervals
        interval = 0.5
        df["time_bin"] = gen_time_bins(df["frame.time_relative"], interval)
        # Grouping the data by 0.5 seconds intervals and getting the total bytes
        # Getting the cumulative bytes across each interval
        len_cumsum = df.groupby("time_bin")["frame.len"].sum().cumsum().sort_index()
        # Plotting
        ax.plot(len_cumsum.index, len_cumsum, label=f"{name}")
    return fig, ax
