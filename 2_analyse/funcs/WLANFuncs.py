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

FIELDS = [
    #
    # ("frame",),
    # ("frame.time_epoch",),
    ("frame.time_relative", np.float64),
    ("frame.number", np.int64),
    ("frame.len", np.int32),
    #
    # ("radiotap", "<U30"),
    #
    # ("wlan_radio", "<30"),
    # ("wlan_radio.phy", "<U16"),
    # ("wlan_radio.data_rate", "<U16"),
    # ("wlan_radio.channel", "<U16"),
    # ("wlan_radio.frequency", "<U16"),
    ("wlan_radio.signal_dbm", np.int8),
    # ("wlan_radio.duration", "<U16"),
    #
    # ("wlan", "<U30"),
    # ("wlan.fc", "<U16"),
    ("wlan.fc.version", np.int8),
    ("wlan.fc.type", np.int8),
    ("wlan.fc.subtype", np.int8),
    ("wlan.frag", np.int8),
    ("wlan.seq", np.int16),
    # ("wlan.ra", "<U17"),
    # ("wlan.ta", "<U17"),
    ("wlan.da", "<U17"),
    ("wlan.sa", "<U17"),
    # ("wlan.bssid", "<U17"),
    # ("wlan.staa", "<U30"),
    #
    # ("ip", "<U30"),
    # ("ip.version", np.int8),
    # ("ip.proto", np.int8),
    ("ip.src", "<U15"),
    ("ip.dst", "<U15"),
    ("ip.len", np.int16),
    #
    # ("tcp", "<U30"),
    ("tcp.srcport", np.int16),
    ("tcp.dstport", np.int16),
    # ("tcp.stream", np.int32),
    # ("tcp.len", np.int16),
    #
    # ("udp", "<U30"),
    ("udp.srcport", np.int16),
    ("udp.dstport", np.int16),
    # ("udp.stream", np.int32),
    # ("udp.length", np.int16),
]

TSHARK_FIELDS = " \n".join([f"-e {i[0]}" for i in FIELDS])

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
def pcap_to_json(wlan_fp, json_fp, ssid, pwd, fields):
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


def df_wlan_clean(df, fields):
    # Filling in missing values
    df = df.fillna(0)
    # Setting the correct column dtypes
    for col, col_type in fields:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(col_type)
    return df


def df_wlan_derive_attrs(df, mac_src):
    df = add_is_upstream_attr(df, mac_src)
    # df = add_stream_attr(df)
    return df


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
