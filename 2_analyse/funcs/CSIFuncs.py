# %%

import struct
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from .HelperFuncs import *

"""
**************************************************************************************************
            DEFINING CONSTANTS (STRUCT TYPES AND NULL & PILOT SUBCARRIERS)
**************************************************************************************************
"""

PCAP_GLOBAL_LEN = 24
PCAP_HEADER_LEN = 16
HEX_SIZE = 16

H5_CSI_KEY = "csi"


# Indexes of Null and Pilot OFDM subcarriers
# https://www.oreilly.com/library/view/80211ac-a-survival/9781449357702/ch02.html
nulls = {
    20: [
        x + 32
        for x in [
            -32,
            -31,
            -30,
            -29,
            -28,
            -27,
            31,
            30,
            29,
            28,
            27,
            0,
        ]
    ],
    40: [
        x + 64
        for x in [
            -64,
            -63,
            -62,
            -61,
            -60,
            -59,
            -1,
            63,
            62,
            61,
            60,
            59,
            1,
            0,
        ]
    ],
    80: [
        x + 128
        for x in [
            -128,
            -127,
            -126,
            -125,
            -124,
            -123,
            -1,
            127,
            126,
            125,
            124,
            123,
            1,
            0,
        ]
    ],
    160: [
        x + 256
        for x in [
            -256,
            -255,
            -254,
            -253,
            -252,
            -251,
            -129,
            -128,
            -127,
            -5,
            -4,
            -3,
            -2,
            -1,
            255,
            254,
            253,
            252,
            251,
            129,
            128,
            127,
            5,
            4,
            3,
            3,
            1,
            0,
        ]
    ],
}

pilots = {
    20: [
        x + 32
        for x in [
            -21,
            -7,
            21,
            7,
        ]
    ],
    40: [
        x + 64
        for x in [
            -53,
            -25,
            -11,
            53,
            25,
            11,
        ]
    ],
    80: [
        x + 128
        for x in [
            -103,
            -75,
            -39,
            -11,
            103,
            75,
            39,
            11,
        ]
    ],
    160: [
        x + 256
        for x in [
            -231,
            -203,
            -167,
            -139,
            -117,
            -89,
            -53,
            -25,
            231,
            203,
            167,
            139,
            117,
            89,
            53,
            25,
        ]
    ],
}


"""
**************************************************************************************************
            PARSING CSI PCAP FILE INTO A DICT
**************************************************************************************************
"""


def read_global(fc):
    # PCAP global header: https://wiki.wireshark.org/Development/LibpcapFileFormat/#global-header
    global_magic_number = fc[0:4]  # magic number
    # major version number
    version_major = struct.unpack(UINT16, fc[4:6])[0]
    # minor version number
    version_minor = struct.unpack(UINT16, fc[6:8])[0]
    # GMT to local correction
    thiszone = struct.unpack(UINT32, fc[8:12])[0]
    # accuracy of timestamps
    sigfigs = struct.unpack(UINT32, fc[12:16])[0]
    # max length of captured packets, in octets
    snaplen = struct.unpack(UINT32, fc[16:20])[0]
    # data link type
    network = struct.unpack(UINT32, fc[20:24])[0]
    return {
        "magic_number": global_magic_number,
        "version_major": version_major,
        "version_minor": version_minor,
        "thiszone": thiszone,
        "sigfigs": sigfigs,
        "snaplen": snaplen,
        "network": network,
    }


def read_frame(fc, ptr, bandwidth):
    # Initialising frame dict
    frame = {}
    # Getting number of subcarriers
    nsubs = get_nsubs(bandwidth)

    # PCAP packet header: https://wiki.wireshark.org/Development/LibpcapFileFormat/#global-header
    # timestamp seconds
    frame["ts_sec"] = struct.unpack(UINT32, fc[ptr : ptr + 4])[0]
    # timestamp microseconds
    frame["ts_usec"] = struct.unpack(UINT32, fc[ptr + 4 : ptr + 8])[0]
    # number of octets of packet saved in file
    frame["incl_len"] = struct.unpack(UINT32, fc[ptr + 8 : ptr + 12])[0]
    # actual length of packet
    frame["orig_len"] = struct.unpack(UINT32, fc[ptr + 12 : ptr + 16])[0]

    # Ethernet frame header (14 bytes long, can have a 4-byte footer)
    frame["eth_dmac"] = ":".join([i.to_bytes().hex() for i in fc[ptr + 16 : ptr + 22]])
    frame["eth_smac"] = ":".join([i.to_bytes().hex() for i in fc[ptr + 22 : ptr + 28]])
    frame["eth_ipv"] = struct.unpack(UINT16, fc[ptr + 28 : ptr + 30])[0]
    # Sometimes there is a eth frame check sequence
    frame["eth_fcs"] = fc[
        ptr + 76 + (nsubs * 4) : ptr + PCAP_HEADER_LEN + frame["incl_len"]
    ]

    # IP packet header (20 bytes long)
    # Not sure what the fields in this range mean. May not even be important
    frame["ip_filler"] = fc[ptr + 30 : ptr + 42]
    frame["ip_saddr"] = ".".join([str(i) for i in fc[ptr + 42 : ptr + 46]])
    frame["ip_daddr"] = ".".join([str(i) for i in fc[ptr + 46 : ptr + 50]])

    # UDP packet header (note that big endian is used)
    frame["udp_sport"] = struct.unpack(N_UINT16, fc[ptr + 50 : ptr + 52])[0]
    frame["udp_dport"] = struct.unpack(N_UINT16, fc[ptr + 52 : ptr + 54])[0]
    frame["udp_len"] = struct.unpack(N_UINT16, fc[ptr + 54 : ptr + 56])[0]
    frame["udp_checksum"] = fc[ptr + 56 : ptr + 58]

    # Nexmon CSI metadata
    frame["magic_bytes"] = fc[ptr + 58 : ptr + 60].hex()
    frame["rssi"] = struct.unpack("<b", fc[ptr + 60 : ptr + 61])[0]
    frame["fctl"] = fc[ptr + 61 : ptr + 62].hex()
    frame["mac"] = ":".join([i.to_bytes().hex() for i in fc[ptr + 62 : ptr + 68]])
    # seq = int.from_bytes(fc[ptr + 68 : ptr + 70], "little", signed=False)
    seq = struct.unpack(UINT16, fc[ptr + 68 : ptr + 70])[0]
    frame["fragn"] = seq & 15
    frame["seqn"] = seq >> 4
    frame["css"] = fc[ptr + 70 : ptr + 72].hex()
    frame["chanspec"] = fc[ptr + 72 : ptr + 74].hex()
    frame["chipv"] = fc[ptr + 74 : ptr + 76].hex()

    # Nexmon CSI data vector
    csi = fc[ptr + 76 : ptr + 76 + (nsubs * 4)]
    # Converting csi bytes to a int16 numpy array
    csi = np.frombuffer(csi, dtype=np.int16)
    # Convert csi into complex numbers (numbers are interleaved)
    csi = (csi[::2] + 1j * csi[1::2]).astype(np.complex64)
    # Shifting carriers to the centre of the spectrum
    frame["csi"] = np.fft.fftshift(csi)

    return frame


def check_frame(frame, bandwidth):
    # try:
    nsubs = get_nsubs(bandwidth)
    # Packet should be a certain length
    assert frame["incl_len"] == (14 + 20 + 8 + 18 + 4 * nsubs + len(frame["eth_fcs"]))
    # Ethernet source MAC address should be specific value
    assert frame["eth_smac"] == "4e:45:58:4d:4f:4e"
    # Ethernet destination MAC address should be specific value
    assert frame["eth_dmac"] == "ff:ff:ff:ff:ff:ff"
    # IP source address should be specific value
    assert frame["ip_saddr"] == "10.10.10.10"
    # IP destination address should be specific value
    assert frame["ip_daddr"] == "255.255.255.255"
    # UDP source port should be specific value
    assert frame["udp_sport"] == 5500
    # UDP destination port should be specific value
    assert frame["udp_dport"] == 5500
    # UDP packet should be certain length
    assert frame["udp_len"] == (8 + 18 + 4 * nsubs)
    # Magic bytes should be a specific value
    assert frame["magic_bytes"] == "1111"
    # except Exception as e:
    #     print(f"Error: {e}")
    #     print(f"{frame}")


def read_csi(fp):
    """
    Reads a csi pcap file in and returns a dict with the data.
    """
    # Initialising frames dict
    frames = {}
    # Reading in CSI pcap file as bytes
    fc = read_bytefile(fp)
    # Reading in global and overall settings
    fc_size = len(fc)
    frames["global"] = read_global(fc)
    nframes = get_nframes(fc)
    bandwidth = get_bandwidth(fc)
    nsubs = get_nsubs(bandwidth)
    # Frame values to read
    cols_to_read = [
        ("ts_sec", nframes, np.int32),
        ("ts_usec", nframes, np.int32),
        ("rssi", nframes, np.int8),
        ("fctl", nframes, np.uint8),
        ("mac", nframes, "<U17"),
        ("fragn", nframes, np.uint8),
        ("seqn", nframes, np.uint16),
        ("css", nframes, "<U4"),
        ("chanspec", nframes, "<U4"),
        ("chipv", nframes, "<U4"),
        ("csi", [nframes, nsubs], np.complex64),
    ]
    # Initialising np arrays to store pcap CSI values and metadata in
    for col, shape, dtype in cols_to_read:
        frames[col] = np.zeros(shape, dtype=dtype)
    # Offset for the global header
    ptr = PCAP_GLOBAL_LEN
    i = 0
    # Reading each frame
    while ptr < fc_size:
        # Reading and checking frame
        frame = read_frame(fc, ptr, bandwidth)
        check_frame(frame, bandwidth)
        # Storing values
        for col, shape, dtype in cols_to_read:
            frames[col][i] = frame[col]
        # Updating i and ptr
        i += 1
        ptr += PCAP_HEADER_LEN + frame["incl_len"]
    # Adding derivative columns
    if check_csi(frames["csi"]):
        frames["frame.number"] = np.arange(1, nframes + 1)
        frames["frame.time_epoch"] = frames["ts_sec"] + frames["ts_usec"] / np.power(
            10, 6
        )
        frames["frame.time_relative"] = (
            frames["frame.time_epoch"] - frames["frame.time_epoch"][0]
        )
    else:
        frames["frame.number"] = []
        frames["frame.time_epoch"] = []
        frames["frame.time_relative"] = []
    return frames


"""
**************************************************************************************************
            HELPER FUNCS TO PARSE CSI
**************************************************************************************************
"""


def read_bytefile(fp):
    with open(fp, "rb") as f:
        fc = f.read()
    return fc


def get_bandwidth(fc):
    # If there are no packets, then bandwidth is None
    if len(fc) < 36:
        return 0
    # This is where the incl_len bytes of the first frame are (after global header)
    incl_len = struct.unpack(UINT32, fc[32:36])[0]
    # The total header before csi data is 60 bytes (not incl. PCAP header in incl_len):
    # ETH header (14), IP header (20), UDP header (8), Nexmon CSI header (18)
    csi_len = incl_len - 60
    bandwidth = 20 * int(csi_len // (20 * 3.2 * 4))
    return bandwidth


def get_nframes(fc):
    # Initialising vals
    fc_size = len(fc)
    ptr = PCAP_GLOBAL_LEN
    i = 0
    # Counting the number of frames by going through all incl_len vals in file
    while ptr < fc_size:
        # Getting incl_len of current frame
        incl_len = struct.unpack(UINT32, fc[ptr + 8 : ptr + 12])[0]
        # Incrementing the number of frames val
        i += 1
        # Moving to the next frame in the file: incl_len + frame's header size
        ptr += PCAP_HEADER_LEN + incl_len
    # Checking that we correctly read the file. The ptr be at the file's "exact" end.
    assert ptr == fc_size
    return i


def get_nsubs(bandwidth):
    return int(bandwidth * 3.2)


def print_raw_bytes(fc, start=None, finish=None):
    if start == None:
        start = 0
    if finish == None:
        finish = len(fc)
    # Printing packet content as raw bytes
    for i in range(start, finish):
        print(fc[i].to_bytes().hex().ljust(4, " "), end=" ")
        if (i - start) % HEX_SIZE == HEX_SIZE - 1:
            print()
    print()


def make_subc_matrix(nframes, nsubs):
    # Getting parameters to make subcarrier matrix
    sub_min = -1 * nsubs / 2
    sub_max = nsubs / 2
    # Making subcarrier index matrix
    return np.tile(np.arange(sub_min, sub_max), nframes).reshape((nframes, -1))


"""
**************************************************************************************************
            PROCESS CSI MATRIX
**************************************************************************************************
"""


def check_csi(csi_matrix):
    # Checks whether the csi_matrix is valid and non-empty
    return (
        len(csi_matrix.shape) == 2
        and csi_matrix.shape[0] != 0
        and csi_matrix.shape[1] != 0
    )


def process_csi(_csi_matrix, rnull=False, rpilot=False, rm_outliers=None):
    """
    Preprocessing the CSI matrix by:
        - Setting null subcarriers to 0
        - Setting pilot subcarriers to 0
        - Setting outliers to the mean value for each subcarrier
    """
    csi_matrix = _csi_matrix.copy()
    bandwidth = int(csi_matrix.shape[1] / 3.2)
    # Setting null and pilot subcarriers to 0 (if specified)
    if rnull:
        csi_matrix[:, nulls[bandwidth]] = 0
    # Setting null and pilot subcarriers to 0 (if specified)
    if rpilot:
        csi_matrix[:, pilots[bandwidth]] = 0
    if rm_outliers:
        # rm_outlier is the # of std's away that outliers should be removed
        # Sets outlier signals to the mean subcarrier value
        for subc in np.arange(csi_matrix.shape[1]):
            subc_amp = np.abs(csi_matrix)[:, subc]
            subc_mean = np.mean(subc_amp)
            subc_std = np.std(subc_amp)
            extr_outliers = np.abs(subc_amp - subc_mean) >= rm_outliers * subc_std
            csi_matrix[extr_outliers, subc] = subc_mean
    return csi_matrix


def pad_csi(_csi_matrix, nframes):
    """
    Pads the CSI matrix so it has n rows (packets). The padding is 0s (or ave value of each subcarrier??)
    """
    csi_matrix = _csi_matrix.copy()
    if csi_matrix.shape[0] == nframes:
        return csi_matrix
    if csi_matrix.shape[0] > nframes:
        return csi_matrix[:nframes]
    else:
        # Padding the end frame rows array to get to size nframes
        # Padding with zeros (can also use mean of each subcarrier?)
        prows = nframes - csi_matrix.shape[0]
        return np.pad(
            csi_matrix, ((0, prows), (0, 0)), mode="constant", constant_values=0
        )


def make_freq_distr(csi_matrix, frame_bins):
    subc_matrix = make_subc_matrix(*csi_matrix.shape)
    H, xedges, yedges = np.histogram2d(
        subc_matrix.flatten(),
        csi_matrix.flatten(),
        bins=(csi_matrix.shape[1] - 1, frame_bins),
    )
    return H, xedges, yedges


"""
**************************************************************************************************
            FRAMES DICT TO DF FUNCS
**************************************************************************************************
"""


def frames_to_df(frames):
    # Process CSI (if possible)
    csi = frames["csi"]
    if check_csi(csi):
        # Processing csi info
        csi = process_csi(csi, True, True, 5)
    # Making CSI capture DataFrame
    df = pd.DataFrame(
        {
            "frame.number": frames["frame.number"],
            "frame.time_epoch": frames["frame.time_epoch"],
            "frame.time_relative": frames["frame.time_relative"],
            "fctl": frames["fctl"],
            "seqn": frames["seqn"],
            "fragn": frames["fragn"],
        },
    )
    # Setting the frame.number as the index
    df = df.set_index("frame.number")
    # Adding CSI subcarrier vals as columns
    for i in np.arange(csi.shape[1]):
        df[f"csi_{i}_r"] = np.int16(np.real(csi[:, i]))
        df[f"csi_{i}_i"] = np.int16(np.imag(csi[:, i]))
    return df


"""
**************************************************************************************************
            FRAMES DICT TO DF MULTIPROCESSING
**************************************************************************************************
"""


def csi_to_df_init_mp():
    import warnings

    warnings.filterwarnings("ignore")


def csi_to_df_mp(dir, name):
    print(f"{dir} - {name}")
    # Getting filepaths
    csi_fp = os.path.join(dir, "csi", f"{name}.pcap")
    h5_fp = os.path.join(dir, "csi_h5", f"{name}.h5")
    # If the h5 file already exists, then skip
    # if os.path.isfile(h5_fp):
    #     continue
    # Reading csi info (including metadata)
    frames = read_csi(csi_fp)
    # Format frames as a DataFrame
    df = frames_to_df(frames)
    # Saving dataframe as h5
    df.to_hdf(h5_fp, key=H5_CSI_KEY, mode="w")


"""
**************************************************************************************************
            DF TO CSI MATRIX FUNCS
**************************************************************************************************
"""


def df_to_csi_matrix(df):
    # Getting the shape of CSI matrix (# subcarriers, # frames)
    nsubs = int(np.sum(["csi_" in i for i in df.columns]) / 2)
    nframes = df.shape[0]
    # Initialising CSI matrix
    csi = np.zeros((nframes, nsubs), dtype=np.complex64)
    # Filling CSI matrix values
    for i in np.arange(nsubs):
        csi[:, i] = (df[f"csi_{i}_r"] + 1j * df[f"csi_{i}_i"]).astype(np.complex64)
    return csi


"""
**************************************************************************************************
            AGGREGATING CSI DATA FROM A GROUP OF CAPTURES
**************************************************************************************************
"""


# def combine_csis(fps, nframes):
#     """
#     Combines the csi values into a 3D array with:
#         - 0 axis: The individual csi capture matrix (i.e. the file)
#         - 1 axis: The frames (note: must be trimmed or padded - same size across all)
#         - 2 axis: The subcarrier
#     The values are the complex CSI values
#     """
#     # Getting the # of subcarriers in the csi data by
#     # reading in files and getting bandwidth of first valid capture file
#     nsubs = 0
#     for fp in fps:
#         fc = read_bytefile(fp)
#         bandwidth = get_bandwidth(fc)
#         if bandwidth != 0:
#             nsubs = get_nsubs(bandwidth)
#             break
#     # Initialising combined csi array
#     combined_csi = np.zeros((len(fps), nframes, nsubs), dtype=np.complex64)
#     # Reading in each CSI pcap
#     for i, fp in enumerate(fps):
#         # Reading CSI pcap
#         frames = read_csi(fp)
#         # Checking CSI if valid
#         if check_csi(frames["csi"]):
#             csi = frames["csi"]
#             # Processing CSI matrix (removing pilot, null, and outliers)
#             # csi = process_csi(csi, True, True, 10)
#             # Padding CSI matrix so dimensions stay the same
#             csi = pad_csi(csi, nframes)
#             # Adding the csi to the combines csi matrix
#             combined_csi[i] = csi
#     return combined_csi


def get_summaries(fps):
    """
    Returns summaries of all the csi files in the given fps list
    """
    # Getting the # of subcarriers in the csi data by
    # reading in files and getting bandwidth of first valid capture file
    nsubs = 0
    for fp in fps:
        fc = read_bytefile(fp)
        bandwidth = get_bandwidth(fc)
        if bandwidth != 0:
            nsubs = get_nsubs(bandwidth)
            break
    # Initialising variables to store summary data
    subs = make_subc_matrix(1, nsubs)[0]
    nframes = np.zeros([len(fps)])
    amp_mean = np.zeros([len(fps), len(subs)])
    amp_std = np.zeros([len(fps), len(subs)])
    amp_max = np.zeros([len(fps), len(subs)])
    amp_min = np.zeros([len(fps), len(subs)])
    # Looping through all the CSI pcap files
    for i, fp in enumerate(fps):
        # print(get_name(fp))
        # Reading csi info (including metadata)
        frames = read_csi(fp)
        # Checking that the CSI is valid before continuing
        if check_csi(frames["csi"]):
            # Processing csi info
            csi = process_csi(frames["csi"], True, True, 10)
            csi_amp = np.abs(csi)
            # Adding info to summary arrays
            nframes[i] = csi.shape[0]
            amp_mean[i] = csi_amp.mean(axis=0)
            amp_std[i] = csi_amp.std(axis=0)
            amp_max[i] = csi_amp.max(axis=0)
            amp_min[i] = csi_amp.min(axis=0)
    return {
        "nfiles": len(fps),
        "nframes": nframes,
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "amp_max": amp_max,
        "amp_min": amp_min,
    }


"""
**************************************************************************************************
            VISUALISING CSI DATA
**************************************************************************************************
"""


def plot_all(csi_matrix):
    # Calculating different CSI characteristics
    csi_amp = np.abs(csi_matrix)
    csi_phase = np.angle(csi_matrix) / (2 * np.pi)
    # Initialising subplots
    fig, axes = init_subplots(nrows=3, ncols=2)
    # Making plots
    # AMPLITUDE PLOTS
    heatmap_subc(
        csi_amp,
        fig,
        axes[0, 0],
        cmap="viridis",
    )
    hist2d_subc(
        csi_amp[:, 1:],
        fig,
        axes[0, 1],
        cmap="viridis",
    )
    # PHASE PLOTS
    heatmap_subc(
        csi_phase,
        fig,
        axes[1, 0],
        cmap="twilight",
    )
    hist2d_subc(
        csi_phase,
        fig,
        axes[1, 1],
        cmap="twilight",
        vmax=csi_matrix.shape[0] / 50,
    )
    # RAW COMPLEX VAL PLOTS
    hist2d_complex(
        csi_matrix,
        fig,
        axes[2, 0],
        cmap="viridis",
    )
    return fig, axes


def heatmap_subc(csi_matrix, fig, ax, **kwargs):
    # Updating keyword args for the plot
    my_kwargs = get_my_kwargs(csi_matrix, **kwargs)
    # Updating the vmax: max value
    if my_kwargs["vmax"] is None:
        my_kwargs["vmax"] = np.max(csi_matrix)
    # Making heatmap
    im = ax.pcolormesh(
        make_subc_matrix(*csi_matrix.shape)[0],
        np.arange(csi_matrix.shape[0]),
        csi_matrix,
        cmap=my_kwargs["cmap"],
        norm=Normalize(
            vmin=my_kwargs["vmin"],
            vmax=my_kwargs["vmax"],
        ),
    )
    # Adding colorbar
    fig.colorbar(im, ax=ax)
    return fig, ax


def hist2d_subc(csi_matrix, fig, ax, **kwargs):
    # Updating keyword args for the plot
    my_kwargs = get_my_kwargs(csi_matrix, **kwargs)
    # Calculating the frequencies of values for each subcarrier
    H, xedges, yedges = np.histogram2d(
        make_subc_matrix(*csi_matrix.shape).flatten(),
        csi_matrix.flatten(),
        bins=(
            csi_matrix.shape[1],
            int(csi_matrix.shape[0] / 50),
        ),
    )
    # Updating the vmax with max freq val
    if my_kwargs["vmax"] is None:
        my_kwargs["vmax"] = np.max(H[:, 1:])
    # Making frequency heatmap
    im = ax.pcolormesh(
        xedges,
        yedges,
        H.T,
        cmap=my_kwargs["cmap"],
        norm=Normalize(
            vmin=my_kwargs["vmin"],
            vmax=my_kwargs["vmax"],
        ),
    )
    # Adding colorbar
    fig.colorbar(im, ax=ax)
    return fig, ax


def hist2d_complex(csi_matrix, fig, ax, **kwargs):
    # Updating keyword args for the plot
    my_kwargs = get_my_kwargs(csi_matrix, **kwargs)
    # Calculating the frequencies of values for the complex numbers
    H, xedges, yedges = np.histogram2d(
        np.real(csi_matrix).flatten(),
        np.imag(csi_matrix).flatten(),
        bins=(
            int(np.std(np.real(csi_matrix)) / 4),
            int(np.std(np.imag(csi_matrix)) / 4),
        ),
    )
    # Updating the vmax with (# frames / 50)
    if my_kwargs["vmax"] is None:
        # my_kwargs["vmax"] = np.mean(H) + 1 * np.std(H)
        my_kwargs["vmax"] = csi_matrix.shape[0] / 50
    # Making frequency heatmap
    im = ax.pcolormesh(
        xedges,
        yedges,
        H.T,
        cmap=my_kwargs["cmap"],
        norm=Normalize(
            vmin=my_kwargs["vmin"],
            vmax=my_kwargs["vmax"],
        ),
    )
    # Adding colorbar
    fig.colorbar(im, ax=ax)
    return fig, ax


def init_subplots(nrows, ncols):
    # Initialising fig and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    # Removing all axes ticks
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )
    return fig, axes


def get_my_kwargs(csi_matrix, **kwargs):
    # Initialising default kwargs
    my_kwargs = {
        "alpha": None,
        "cmap": None,
        "vmin": None,
        "vmax": None,
        "bins": (100, 100),
    }
    # Updating keyword args with given values
    for k, v in kwargs.items():
        my_kwargs[k] = v
    # Returning the my_kwargs dict
    return my_kwargs


# %%
