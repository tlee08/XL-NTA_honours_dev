# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# OFDM modulation example
# WORKS!!!!

# Define OFDM parameters
# NOTE: Ensure that all subcarriers are a multipe of the symbol duration (so every frequency's period ends on the symbol's duration) to avoid FFT smearing.
# NOTE: ENsure the total number of samples is at least twice the max frequency (Nyquist rate).
n_subs = 640  # Number of subcarriers
c_subs = 500  # Centre of the channel frequency
s_subs = 1  # Subcarrier spacing in Hz
symbol_duration = 1  # Symbol duration in seconds
sampling_frequency = 10000  # Sampling frequency
N = int(sampling_frequency * symbol_duration)

# Possible QAM files (16-QAM)
n_qam = 2
vals = np.linspace(-1, 1, np.power(2, n_qam))
qam_vals = np.reshape(np.meshgrid(vals, vals), [2, -1]).transpose()
qam_vals = qam_vals[:, 0] + qam_vals[:, 1] * 1j

# Create the time vector for one OFDM symbol
t = np.linspace(0, symbol_duration, N, endpoint=False)

# Create the frequency vector for one OFDM symbol
# The frequencies MUST be multiples of the chosen period, so the period always ends at the same place (in time).
frequencies = np.linspace(
    c_subs - s_subs * n_subs / 2 + s_subs,
    c_subs + s_subs * n_subs / 2,
    n_subs,
)

# Generate some example data to modulate onto the subcarriers (complex symbols)
# data_symbol = np.concatenate(
#     [
#         np.repeat(0, 6),
#         np.repeat(1 + 1j, 26),
#         np.repeat(0, 1),
#         np.repeat(1 - 1j, 25),
#         np.repeat(0, 6),
#     ]
# )
data_symbol = qam_vals[np.random.randint(0, qam_vals.shape[0], n_subs)]
data_symbol[:7] = 0
data_symbol[int(n_subs / 2) - 1] = 0
data_symbol[-7:] = 0

# Adding a) gaussian and b) shift noise to the signal (TODO need to implement phase shift noise)
v_g = 0.05
v_p = 0.01
noise_g = v_g * (np.random.randn(n_subs) + 1j * np.random.randn(n_subs)) / np.sqrt(2)
noise_p = np.exp(2j * np.pi * v_p * np.random.randn(n_subs))
data_symbol_noisy = (data_symbol + noise_g) * noise_p

# We can visualise how much noise we've added with the plots below
# GAUSSIAN WHITE NOISE
# sns.jointplot(x=np.real(noise_g), y=np.imag(noise_g), edgecolor=None, alpha=0.1)
# plt.show()
# PHASE SHIFT NOISE
# sns.scatterplot(x=np.real(noise_p), y=np.imag(noise_p), edgecolor=None, alpha=0.1)
# plt.ylim(-1.5, 1.5)
# plt.xlim(-1.5, 1.5)
# plt.show()

# Create the OFDM symbol by modulating data onto subcarriers
data_signal = data_symbol_noisy * np.exp(
    1j * 2 * np.pi * frequencies * np.transpose([t])
)
# Summing the signals of different frequency together to form a single waveform (like IFFT)
signal = np.nansum(data_signal, axis=1)
# Taking real component of signal
signal = np.real(signal)

# DEMODULATING SIGNAL
demod_freq = np.fft.rfftfreq(N, 1 / sampling_frequency)
demod_vals = np.fft.rfft(signal) * 2 / N

# Filtering signal with simple "low-pass"?? filter
for i, f in enumerate(demod_freq):
    if np.abs(demod_vals[i]) < 0.1:
        demod_vals[i] = 0
# Filtering frequencies for only those in the chosen frequency band
my_filter = np.logical_and(
    demod_freq >= np.min(frequencies), demod_freq <= np.max(frequencies)
)
demod_freq = demod_freq[my_filter]
demod_vals = demod_vals[my_filter]

# Plot the real part of the OFDM symbol in the time domain
plt.figure(figsize=(12, 4))
plt.plot(t, signal)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("OFDM Symbol through time")
plt.grid(True)
plt.show()
np.random.randn

# Plot the OFDM signal in the frequency domain (Magnitude)
plt.figure(figsize=(12, 4))
plt.plot(
    frequencies,
    np.abs(data_symbol),
    label="Transmitted",
    alpha=0.5,
)
plt.plot(
    demod_freq,
    np.abs(demod_vals),
    label="Received",
    alpha=0.5,
)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("OFDM symbol frequency (magnitude)")
plt.grid(True)
plt.legend()
plt.show()

# Plot the OFDM signal in the frequency domain (Phase)
plt.figure(figsize=(12, 4))
plt.plot(
    frequencies,
    np.angle(data_symbol) / (2 * np.pi),
    label="Transmitted",
    alpha=0.5,
)
plt.plot(
    demod_freq,
    np.angle(demod_vals) / (2 * np.pi),
    label="Received",
    alpha=0.5,
)
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.title("OFDM symbol frequency (phase)")
plt.grid(True)
plt.legend()
plt.show()

# Plot the demodulated OFDM symbol constellation diagram
plt.figure(figsize=(4, 4))
sns.scatterplot(
    x=np.real(qam_vals),
    y=np.imag(qam_vals),
    marker="+",
    s=200,
    alpha=0.5,
    edgecolor=None,
)
# Getting intended symbols of demodulated (to check error rate)
x = np.array([])
for i, v in enumerate(data_symbol):
    x = np.append(x, v)
    if i != len(data_symbol) - 1:
        x = np.append(x, np.zeros(int(s_subs / (sampling_frequency / N) - 1)))
sns.scatterplot(
    x=np.real(demod_vals),
    y=np.imag(demod_vals),
    hue=x.astype(str),
    marker=".",
    edgecolor=None,
    palette="rainbow",
    legend=None,
    alpha=0.4,
)
plt.show()

# %%
