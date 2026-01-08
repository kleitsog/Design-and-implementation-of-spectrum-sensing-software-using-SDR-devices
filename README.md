# Design-and-implementation-of-spectrum-sensing-software-using-SDR-devices

# About the Project

This project implements a spectrum analyzer using an ADALM-Pluto SDR and the pyadi-iio Python library.
Pluto SDR performs a frequency sweep across the target bandwidth to acquire raw IQ samples. These samples are processed using a Fast Fourier Transform (FFT) to generate a Power Spectral Density (PSD) plot, visualizing the signal's power distribution.

The user must define the frequency and the bandwidth that he/she wants to scan in MHz. Additionally the user can select the channel bandwidth, the total time of the scan, the iterations of the scan, the FFTsize. Input is passed via the command line.

| Flag | Parameter | Unit | Description |
| --- | --- | --- | --- | 
| `-f` | Frequency | MHz (Required) | Start frequency.
| `-b` | Bandwidth | MHz (Required) | Total span of the frequency range to capture.
| `-c` | Channel Bandwidth | MHz | The width of individual channels within the scan.
| `-t` | Time | sec | Duration of the scan.
| `-i` | Iterations | - | Number of iterations of the scan.
| `-s` | FFTsize | - | Resolution of the spectrum analysis (e.g., 1024, 2048).

According to the inputs, the program features four operational modes:
 + Time and iteration restricted scan
 + Time restricted scan
 + Iteration restricted scan
 + Classic scan

## Input Examples

+ python3 pluto_spectrum_analyzer.py -f 2400 -b 30 -c 5
+ python3 pluto_spectrum_analyzer.py -f 2400 -b 30 -t 10 -s 2048
+ python3 pluto_spectrum_analyzer.py -f 2400 -b 30 -c 5 
+ python3 pluto_spectrum_analyzer.py -f 2400 -b 30 -c 5 -i 15 -t 40 -s 1024

## Restrictions

+ Frequency is between 70 MHz and 6 GHz(5995MHz).
+ Bandwidth is greater than 521KHz.
+ Channel bandwidth is between 0.53MHz and 10MHz and not greater than the bandwidth.
+ Iterations are between 1 and 300.
+ Time is between 1 and 60 seconds.
+ FFT is between 32 and 32768. (The FFT size must be a power of two for optimal performance so we accept values from 2^5 to 2^15).
