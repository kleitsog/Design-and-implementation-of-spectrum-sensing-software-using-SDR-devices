import adi # type: ignore
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import time
from create_table import create_raw_table
from connection_manager import db_connection

# This class initializes our command line arguments and checks if they are proper
class Arguments:
    def __init__(self, f, b, c, i, t, s):   # Constructor
        self.frequency, self.bandwidth = self.frequency_bandwidth_check(f, b)
        self.channel = self.channel_check(c)
        self.iterations = self.iterations_check(i)
        self.time = self.time_check(t)
        self.fftsize = self.fftsize_check(s)
        self.num_samples = 35000
        self.final_iterations = 0   # Shows us the real iterations because we update its value during the execution


    # We check if frequency and bandwidth are proper
    def frequency_bandwidth_check(self, f, b):
        try:
            freq = float(f)
        except:
            print("Illegal value of frequency. Program terminates!!!")
            sys.exit()
        try:
            bw = float(b)
        except:
            print("Illegal value of bandwidth. Program terminates!!!'")
        
        if freq < 70.0:     # Pluto restriction-frequency is between 70 MHz and 6 GHz
            freq = 70.0
        if bw < 0.521:     # Pluto restriction-lowest sample rate is 521 kHz
            bw = 0.53                       
        if freq + bw > 5995.0:
            print("Sum of frequency and bandwidth must be lower than 5995 MHz. Program terminates!!!")
            sys.exit()
        return freq, bw
    
    
    # We check the given channel bandwidth if it is legal or not and we compute it if the user did not give us a value for channel bandwidth
    def channel_check(self, channel):
        if channel == None:     # If user does not provide channel bandwidth
            if self.bandwidth <= 10.0:
                channel = self.bandwidth
                return channel
            
            # If bandwidth > 10 MHz we divide the bandwidth with 2, 3, 4... until we have a channel <= 10 MHz
            div = 2.0
            channel = self.bandwidth / div
            while channel > 10.0:
                div += 1.0
                channel = self.bandwidth / div
            return int(channel)   
        # User provided us channel bandwidth
        try:
            channel = abs(float(channel))   # If user provides us with negative number
        except:
            print("Illegal value of channel bandwidth. Program terminates!!!")
            sys.exit()
        # Pluto restriction-max sample rate for this project is 10 MHz. For bigger sample rates we lose samples because of USB 2.0
        if channel > 10.0:   
            print("This program supports channel bandwidth from 0.53 MHz to 10 MHz. Program terminates!!!")
            sys.exit()
        elif channel < 0.53:   
            print("This program supports channel bandwidth from 0.53 MHz to 10 MHz. Program terminates!!!")
            sys.exit()
        elif channel > self.bandwidth:   
            print("Channel bandwidth cannot be greater than bandwidth. Program terminates!!!")
            sys.exit()
        return channel
        
    
    # We check if the number of iterations is proper
    def iterations_check(self, i):  
        if i == None:    # If user did not provide number of iterations
            return -1    # i =-1 means that user does want specific number of iterations 
        try:
            iterr = abs(int(i))
        except:
            print("Illegal value of iterations. Program terminates!!!")
            sys.exit()
        if iterr > 300:     # Max iterations = 300
            return 300
        elif iterr == 0:    # Min iterations = 1
            return 1
        else:
            return iterr


    # We check if the time in seconds is proper
    def time_check(self, t):
        if t == None:
            return -1    # time = -1 means there is no time restriction
        try:
            ttime = abs(int(t))
        except:
            print("Illegal value of seconds. Program terminates!!!")
            sys.exit()
        if ttime > 60:
            return 60    # Max scan time = 60 seconds
        elif ttime == 0:
            return -1
        else:
            return ttime
    

    # We check if the FFTsize is proper (is power of 2 , between 2^5 and 2^15)
    def fftsize_check(self, s):
        if s == None:
            return 8192     # Default FFTsize value
        try:
            fftsize = abs(int(s))
        except:
            print("Illegal value of FFTsize. Program terminates!!!")
            sys.exit()
        # We check if FFTsize is power of 2 
        fftsize_temp = 32
        while fftsize_temp < fftsize:
            fftsize_temp *= 2
        if fftsize_temp > 32768: 
            return 32768   # Max FFTsize is 32768
        else:
            return fftsize_temp


# Psd function computes the FFT of our received samples and the Power Spectral Density
def psd(rx_samples, args):
    rx_samples = rx_samples[0:args.fftsize]     # We check only FFTsize samples out of the received samples
    rx_samples = rx_samples * np.hamming(len(rx_samples))   # Apply a Hamming window 
    ffted = (np.fft.fft(rx_samples))    # FFT
    power = (abs(ffted))**2
    ps_density = power / ((args.channel * (10**6)) * args.fftsize)
    psd_dB = 10.0 * np.log10(ps_density)    # dBs
    psd_dB_shifted = np.fft.fftshift(psd_dB)    # FFTshift
    return(psd_dB_shifted)


# Saves specific data channel samples in database
def insert_in_database(samples_dbs, iteration_no):
    with db_connection() as conn:
        curr = conn.cursor()
        curr.execute("INSERT INTO samples_database.numpy_raw_data(scan_number, channel_data) VALUES (%s, %s)",
                        (iteration_no, pickle.dumps(samples_dbs))  # For serialization
                    )
        conn.commit()


# Scan_bandwidth sets up Pluto SDR for reception and gets the samples in every center_frequency
def scan_bandwidth(sdr, args):
    iteration_samples = []
    num_samps = args.num_samples    
    # Config Rx
    sdr.sample_rate = int(sample_rate)
    sdr.gain_control_mode_chan0 = "manual"
    gain = 50.0
    sdr.rx_hardwaregain_chan0 = gain    # Set receive gain 
    sdr.rx_rf_bandwidth = int(sample_rate)   # Set the bandwidth of the reception filter (Hz)
    sdr.rx_buffer_size = num_samps   # Each call of rx() will return num_samps number of samples
    iteration_sense = 0    # Sense time for one iteration
    for freq in center_freqs:
        sdr.rx_lo = int(freq)   # The SDR will scan at this center_frequency
        channel_scan_time_start = time.time()   # Start time of channel sense
        sdr.rx_destroy_buffer()
        rx_samples = sdr.rx()
        channel_scan_time_finish = time.time()   # Finish time of channel sense
        channel_time_sense = channel_scan_time_finish - channel_scan_time_start   # Time of channel sense
        iteration_sense += channel_time_sense
        iteration_samples.append(rx_samples)
    return [iteration_samples, channel_time_sense, iteration_sense]


# For every scan iteration takes samples from all bandwidth and saves them in a list.
# Called when user does not provide iterations and time. Default scan iterations = 4
def no_restricted_scan(sdr, args):
    all_samples = []
    scan_sense = 0   # Scan sense time
    for i in range(1, args.iterations + 1): 
        scan_b = scan_bandwidth(sdr, args)
        all_samples.append(scan_b[0])
        channel_time_sense = scan_b[1]   # Channel sense time
        iteration_sense = scan_b[2]     # Iteration sense time
        scan_sense += iteration_sense    
    args.final_iterations = args.iterations   # We update the real iterations counter
    return [all_samples, channel_time_sense, iteration_sense, scan_sense]


# For every scan iteration takes samples from all bandwidth and saves them in a list.
# Called when user wants specific number of scan iterations
def iteration_restricted_scan(sdr, args):
    all_samples = []
    scan_sense = 0   # Scan sense time
    for i in range(1, args.iterations + 1):
        scan_b = scan_bandwidth(sdr, args)
        all_samples.append(scan_b[0])
        channel_time_sense = scan_b[1]   # Channel sense time
        iteration_sense = scan_b[2]  # Iteration sense time
        scan_sense += iteration_sense
    args.final_iterations = args.iterations   # We update the real iterations counter
    return [all_samples, channel_time_sense, iteration_sense, scan_sense]    


# Scans bandwidth with time restriction
def time_restricted_scan(sdr, args):
    scan_sense = 0   # Scan sense time
    all_samples = []
    scan_b = scan_bandwidth(sdr, args)
    channel_time_sense = scan_b[1]   # Channel sense time
    iteration_sense = scan_b[2]     # Iteration sense time
    scan_sense += iteration_sense
    all_samples.append(scan_b[0])   # Scan once in order to measure time for one iteration
    user_time = args.time - iteration_sense   # Time left after one iteration
    count = 2   # Counts number of iterations
    while(user_time > iteration_sense):   # If we have time we continue scanning
        scan_b = scan_bandwidth(sdr, args)
        if scan_b[2] > user_time:
            break
        channel_time_sense = scan_b[1]
        iteration_sense = scan_b[2]
        scan_sense += iteration_sense
        all_samples.append(scan_b[0])   # Scan once in order to measure time for one iteration
        user_time -= iteration_sense    # Time left after this iteration
        count += 1
    args.final_iterations = count - 1   # We update the real iterations counter
    return [all_samples, channel_time_sense, iteration_sense, scan_sense]


# Scans bandwidth with time and iterations restriction
def iterations_time_restricted_scan(sdr, args):
    scan_sense = 0   # Scan sense time
    scan_b = scan_bandwidth(sdr, args)
    channel_time_sense = scan_b[1]   # Channel sense time
    iteration_sense = scan_b[2]     # Iteration sense time
    scan_sense += iteration_sense
    all_samples.append(scan_b[0])   # Scan once in order to measure time for one iteration
    user_time = args.time - iteration_sense   # Time left after one iteration
    args.final_iterations += 1   # We update the real iterations counter
    for i in range(2, args.iterations + 1):
        scan_b = scan_bandwidth(sdr, args)
        if scan_b[2] > user_time:
            args.final_iterations = i - 1
            break
        channel_time_sense = scan_b[1]
        iteration_sense = scan_b[2]
        scan_sense += iteration_sense
        all_samples.append(scan_b[0])   # Scan once in order to measure time for one iteration
        user_time -= iteration_sense    # Time left after this iteration
        args.final_iterations += 1   # We update the real iterations counter
    return [all_samples, channel_time_sense, iteration_sense, scan_sense]
      

# Returns channel data of a specific channel and a specific iteration
def get_channel_data_from_database(channel_id):
    with db_connection() as conn:
        curr = conn.cursor()
        curr.execute(
            """
            SELECT channel_data
            FROM samples_database.numpy_raw_data
            WHERE id=%s;
            """, 
            (channel_id, )
        )
        return np.array(pickle.loads(curr.fetchone()[0]))   # Diserialization



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help = 'lowest frequency')   # User provides low limit of frequency in MHz
    parser.add_argument('-b', help = 'bandwidth')   # User provides the bandwidth he wants to scan in MHz
    parser.add_argument('-c', help = 'channel')   # User provides channel bandwidth in MHz
    parser.add_argument('-i', help = 'iterations')   # User provides number of iterations of the scan
    parser.add_argument('-t', help = 'time')   # User provides time he wants to scan in seconds
    parser.add_argument('-s', help = 'FFT size')  # User provides specific FFTsize
    args = parser.parse_args()
    arguments = Arguments(args.f, args.b, args.c, args.i, args.t, args.s )
    
    sdr = adi.Pluto("ip:192.168.2.1")
    sample_rate = arguments.channel * (10**6)   # Hz
    if arguments.channel == arguments.bandwidth:   # If we have just one channel to scan
        f_start = arguments.frequency * (10**6) + (sample_rate / 2)  # First center frequency
    else:   # If we have more than one channel
        f_start = arguments.frequency * (10**6)   # First center frequency
    f_end = arguments.frequency * (10**6) + arguments.bandwidth * (10**6)   # User wants to scan until f_end frequency

    overlap = 0.05 * sample_rate    # 5% overlap between channels
    step_size = sample_rate - overlap   # New center frequency step
    bin = sample_rate / arguments.fftsize   # Bin size
    overlap_bins = int(overlap / bin)   # Number of bins that are overlapped
    center_freqs = np.arange(f_start, f_end, step_size)   # List with all the center frequencies of our channels
    coverage_start = center_freqs[0] - sample_rate / 2   # The real starting frequency of our scan
    coverage_end = center_freqs[-1] + sample_rate / 2   # Coverage end is the ending frequency of our scan
    # Case that we have some more bandwidth to scan, so we need one more channel due to the overlap
    if coverage_end < f_end :  
        extra_center_freq = center_freqs[-1] + step_size
        center_freqs = np.append(center_freqs, extra_center_freq) 
        coverage_end = center_freqs[-1] + sample_rate / 2   # In that case we update coverage_end
   
    create_raw_table()   # We then create the database array
    
    all_samples = []    # All samples from our scan
    # Four different operations 
    if arguments.iterations != -1 and arguments.time != -1:  # User provides us with iterations number and time
        scan = iterations_time_restricted_scan(sdr, arguments)
        all_samples = scan[0].copy()
        channel_sense_time = scan[1]    # Time to sense one channel
        iteration_sense_time = scan[2]    # Time to sense one iteration
        scan_sense_time = scan[3]   # Time to sense all iterations of the scan
    elif arguments.iterations != -1 :   # User provides us with iterations number
        scan = iteration_restricted_scan(sdr, arguments)
        all_samples = scan[0].copy()
        channel_sense_time = scan[1]    # Time to sense one channel
        iteration_sense_time = scan[2]    # Time to sense one iteration
        scan_sense_time = scan[3]   # Time to sense all iterations of the scan
    elif arguments.time != -1 :   # User provides us with time
        scan = time_restricted_scan(sdr, arguments)
        all_samples = scan[0].copy()
        channel_sense_time = scan[1]    # Time to sense one channel
        iteration_sense_time = scan[2]    # Time to sense one iteration
        scan_sense_time = scan[3]   # Time to sense all iterations of the scan
    else :   # User does not provide us with iterations number and time, our scan will consist of 4 iterations
        arguments.iterations = 4    # Default value of iterations
        scan = no_restricted_scan(sdr, arguments)
        all_samples = scan[0].copy()
        channel_sense_time = scan[1]    # Time to sense one channel
        iteration_sense_time = scan[2]    # Time to sense one iteration
        scan_sense_time = scan[3]   # Time to sense all iterations of the scan
    

    # Here we do signal processing of our samples and transfer data to database
    scan_psd_time = 0
    scan_d_time = 0
    for i in range(0, len(all_samples)):
        iteration_psd_time = 0  # Time to process iteration data 
        iteration_d_time = 0   # Time to transfer iteration data to database
        for j in range(0, len(center_freqs)):
            start_psd_channel = time.time()  
            samples_dbs = psd(all_samples[i][j], arguments)
            finish_psd_channel = time.time()
            time_psd_channel = finish_psd_channel - start_psd_channel   # Channel processing time
            iteration_psd_time += time_psd_channel
            start_d_channel = time.time()
            insert_in_database(samples_dbs, i+1)
            finish_d_channel = time.time()
            time_d_channel = finish_d_channel - start_d_channel   # Time to transfer channel data to database
            iteration_d_time += time_d_channel
            samples_dbs = []
        scan_psd_time += iteration_psd_time     # Time to process data from all iterations of the scan
        scan_d_time += iteration_d_time     # Time to transfer all data of the scan to database
        
    channel_total_time = channel_sense_time + time_psd_channel + time_d_channel     # Total time to scan a channel
    iteration_total_time = iteration_sense_time + iteration_psd_time + iteration_d_time     # Time to scan an iteration
    scan_total_time = scan_sense_time + scan_psd_time + scan_d_time     # Total scan time
    
    # print(f"After {arguments.final_iterations} iterations the total scan time is : {scan_total_time:.4f} seconds.")
    # print(f"Time to sense one single channel is : {channel_sense_time:.4f} seconds.")  
    # print(f"Time for processing one single channel is : {time_psd_channel:.4f} seconds.")
    # print(f"TIme to transfer to database data from one single channel is : {time_d_channel:.4f} seconds.")
    
    # print(f"Time to sense one iteration is : {iteration_sense_time:.4f} seconds.")
    # print(f"Time for processing one iteration is : {iteration_psd_time:.4f} seconds.")
    # print(f"Time for tranfering iteration data to database is : {iteration_d_time:.4f} seconds.")
    
    # print(f"Time for sensing all iterations of the scan is : {scan_sense_time:.4f} seconds.")
    # print(f"Time for processing of all iterations is : {scan_psd_time:.4f} seconds.")
    # print(f"Time for database transfer of all iteration data is : {scan_d_time:.4f} seconds.")

    # print(f"Time to scan one single channel(sense + processing + database) is : {channel_total_time:.4f} seconds.")   
    # print(f"Time to scan one iteration(sense + processing + database) is : {iteration_total_time:.4f} seconds.")   
    # print(f"Time to scan all iterations(sense + processing + database) is : {scan_total_time:.4f} seconds.")   


    # Scan has finished and dBs have been stored to database. Now we retrieve data from the database and show plots.     
    db_list = []   # Contains numpy arrays with average values of dBs for every channel
    for channel in range(1, len(center_freqs) + 1):   # For every channel create average numpy array in dBs
        dbs_temp = np.zeros(arguments.fftsize, dtype = float )   
        channel_iter = channel
        for iter in range(1, arguments.final_iterations + 1):   # For every iteration of a specific channel retrieve the dBs
            dbs_temp += get_channel_data_from_database(channel_iter)
            channel_iter += len(center_freqs)    
        db_list.append(dbs_temp / arguments.final_iterations)  

     # We concatenate all channels together in final_dBs for the plot
    if len(center_freqs) == 1:  # If we have only one channel
        final_dBs = db_list[0]
        # Frequencies for the plot
        frequencies = np.linspace((coverage_start/(10**6)), coverage_end/(10**6), num=(arguments.fftsize * len(center_freqs)))  
    else:   # If we have more than one channels we have to fix the overlap issue
        # We use start and end to specify the bins that need to be averaged together from neighbour channels
        start = arguments.fftsize - overlap_bins   
        end = overlap_bins
        sum = []
        avg = []
        final_dBs = db_list[0][0:start]
        for i in range(1, len(db_list), 1):
            sum = db_list[i-1][start:] + db_list[i][0:end]
            avg = sum / 2
            if i > 1:
                final_dBs = np.concatenate((final_dBs, db_list[i-1][end:start]))
            final_dBs = np.concatenate((final_dBs, avg))
            sum = []
            avg = []
        final_dBs = np.concatenate((final_dBs,db_list[i][end:]))
        # Frequencies for the plot
        num_freq = (arguments.fftsize + ((len(center_freqs)-1) * (arguments.fftsize - overlap_bins)))
        frequencies = np.linspace((coverage_start/(10**6)), coverage_end/(10**6), num= num_freq) 
    

    # Plotting the spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, final_dBs) 
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()