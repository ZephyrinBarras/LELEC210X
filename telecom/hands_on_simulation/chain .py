from typing import Optional


import numpy as np

BIT_RATE = 50e3
PREAMBLE = np.array([int(bit) for bit in f"{0xAAAAAAAA:0>32b}"])
SYNC_WORD = np.array([int(bit) for bit in f"{0x3E2A54B7:0>32b}"])


class Chain:
    name = ""

    ## Communication parameters
    bit_rate = BIT_RATE
    freq_dev = BIT_RATE / 4

    osr_tx = 64
    osr_rx = 8

    preamble = PREAMBLE
    sync_word = SYNC_WORD

    payload_len = 800  # Number of bits per packet

    ## Simulation parameters
    n_packets = 20  # Number of sent packets

    ## Channel parameters
    sto_val = 0
    sto_range = 10 / BIT_RATE  # defines the delay range when random

    cfo_val = 0
    cfo_range = 12500  # defines the CFO range when random (in Hz) #(1000 in old repo)

    snr_range = np.arange(-10, 25)

    ## Lowpass filter parameters
    numtaps = 100
    cutoff = BIT_RATE * osr_rx / 2.0001  # or 2*BIT_RATE,...

    ## Tx methods

    def modulate(self, bits: np.array) -> np.array:
        """
        Modulates a stream of bits of size N
        with a given TX oversampling factor R (osr_tx).

        Uses Continuous-Phase FSK modulation.

        :param bits: The bit stream, (N,).
        :return: The modulates bit sequence, (N * R,).
        """
        fd = self.freq_dev  # Frequency deviation, Delta_f
        B = self.bit_rate  # B=1/T
        h = 2 * fd / B  # Modulation index
        R = self.osr_tx  # Oversampling factor

        x = np.zeros(len(bits) * R, dtype=np.complex64)
        ph = 2 * np.pi * fd * (np.arange(R) / R) / B  # Phase of reference waveform

        phase_shifts = np.zeros(
            len(bits) + 1
        )  # To store all phase shifts between symbols
        phase_shifts[0] = 0  # Initial phase

        for i, b in enumerate(bits):
            x[i * R : (i + 1) * R] = np.exp(1j * phase_shifts[i]) * np.exp(
                1j * (1 if b else -1) * ph
            )  # Sent waveforms, with starting phase coming from previous symbol
            phase_shifts[i + 1] = phase_shifts[i] + h * np.pi * (
                1 if b else -1
            )  # Update phase to start with for next symbol

        return x

    ## Rx methods
    bypass_preamble_detect = False

    def preamble_detect(self, y: np.array) -> Optional[int]:
        """
        Detects the preamlbe in a given received signal.

        :param y: The received signal, (N * R,).
        :return: The index where the preamble starts,
            or None if not found.
        """
        raise NotImplementedError

    bypass_cfo_estimation = False

    def cfo_estimation(self, y: np.array) -> float:
        """
        Estimates the CFO based on the received signal.

        :param y: The received signal, (N * R,).
        :return: The estimated CFO.
        """
        raise NotImplementedError

    bypass_sto_estimation = False

    def sto_estimation(self, y: np.array) -> float:
        """
        Estimates the STO based on the received signal.

        :param y: The received signal, (N * R,).
        :return: The estimated STO.
        """
        raise NotImplementedError

    def demodulate(self, y: np.array) -> np.array:
        """
        Demodulates the received signal.

        :param y: The received signal, (N * R,).
        :return: The signal, after demodulation.
        """
        raise NotImplementedError


class BasicChain(Chain):
    name = "Basic Tx/Rx chain"

    cfo_val, sto_val = np.nan, np.nan  # CFO and STO are random

    bypass_preamble_detect = False

    def preamble_detect(self, y):
        """
        Detect a preamble computing the received energy (average on a window).
        """
        L = 4 * self.osr_rx
        y_abs = np.abs(y)

        for i in range(0, int(len(y) / L)):
            sum_abs = np.sum(y_abs[i * L : (i + 1) * L])
            if sum_abs > (L - 1):  # fix threshold
                return i * L

        return None

    bypass_cfo_estimation = False

    def cfo_estimation(self, y):
        """
        Estimates CFO using Moose algorithm, on first samples of preamble.
        """
        """N = 2
        Nt = N*R
        sum1 = 0
        T = 1/B
        for k in range (0 , Nt) :
            sum1 += y [k + Nt] * y [k].conjugate()
        cfo_est = np.phase ( sum1 ) / ( 2 * np . pi * Nt * T / R )
        return cfo_est"""
        N = 2
        N_t= self.osr_rx*N
        sum1=0
        T=1/self.bit_rate
        for k in range (0 , N_t) :
            sum1 += y [k + N_t] * y [k].conjugate()
        cfo_est = np.angle ( sum1 ) / ( 2 * np . pi * N_t * T / self.osr_rx )
        return cfo_est
        # TO DO: extract 2 blocks of size N*R at the start of y
        #print(y)
        """a = y[0:N*self.osr_rx]
        a_star = np.conj(a)
        b= y[N*self.osr_rx:2*N*self.osr_rx]
        c = b*a_star
        sum_= np.sum(c)
        phase = np.angle(sum_)"""


        # TO DO: apply the Moose algorithm on these two blocks to estimate the CFO

        """cfo_est = phase/(2*np.pi*N/(self.bit_rate))  # Default value, to change

        return cfo_est"""

    bypass_sto_estimation = False

    def sto_estimation(self, y):
        """
        Estimates symbol timing (fractional) based on phase shifts.
        """
        R = self.osr_rx

        # Computation of derivatives of phase function
        phase_function = np.unwrap(np.angle(y))
        phase_derivative_1 = phase_function[1:] - phase_function[:-1]
        phase_derivative_2 = np.abs(phase_derivative_1[1:] - phase_derivative_1[:-1])

        sum_der_saved = -np.inf
        save_i = 0
        for i in range(0, R):
            sum_der = np.sum(phase_derivative_2[i::R])  # Sum every R samples

            if sum_der > sum_der_saved:
                sum_der_saved = sum_der
                save_i = i

        return np.mod(save_i + 1, R)

    def demodulate(self, y):
        """
        Non-coherent demodulator.
        """
        
        R = self.osr_rx  # Receiver oversampling factor
        nb_syms = len(y) // R  # Number of CPFSK symbols in y       ok
        fd = self.freq_dev  # Frequency deviation, Delta_f
        B = self.bit_rate  # B=1/T
        T=1/B
        signal = np . zeros ( nb_syms , dtype = int )
        y = np.resize(y, (nb_syms, R))
        bits_hat = np.zeros(nb_syms, dtype=int)  # Default value, all bits=0. TO CHANGE!


        # TO DO: generate the reference waveforms used for the correlation
        # hint: look at what is done in modulate() in chain.py

        # TO DO: compute the correlations with the two reference waveforms (r0 and r1)
        for k in range(nb_syms):
            r0=0
            r1=0
            for b in range(R):
                r1+=y[k,b]*np.exp(-1j*2*np.pi*fd*b*(1/(B*R)))
                r0+=y[k,b]*np.exp(1j*2*np.pi*fd*b*(1/(B*R)))
            if(abs(r1)<abs(r0)):
                bits_hat[k]=0
            elif(abs(r1)>abs(r0)):
                bits_hat[k]=1
                
        # TO DO: performs the decision based on r0 and r1
            
        
        return bits_hat
