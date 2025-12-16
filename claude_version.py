import numpy as np
from enum import Enum
from typing import Union, Tuple


class FilterType(Enum):
    """Enumeration of available biquad filter types."""
    LPF = "lowpass"
    HPF = "highpass"
    BPF_SKIRT = "bandpass_skirt"  # constant skirt gain, peak gain = Q
    BPF_PEAK = "bandpass_peak"    # constant 0 dB peak gain
    NOTCH = "notch"
    APF = "allpass"
    PEAKING_EQ = "peaking"
    LOW_SHELF = "lowshelf"
    HIGH_SHELF = "highshelf"


class BiquadFilter:
    """
    Implementation of Robert Bristow-Johnson's Audio EQ Cookbook biquad filters.
    
    All filters are derived from analog prototypes using the Bilinear Transform (BLT)
    with frequency warping compensation.
    """
    
    def __init__(self, filter_type: FilterType, sample_rate: float, 
                 frequency: float, q: float = 0.707, 
                 db_gain: float = 0.0, bandwidth: float = None, 
                 shelf_slope: float = None):
        """
        Initialize a biquad filter.
        
        Parameters:
        -----------
        filter_type : FilterType
            The type of filter to create
        sample_rate : float
            Sampling frequency in Hz (Fs)
        frequency : float
            Center/corner/shelf frequency in Hz (f0)
        q : float, optional
            Quality factor (default: 0.707, Butterworth response)
        db_gain : float, optional
            Gain in dB for peaking and shelving filters (default: 0.0)
        bandwidth : float, optional
            Bandwidth in octaves (alternative to Q)
        shelf_slope : float, optional
            Shelf slope parameter for shelving filters (alternative to Q)
        """
        self.filter_type = filter_type
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.q = q
        self.db_gain = db_gain
        self.bandwidth = bandwidth
        self.shelf_slope = shelf_slope
        
        # Calculate coefficients
        self.b0, self.b1, self.b2, self.a0, self.a1, self.a2 = self._calculate_coefficients()
        
        # Normalize coefficients by a0
        self.b0 /= self.a0
        self.b1 /= self.a0
        self.b2 /= self.a0
        self.a1 /= self.a0
        self.a2 /= self.a0
        self.a0 = 1.0
        
        # Initialize state variables for Direct Form 1
        self.x1 = 0.0  # x[n-1]
        self.x2 = 0.0  # x[n-2]
        self.y1 = 0.0  # y[n-1]
        self.y2 = 0.0  # y[n-2]
    
    def _calculate_coefficients(self) -> Tuple[float, float, float, float, float, float]:
        """Calculate the six biquad coefficients based on filter parameters."""
        
        # Compute intermediate variables
        A = np.sqrt(10 ** (self.db_gain / 20))  # amplitude for shelving/peaking
        w0 = 2 * np.pi * self.frequency / self.sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        
        # Calculate alpha based on the provided parameter
        if self.bandwidth is not None:
            # alpha from bandwidth
            alpha = sin_w0 * np.sinh(np.log(2) / 2 * self.bandwidth * w0 / sin_w0)
        elif self.shelf_slope is not None:
            # alpha from shelf slope
            alpha = sin_w0 / 2 * np.sqrt((A + 1/A) * (1/self.shelf_slope - 1) + 2)
        else:
            # alpha from Q
            alpha = sin_w0 / (2 * self.q)
        
        # Calculate coefficients based on filter type
        if self.filter_type == FilterType.LPF:
            b0 = (1 - cos_w0) / 2
            b1 = 1 - cos_w0
            b2 = (1 - cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif self.filter_type == FilterType.HPF:
            b0 = (1 + cos_w0) / 2
            b1 = -(1 + cos_w0)
            b2 = (1 + cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif self.filter_type == FilterType.BPF_SKIRT:
            b0 = sin_w0 / 2  # = Q * alpha
            b1 = 0
            b2 = -sin_w0 / 2  # = -Q * alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif self.filter_type == FilterType.BPF_PEAK:
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif self.filter_type == FilterType.NOTCH:
            b0 = 1
            b1 = -2 * cos_w0
            b2 = 1
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif self.filter_type == FilterType.APF:
            b0 = 1 - alpha
            b1 = -2 * cos_w0
            b2 = 1 + alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif self.filter_type == FilterType.PEAKING_EQ:
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
            
        elif self.filter_type == FilterType.LOW_SHELF:
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
            
        elif self.filter_type == FilterType.HIGH_SHELF:
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
        
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        return b0, b1, b2, a0, a1, a2
    
    def process_sample(self, x: float) -> float:
        """
        Process a single sample through the filter using Direct Form 1.
        
        Parameters:
        -----------
        x : float
            Input sample
            
        Returns:
        --------
        float
            Filtered output sample
        """
        # Direct Form 1 implementation
        # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 \
            - self.a1 * self.y1 - self.a2 * self.y2
        
        # Update state variables
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        
        return y
    
    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Process an array of samples through the filter.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal array
            
        Returns:
        --------
        np.ndarray
            Filtered output signal
        """
        output = np.zeros_like(signal)
        for i, sample in enumerate(signal):
            output[i] = self.process_sample(sample)
        return output
    
    def reset(self):
        """Reset the filter state (clear memory)."""
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
    
    def get_frequency_response(self, frequencies: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the frequency response of the filter.
        
        Parameters:
        -----------
        frequencies : np.ndarray, optional
            Array of frequencies to evaluate (Hz). If None, uses logarithmic spacing.
            
        Returns:
        --------
        tuple of (frequencies, magnitude_db, phase_degrees)
        """
        if frequencies is None:
            frequencies = np.logspace(0, np.log10(self.sample_rate / 2), 1000)
        
        # Calculate H(z) at z = e^(j*w)
        w = 2 * np.pi * frequencies / self.sample_rate
        z = np.exp(1j * w)
        
        # H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
        numerator = self.b0 + self.b1 * z**(-1) + self.b2 * z**(-2)
        denominator = self.a0 + self.a1 * z**(-1) + self.a2 * z**(-2)
        H = numerator / denominator
        
        #magnitude_db = 20 * np.log10(np.abs(H))
        # Add a small epsilon to prevent divides by zero
        magnitude_db = 20 * np.log10(np.abs(H) + 1e-10)
        phase_deg = np.angle(H, deg=True)
        
        return frequencies, magnitude_db, phase_deg
    
    def __repr__(self):
        return (f"BiquadFilter(type={self.filter_type.value}, "
                f"fs={self.sample_rate}Hz, f0={self.frequency}Hz, "
                f"Q={self.q}, gain={self.db_gain}dB)")


# Example usage and testing
if __name__ == "__main__":
    # Create a test signal: 1 second of signal with multiple frequency components
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Test signal: mix of frequencies
    test_signal = (np.sin(2 * np.pi * 100 * t) +  # 100 Hz
                   np.sin(2 * np.pi * 1000 * t) +  # 1 kHz
                   np.sin(2 * np.pi * 5000 * t))   # 5 kHz
    
    # Create different filter types
    lpf = BiquadFilter(FilterType.LPF, sample_rate, frequency=2000, q=0.707)
    hpf = BiquadFilter(FilterType.HPF, sample_rate, frequency=500, q=0.707)
    peaking = BiquadFilter(FilterType.PEAKING_EQ, sample_rate, frequency=1000, 
                           q=2.0, db_gain=6.0)
    
    # Process signals
    lpf_output = lpf.process(test_signal)
    hpf_output = hpf.process(test_signal)
    peaking_output = peaking.process(test_signal)
    
    print(f"Low-pass filter: {lpf}")
    print(f"High-pass filter: {hpf}")
    print(f"Peaking EQ: {peaking}")
    print(f"\nProcessed {len(test_signal)} samples")
    
    # Calculate frequency responses
    freqs, lpf_mag, _ = lpf.get_frequency_response()
    print(f"\nLPF magnitude at 1kHz: {lpf_mag[np.argmin(np.abs(freqs - 1000))]:.2f} dB")

