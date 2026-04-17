import smbus2
import struct
import time
import json
from collections import deque
from fft_analysis import compute_fft, band_summary

# --- Config ---
ADDR     = 0x53   # ADXL345 I2C address (SDO low)
SCALE    = 0.004  # G per LSB in +/-16G full-res mode

FFT_BUFFER_SIZE = 100
POLL_INTERVAL   = 0.25   # seconds - poll FIFO 4x/sec (watermark = 25 samples @ 100 Hz)

# --- ADXL345 init ---
def init_adxl345_fifo(bus):
    bus.write_byte_data(ADDR, 0x38, 0x00)        # FIFO_CTL: bypass (flushes FIFO)
    bus.write_byte_data(ADDR, 0x2D, 0x08)        # POWER_CTL: measure mode
    bus.write_byte_data(ADDR, 0x31, 0x0B)        # DATA_FORMAT: +/-16G full resolution
    bus.write_byte_data(ADDR, 0x2C, 0x0A)        # BW_RATE: 100 Hz ODR
    bus.write_byte_data(ADDR, 0x38, 0b01011001)  # FIFO_CTL: stream mode, 25-sample watermark

def read_fifo(bus):
    fifo_status = bus.read_byte_data(ADDR, 0x39)
    count   = fifo_status & 0x3F   # number of samples currently in FIFO
    samples = []
    for _ in range(count):
        raw     = bus.read_i2c_block_data(ADDR, 0x32, 6)
        x, y, z = struct.unpack_from('<3h', bytes(raw))
        samples.append((x * SCALE, y * SCALE, z * SCALE))
    return samples

# --- FFT buffers ---
fft_buffer_x = deque(maxlen=FFT_BUFFER_SIZE)
fft_buffer_y = deque(maxlen=FFT_BUFFER_SIZE)
fft_buffer_z = deque(maxlen=FFT_BUFFER_SIZE)

# --- Poll function ---
def poll_fifo(bus):
    samples = read_fifo(bus)
    if not samples:
        return

    ts = time.time()

    for i, (x, y, z) in enumerate(samples):
        sample_ts = ts - (len(samples) - i) / 100.0

        print(f"raw | ts: {sample_ts:.4f} | x: {x:.4f}G  y: {y:.4f}G  z: {z:.4f}G")

        # publish raw over MQTT when ready:
        # raw_payload = json.dumps({"x": round(x,4), "y": round(y,4),
        #                           "z": round(z,4), "ts": sample_ts})
        # client.publish("sensors/adxl345/raw", raw_payload, qos=1)

        fft_buffer_x.append(x)
        fft_buffer_y.append(y)
        fft_buffer_z.append(z)

    # run FFT once buffer is full (every 100 samples = 1 second at 100 Hz)
    if len(fft_buffer_x) == FFT_BUFFER_SIZE:
        freqs, x_amp, y_amp, z_amp = compute_fft(
            list(fft_buffer_x),
            list(fft_buffer_y),
            list(fft_buffer_z),
        )
        summary = band_summary(freqs, x_amp, y_amp, z_amp, low_hz=4.0, high_hz=7.0)

        print(f"fft | X peak: {summary['x']['peak_hz']} Hz  amp: {summary['x']['peak_amp_g']} G")
        print(f"fft | Y peak: {summary['y']['peak_hz']} Hz  amp: {summary['y']['peak_amp_g']} G")
        print(f"fft | Z peak: {summary['z']['peak_hz']} Hz  amp: {summary['z']['peak_amp_g']} G")

        # publish FFT result over MQTT when ready:
        # fft_payload = json.dumps({"fft": summary, "ts": ts})
        # client.publish("sensors/adxl345/fft", fft_payload, qos=1)

# --- Main ---
bus = smbus2.SMBus(1)
init_adxl345_fifo(bus)

print(f"Acquisition running - polling FIFO every {POLL_INTERVAL*1000:.0f} ms (Ctrl+C to stop)")

try:
    next_tick = time.monotonic()
    while True:
        next_tick += POLL_INTERVAL
        poll_fifo(bus)
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
finally:
    bus.close()
    print("Cleaned up.")