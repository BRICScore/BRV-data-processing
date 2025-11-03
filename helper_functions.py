
def u2_to_i(value, b1, b2, b3):
    value = (b1 << 16) | (b2 << 8) | b3
    return value - (1 << 24) if value & (1 << 23) else value

def adc_to_voltage(adc_value):
    return (adc_value + 2**23) * 10**(-9) * 23.84
