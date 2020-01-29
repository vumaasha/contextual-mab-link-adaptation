import itpp

'''
'''
def calculate_nrof_data_resource_elements(nrof_subcarriers):
    # Assume all resource elements are used for carrying data
    return nrof_subcarriers

''' Generate Turbo internal interleaver sequence for given block sizess
'''
def generate_turbo_internal_interleaver_sequence(block_length):
    f1 = 3
    f2 = 10
    
    internal_interleaver_sequence = itpp.ivec(block_length)
    for i in range(block_length):
        internal_interleaver_sequence[i] = (f1 * i + f2 * i * i) % block_length
    
    return internal_interleaver_sequence
                                    
''' Channel encode bits and interleave them using a randomized
    sequence interleaver 
'''    
def channel_encode_and_interleave_bits(bits):
    conv_code = itpp.comm.Convolutional_Code()
 
    generators = itpp.ivec(3)
    generators[0] = 91  # Octal 0133
    generators[1] = 101 # Octal 0145
    generators[2] = 125 # Octal 0175
    constraint_length = 7
    conv_code.set_generator_polynomials(generators, constraint_length)
     
    coded_bits = conv_code.encode(bits)
    
#     turbo_codec = itpp.comm.turbo_codec()
#     
#     gen = itpp.ivec('11, 13')
#     constraint_length = 4
#     
#     turbo_codec.set_parameters(gen, gen, constraint_length, itpp.ivec())
#     turbo_interleaver_sequence = generate_turbo_internal_interleaver_sequence(bits.length())
#     turbo_codec.set_interleaver(turbo_interleaver_sequence)
#     
#     coded_bits = itpp.bvec()
#     turbo_codec.encode(bits, coded_bits)
    
    sequence_interleaver_b = itpp.comm.sequence_interleaver_bin(coded_bits.length())
    sequence_interleaver_b.randomize_interleaver_sequence()
    
    interleaved_bits = sequence_interleaver_b.interleave(coded_bits) 
    
    return (interleaved_bits, sequence_interleaver_b.get_interleaver_sequence())

def deinterleave_and_channel_decode_symbols(symbols, interleaver_sequence):
        
    sequence_interleaver_d = itpp.comm.sequence_interleaver_double(symbols.length())
    sequence_interleaver_d.set_interleaver_sequence(interleaver_sequence)
    
    deinterleaved_symbols = sequence_interleaver_d.deinterleave(symbols, keepzeroes=0) 
        
    conv_code = itpp.comm.Convolutional_Code()
 
    generators = itpp.ivec(3)
    generators[0] = 91  # Octal 0133
    generators[1] = 101 # Octal 0145
    generators[2] = 125 # Octal 0175
    constraint_length = 7
    conv_code.set_generator_polynomials(generators, constraint_length)
     
    decoded_bits = conv_code.decode(deinterleaved_symbols)
    
#     turbo_codec = itpp.comm.turbo_codec()
#     
#     gen = itpp.ivec('11, 13')
#     constraint_length = 4
#     
#     turbo_codec.set_parameters(gen, gen, constraint_length, itpp.ivec())
#     turbo_interleaver_sequence = generate_turbo_internal_interleaver_sequence(symbols.length())
#     turbo_codec.set_interleaver(turbo_interleaver_sequence)
#     
#     decoded_bits = itpp.bvec()
#     turbo_codec.decode(symbols, decoded_bits, itpp.bvec())
    
    return decoded_bits
