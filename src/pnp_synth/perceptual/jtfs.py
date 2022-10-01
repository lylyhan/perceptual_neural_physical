"""
# kymatio main branch parameters
jtfs_params = dict(
            J = 14, #scale
            shape = (2**16, ), 
            T = 2**16,
            J_fr = 3, 
            Q = 1,
            Q_fr = 1,
            F = 2
            )
"""
#wavespin parameters
jtfs_params = dict(
            J = 14, #scale
            shape = (2**16, ), 
            Q = 1, #filters per octave, frequency resolution
            T = 2**13,  #local averaging
            F = 2,
            max_pad_factor=1,
            max_pad_factor_fr=1,
            average = True,
            average_fr = True,
)