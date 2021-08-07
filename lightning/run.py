import os
for f in [0, 1, 2, 3, 4]:
    print(f'python3 train.py --aux --psd --f {f} --g 0')
    os.system(f'python3 train.py --aux --psd --f {f} --g 0')