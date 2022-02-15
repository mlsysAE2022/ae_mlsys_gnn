import sys
import os
def exe():
    file = 'config.txt'
    cuda_home = os.getenv('CUDA_HOME')
    with open(file, 'w') as f:
        f.write(str(sys.executable) + '\n')
        f.write(str(cuda_home))

if __name__ == '__main__':
    exe()