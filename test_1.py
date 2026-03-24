import sys
sys.path.insert(0, '.')
import os
os.environ['LIMIT'] = '1'

# Modify train.py LIMIT in memory
exec(open('train.py').read().replace('LIMIT = None', 'LIMIT = 1'))
