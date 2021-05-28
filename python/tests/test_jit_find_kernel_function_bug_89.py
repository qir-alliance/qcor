import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestJITBug89(unittest.TestCase):
    def test_89(self):
        set_qpu('qpp', {'shots': 100})
        
        @qjit
        def t(q: qreg):
            H(q)
            Measure(q)
        
        
        q = qalloc(2)
        t(q)
        q.print()
        counts = q.counts()
        print(counts)
        self.assertEqual(len(counts), 4)


if __name__ == '__main__':
  unittest.main()