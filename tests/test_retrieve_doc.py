import unittest

from bapred.retrieve_doc import *


class RetrieveDocTestCase(unittest.TestCase):
    def test_include_all_subs(self):
        s = 'abcde'

        subs_a = ['ab', 'de']
        subs_b = ['ab', 'ae']
        subs_c = ['ab', 'f']
        subs_d = ['AB', 'DE']

        self.assertEqual(include_all_subs(s, subs_a), True)
        self.assertEqual(include_all_subs(s, subs_b), False)
        self.assertEqual(include_all_subs(s, subs_c), False)
        self.assertEqual(include_all_subs(s, subs_d), True)

    def test_filter_with_substring(self):
        strings = ['I have a dream', 'He has a dream']

        subs_a = ['I', 'have']
        expected_a = [(0, strings[0])]

        subs_b = ['a', 'dream']
        expected_b = [(i,s) for i, s in enumerate(strings)]

        self.assertEqual(filter_with_substring(strings, subs_a), expected_a)
        self.assertEqual(filter_with_substring(strings, subs_b), expected_b)




if __name__ == '__main__':
    unittest.main()
