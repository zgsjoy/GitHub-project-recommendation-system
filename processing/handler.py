#!/usr/bin/env python
# -*- coding: utf-8 -*-
# handler.py
from collections import defaultdict
from itertools import islice

import pandas as pd


class Handler:

    @staticmethod
    def normalize(vec, max_val, min_val):
        # get the normalized value using min-max normalization
        if max_val > min_val:
            return float(vec - min_val) / (max_val - min_val) + 0.01
        elif max_val == min_val:
            return vec / max_val
        else:
            print("error... maximum value is less than minimum value.")
            raise ArithmeticError

    @staticmethod
    def denormalize(vec, max_val, min_val):
        return min_val + (vec - 0.01) * (max_val - min_val)

    @staticmethod
    def get_data(path):
        data = []
        with open(path, 'r') as f:
            for line in f.readlines():
                line_arr = line.strip().split(',')
                # print(line_arr,line_arr[0])
                data.append([line_arr[0], line_arr[1], float(line_arr[2])])
        return data
