#! /usr/local/bin/python3

import subprocess


def do(env, limit=-1):
    p1 = subprocess.Popen(['python3', 'preprocess.py', str(env), str(limit), '4', '0'])
    p2 = subprocess.Popen(['python3', 'preprocess.py', str(env), str(limit), '4', '1'])
    p3 = subprocess.Popen(['python3', 'preprocess.py', str(env), str(limit), '4', '2'])
    p4 = subprocess.Popen(['python3', 'preprocess.py', str(env), str(limit), '4', '3'])
    p1.wait()
    p2.wait()
    p3.wait()
    p4.wait()


if __name__ == '__main__':
    do('train')
    do('test')
