import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

file_name = "eval-set.csv"
lines = open(file_name, encoding='utf-8').readlines()
for line in lines:
    print(line)

