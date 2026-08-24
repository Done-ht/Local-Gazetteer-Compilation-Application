# -*- coding: utf-8 -*-
"""修复 _gen_manual.py 中字符串内部嵌套的双引号"""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen_manual.py")
lines = open(p, encoding="utf-8").readlines()
fixed = []
for line in lines:
    if '"""' in line:
        fixed.append(line)
        continue
    quotes = [j for j, c in enumerate(line) if c == '"']
    if len(quotes) <= 2:
        fixed.append(line)
        continue
    result = []
    in_str = False
    for j, c in enumerate(line):
        if c == '"':
            if not in_str:
                in_str = True
                result.append(c)
            else:
                rest = line[j+1:].lstrip()
                if rest and rest[0] in ",)]:}":
                    in_str = False
                    result.append(c)
                elif j == len(line) - 1 or line[j+1] == "\n":
                    in_str = False
                    result.append(c)
                else:
                    result.append("\u300c")
        else:
            result.append(c)
    fixed.append("".join(result))
open(p, "w", encoding="utf-8").write("".join(fixed))
print("done")
