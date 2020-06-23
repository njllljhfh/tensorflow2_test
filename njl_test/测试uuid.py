# -*- coding:utf-8 -*-
import uuid

a = list()
for i in range(1000000):
    a.append(uuid.uuid4())

print(len(a))

b = set(a)
print(len(b))
