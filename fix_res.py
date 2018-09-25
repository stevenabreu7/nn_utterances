with open('result.txt', 'r') as f:
    lines = f.read().split('\n')

with open('result1.txt', 'w') as f:
    for line in lines:
        line = line.replace('tensor([[', '')
        line = line.replace("]], device='cuda:0')", '')
        f.write(line + '\n')