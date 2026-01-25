import re
f = 'c:/Users/Crowsoda/CodingProjects/g20_demo/docs/creamy_chicken_dets.json'
content = open(f).read()
chunk_ids = re.findall(r'"chunk_id":\s*(\d+)', content)
print(f'Chunk IDs found: {len(chunk_ids)}')
print(f'Range: {min(int(x) for x in chunk_ids)} to {max(int(x) for x in chunk_ids)}')
boxes = re.findall(r'"box_id":\s*\d+', content)
print(f'Total boxes: {len(boxes)}')
