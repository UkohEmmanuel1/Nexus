from gguf import GGUFReader
r = GGUFReader('checkpoints/qwen2.5-1.5b-instruct-q4_k_m.gguf')
names = [t.name for t in list(r.tensors)[:30]]
print('\n'.join(names))
total = len(list(r.tensors))
print(f"\nTotal tensors: {total}")
r.close()
