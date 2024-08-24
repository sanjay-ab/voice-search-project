import os
import sys
import tqdm
c=0
pair_file=sys.argv[1]
output_file=sys.argv[2]
set_sizes=100 # number of different speaker in a batch
              # should be the number of speakers in each FAISS index used to find pairs
with open(pair_file, "r") as file:
    file_items = file.readlines()
spks={}
batch_ind=0
# reading pairs
for pair in tqdm.tqdm(file_items):
    fid1,spk1,s1,e1,fid2,spk2,s2,e2,sim=pair.split(' ')
    if spk1 not in spks:
        spks[spk1]=set()
    if spk2 not in spks:
        spks[spk2]=set()
    spks[spk1].add(spk1)
    spks[spk1].add(spk2)
    spks[spk2].add(spk1)
    spks[spk2].add(spk2)

# getting uniq sets
uniq_sets=[]
for spk in spks:
    if len(spks[spk])!=set_sizes:
        #print('short sets',len(spks[spk]))
        continue
    if len(uniq_sets)==0:
        uniq_sets.append(spks[spk])
        continue
    is_new=True
    for s in uniq_sets:
        if spks[spk]==s:
            is_new=False
            break
    if is_new:
        uniq_sets.append(spks[spk])
print('nb of uniq sets',len(uniq_sets),' each of lenght',set_sizes)
batches={}
# assigning a batch count to each speaker
set_count=0
for s in uniq_sets:
    for spk in s:
        assert spk not in batches,(spk,batches)
        batches[spk]=str(set_count)
    set_count+=1
pairs=[]
for pair in tqdm.tqdm(file_items):
    fid1,spk1,s1,e1,fid2,spk2,s2,e2,sim=pair.rstrip().split(' ')
    if spk1 not in batches or spk2 not in batches:
        continue
    assert batches[spk1]==batches[spk2],(batches[spk1],spk1,spk2)
    pairs.append(' '.join((fid1,spk1,s1,e1,fid2,spk2,s2,e2,sim,batches[spk1])))
    
print('nb of pairs',len(pairs))
with open(output_file,'w') as buf:
    buf.write('\n'.join(pairs)+'\n')
