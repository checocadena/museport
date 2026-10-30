import struct, numpy as np
# load bf16 weights as fp32
f=open('gpt2_124M_bf16.bin','rb'); h=struct.unpack('256i',f.read(1024))
V,Vp,C,maxT,L,NH=h[3],h[7],h[6],h[2],h[4],h[5]
def rd(n):
    raw=f.read(n*2)
    u=np.frombuffer(raw,dtype='<u2').astype(np.uint32)
    return (u<<16).view(np.float32)
names=['wte','wpe','ln1w','ln1b','qkvw','qkvb','apw','apb','ln2w','ln2b','fcw','fcb','fpw','fpb','lnfw','lnfb']
ns=[Vp*C,maxT*C,L*C,L*C,L*3*C*C,L*3*C,L*C*C,L*C,L*C,L*C,L*4*C*C,L*4*C,L*C*4*C,L*C,C,C]
P={n:rd(s) for n,s in zip(names,ns)}
sf=open('gpt2_124M_debug_state.bin','rb'); sh=struct.unpack('256i',sf.read(1024)); B,T=sh[2],sh[3]
x=np.fromfile(sf,dtype=np.int32,count=B*T); y=np.fromfile(sf,dtype=np.int32,count=B*T)
lg=np.fromfile(sf,dtype=np.float32,count=B*T*V)
def ln_fwd(inp,w,b):
    m=inp.mean(-1,keepdims=True); v=((inp-m)**2).mean(-1,keepdims=True); s=1/np.sqrt(v+1e-5)
    return (s*(inp-m))*w+b, m[...,0], s[...,0]
enc=P['wte'].reshape(-1,C)[x]+np.tile(P['wpe'].reshape(-1,C)[:T],(B,1))
enc=enc.reshape(B*T,C)
ln1,m1,r1=ln_fwd(enc,P['ln1w'][:C],P['ln1b'][:C])
print('ref enc maxabs',np.abs(enc).max(),'ln1 maxabs',np.abs(ln1).max())
# layer 0 qkv
qkv=ln1@P['qkvw'][:3*C*C].reshape(3*C,C).T+P['qkvb'][:3*C]
print('ref qkv maxabs',np.abs(qkv).max())
# attention
HS=C//NH
Q=qkv[:,:C].reshape(B*T,NH,HS).transpose(0,1,2) # (BT,NH,HS) -> per batch
import math
scale=1/math.sqrt(HS)
atty=np.zeros((B*T,C),dtype=np.float64)
for b in range(B):
    Qb=qkv[b*T:(b+1)*T,:C].reshape(T,NH,HS).transpose(1,0,2)  # (NH,T,HS)
    Kb=qkv[b*T:(b+1)*T,C:2*C].reshape(T,NH,HS).transpose(1,0,2)
    Vb=qkv[b*T:(b+1)*T,2*C:].reshape(T,NH,HS).transpose(1,0,2)
    pre=Qb@Kb.transpose(0,2,1)*scale
    # causal softmax
    m2=np.where(np.tri(T,dtype=bool),pre,-1e10)
    e=np.exp(m2-m2.max(-1,keepdims=True)); e=e/e.sum(-1,keepdims=True)
    e=e*np.tri(T)
    Y=e@Vb  # (NH,T,HS)
    atty[b*T:(b+1)*T]=Y.transpose(1,0,2).reshape(T,C)
print('ref atty maxabs',np.abs(atty).max())
Wap=P['apw'][:C*C].reshape(C,C)
print('Wap layout check: row-major (OC x C): Wap[0,:8]=',Wap[0,:8])
proj=atty@Wap.T+P['apb'][:C]
print('ref proj maxabs',np.abs(proj).max())
res2=enc+proj
ln2,_,_=ln_fwd(res2,P['ln2w'][:C],P['ln2b'][:C])
print('ref res2 maxabs',np.abs(res2).max(),'ln2 maxabs',np.abs(ln2).max())
