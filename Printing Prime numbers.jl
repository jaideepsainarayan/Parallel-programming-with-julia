using CUDAdrv, CUDAnative,CuArrays

function prime(p,a)
    n=0
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x
    
    for k=1:i
    if(p>=i>=2 && k<=i && i%k==0)
       n+=1
    end
    end

    if(n==2)
    a[i]=i
    end

  return
end

N=100000 # 'N' is the input 
# generate some data

a = [0 for i=1:N]
# allocate & upload to the GPU

d_a=CuArray(a)

# execute and fetch results

@cuda blocks=(ceil(Int,N/1024)) threads=(1024) prime(N,d_a)
synchronize()

a=Array(d_a)
filter!(x->x≠0,a)
