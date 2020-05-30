using CUDAdrv, CUDAnative,CuArrays
function mat_add(a, b, c,l)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x
    j=  threadIdx().y + (blockIdx().y-1)*blockDim().y
    
    if(i<=l && j<=l)
    c[i,j] = a[i,j] + b[i,j]
    end
    return
end

# generate some data
len=2^24
a = Vector(1:len)
p= Int(len^0.5)
a=reshape(a , (p,p))

b=Vector(2:2:2*len)
b=reshape(b, (p,p))

# allocate & upload to the GPU
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

# execute and fetch results
@cuda blocks=(ceil(Int,len/1024),ceil(Int,len/1024)) threads=(32,32) mat_add(d_a, d_b, d_c,p)
Array(d_c)
