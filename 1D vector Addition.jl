using CUDAdrv, CUDAnative,CuArrays

# kernel
function kernel_vadd(a, b, c,l)
    i = threadIdx().x + (blockIdx().x-1)*(1024)
    if(i<=l)
    c[i] = a[i] + b[i]
    end
    return
end

# generate some data
len = 5000
a = [i for i=1:len]
b = [j for j=1:len]

# allocate & upload to the GPU
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

# execute and fetch results
@cuda blocks=ceil(Int,len/1024) threads=1024 kernel_vadd(d_a, d_b, d_c,len)
print(d_c)
