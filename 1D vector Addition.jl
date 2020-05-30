using CUDAdrv, CUDAnative,CuArrays
# kernel
function kernel_vadd(a, b, c)
    i = threadIdx().x
    c[i] = a[i] + b[i]
    return
end

# generate some data
len = 1024
a = [i for i=1:len]
b = [j for j=len:-1:1]

# allocate & upload to the GPU
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

# execute and fetch results
@cuda blocks=1 threads=len kernel_vadd(d_a, d_b, d_c)
print(d_c)
