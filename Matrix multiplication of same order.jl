using CUDAdrv, CUDAnative,CuArrays
function mat_add(a, b, c,l)
    i = (threadIdx().x) + (blockIdx().x-1)*(blockDim().x)
    j=  (threadIdx().y) + (blockIdx().y-1)*(blockDim().y)
    sum=0
    if(i<=l && j<=l)
    for k=1:l
      sum= sum +a[(i-1) * l + k] * b[(k-1) * l + j]
      end
      c[(i-1) * l + j]=sum
    end
    return
end

# generate some data
len=9 # number of elements in a square matrix(should be perfect square)
p= Int(len^0.5) # order of the matrix(p*p)
a = Vector(1:len)
a=reshape(a , (p,p))
a=transpose(a)

b=Vector(2:2:2*len)
b=reshape(b, (p,p))
b=transpose(b)


# allocate & upload to the GPU
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

# execute and fetch results
@cuda blocks=(ceil(Int,len/1024),ceil(Int,len/1024)) threads=(32,32,1) mat_add(d_a, d_b, d_c,p)
Array(d_c)
