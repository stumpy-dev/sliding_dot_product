# Sliding Dot product, (Circular) Convolution, and Overlap-Add!

One way to compute the sliding-dot-product (sdp) between a query Q and a time series T is
FFT-based convolution. But first, let's start with a simple example to understand how the concepts are related.

```
T = [1, 2, 3, 4]
Q = [A, B]
```

Let's first see the sdp:

```
sdp(T, Q) = [1*A + 2*B, 2*A + 3*B, 3*A + 4*B]
```

To compute this using FFT-based convolution, we need to reverse the query Q and pad it with zeros to match the length of T.

```
T = [1, 2, 3, 4]
Q_reversed_padded = [B, A, 0, 0]
```

Then, we use circular convolution to compute the result, $QT_{conv}$. The formula for circular convolution in time domain is:


$$QT_{conv}[i] = \sum_{j=0}^{N-1} T[j] \cdot Q[(i - j) \mod N]$$

where $N$ is the length of the sequences (in this case, 4). 

Let's compute the circular convolution:

```
conv(T, Q_reversed_padded) = [
    1B + 4A,
    1A + 2B,
    2A + 3B,
    3A + 4B, 
] = 
```

In sdp, we only care about the slice [M-1:N], which is called 'valid' mode in convolution terminology, and that slice gives us the same result as sdp. 


Now, let's consider one more elelment in T, say: 
<br> 
`T_new = [1, 2, 3, 4, 5]`

We know that the sdp between `Q` and `T_new` is:

```
sdp(T_new, Q) = [
    1A + 2B,
    2A + 3B,
    3A + 4B,
    4A + 5B
]
```

So, the only new item is `4A + 5B`. However, if we look closely, we can see that the only new multiplication we need to perform is `5*B` IF we already have `4A` computed from the previous sdp. Note that we did not compute `4A` previously because the circular convolution wraps around and mixes it with `1B`. However, if there is a way to get that `4A` only in the previous step, we can only compute the new multiplication `5*B` and add it to `4A` to get the new sdp value. If the circular convolution avoids the wrap-around, we can achieve this. This is where zero-padding comes into play! We can see `T_new` as two parts: `T1=[1, 2, 3, 4]` and `T2=[5]`. 

```
T1_with_0 = [1, 2, 3, 4, 0]
Q_reversed_padded = [B, A, 0, 0, 0]

conv(T1_with_0, Q_reversed_padded) = [
    1B + 0A,
    1A + 2B,
    2A + 3B,
    3A + 4B,
    4A + 0B,
]
```

Great! We have `4A`! How about `5B` part? We can compute this by convolving `T2` with `Q` as well:

``` 
T2_with_0 = [5, 0, 0, 0, 0]
Q_reversed_padded = [B, A, 0, 0, 0]

conv(T2_with_0, Q_reversed_padded) = [
    5B + 0A,
    0,
    0,
    0,
    0,
]
```

So, we can see the sdp can be computed as:

```
SDP([A, B], [1, 2, 3, 4, 5]) = [
    1A + 2B,    # from "valid" portion 
    2A + 3B,    # from "valid" portion
    3A + 4B,    # from "valid" portion
    (4A + 0B) + (5B + 0A) = 4A + 5B    
    # last element of conv(T1...) + first element of conv(T2...)
]
```

This is the basic concept behind overlap-add. As far as I understand, there is no need to have same-size chunks. However, having same-size chunks makes the implementation easier. Note that we need `(M-1)` zeros to avoid wrap-around, where `M` is the length of the query. So, for a given block size of `B`, we need to have chunks of size `B - (M - 1)`. That last chunk can be padded with more zeroes to make it of size `B`. We know that the first `chunksize` elements of each circular convolution are what we need for the sdp, while adding the last `(M-1)` elements of `c-th` chunk  to the first `(M-1)` elements of `(c+1)-th` chunk.

