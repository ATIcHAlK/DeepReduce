# DeepReduce

A Sparse-tensor CommunicationFramework for Federated Deep Learning

## Prerequisites

The code is built with following libraries:

- Python >= 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.4
- [TensorFlow](https://www.tensorflow.org/) >= 1.14
- [GRACE](https://github.com/sands-lab/grace) >=1.0

## Benchmarks

We use the following benchmarks to run our experiments:

- [Image Classification/tf_cnn_benchmarks](https://github.com/sands-lab/grace-benchmarks/tree/master/tensorflow/Classification/tf_cnn_benchmarks) [TensorFlow] ResNet-20, ResNet-50
- [Image_Classification/Cifar10](https://github.com/sands-lab/grace-benchmarks/tree/master/torch/cifar10) [PyTorch] ResNet-20
- [Recommendation/NCF](https://github.com/sands-lab/grace-benchmarks/tree/master/torch/Recommendation/NCF) [PyTorch] NCF

## Usage

For the usage of GRACE and environment setup etc., please check the guides [here](https://github.com/sands-lab/grace).

First, create a GRACE instance from `params`. `params` should include parameters for both GRACE and DeepReduce. The valid parameter options for DeepReduce is listed as below:

```python
'''
'deepreduce': None, 'value', 'index', 'both'
'value': None, 'polyfit', ...(other custom methods)
'index': None, 'bloom', ...(other custom methods)
'''
from grace_dl.dist.helper import grace_from_params
params = {'compressor': 'topk', 'memory': 'residual', 'communicator': 'allgather', 'compress_ratio': 0.01, 'deepreduce':'index', 'index':'bloom'}
grc = grace_from_params(params)
```

Once you get a desired GRACE instance, warp the compressor by DeepReduce. After that, you can use DeepReduce in the same way as GRACE.

```python
deepreduce_wrapper = {'value': ValueCompressor,
                      'index': IndexCompressor,
                      'both': DeepReduce}
DReduce = deepreduce_wrapper[deepreduce](grc.compressor, params)
grc.compressor = DReduce
```

