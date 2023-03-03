RuntimeError: Conv3D is not supported on MPS

[torch.nn.Conv3D on MPS backend #77818](https://github.com/pytorch/pytorch/issues/77818)

[General MPS op coverage tracking issue #77764](https://github.com/pytorch/pytorch/issues/77764)

Attempted to use PYTORCH_ENABLE_MPS_FALLBACK=1 env var to overcome the "Conv3D is not supported on MPS" issue. Did not work.

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python ./inference.py
MPS is available. PyTorch install was built with MPS enabled.
Traceback (most recent call last):
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/./inference.py", line 100, in <module>
    inference(X_batch, Y_batch, NX_batch, config, device)
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/./inference.py", line 21, in inference
    output = model.forward(X_batch, NX_batch)
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/utils/models.py", line 135, in forward
    X = self.Phi_fun(X_batch) # cnn out: batchsize, num of nodes, feature length
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/venv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/utils/models.py", line 68, in forward
    out = self.Conv_1(out)
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/venv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/utils/models.py", line 42, in forward
    y = self.leakyrelu(self.bn(self.conv3d(x)))
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/venv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/venv/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 613, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/Users/darreng/code/Linking-cnn-with-gcn-pytorch/venv/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 608, in _conv_forward
    return F.conv3d(
RuntimeError: Conv3D is not supported on MPS
```


