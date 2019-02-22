import onnx
from onnx import numpy_helper, helper
from onnx import TensorProto
# import onnxruntime as onnxrt
import onnxruntime.backend as backend 
# from onnx_tf import backend
# import caffe2.python.onnx.backend as backend

import numpy as np

def _extract_value_info(arr, name):  # type: (np.ndarray, Text) -> onnx.ValueInfoProto
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
        shape=arr.shape)


def expect(node,  # type: onnx.NodeProto
           inputs,  # type: Sequence[np.ndarray]
           outputs,  # type: Sequence[np.ndarray]
           name,  # type: Text
           **kwargs  # type: Any
           ):  # type: (...) -> None
    present_inputs = [x for x in node.input if (x != '')]
    present_outputs = [x for x in node.output if (x != '')]
    inputs_vi = [_extract_value_info(arr, arr_name)
                 for arr, arr_name in zip(inputs, present_inputs)]
    outputs_vi = [_extract_value_info(arr, arr_name)
                  for arr, arr_name in zip(outputs, present_outputs)]
    graph = onnx.helper.make_graph(
        nodes=[node],
        name=name,
        inputs=inputs_vi,
        outputs=outputs_vi)
    kwargs[str('producer_name')] = 'backend-test'
    model_def = onnx.helper.make_model(graph, **kwargs)
    onnx.checker.check_model(model_def)
    pm = backend.prepare(model_def)
    outs = list(pm.run(inputs))
    oo = np.asarray(outs[0])
    print(np.asarray(outs[0]))
    # import ipdb; ipdb.set_trace()
    for ref_o, o in zip(outputs, outs):
        np.testing.assert_almost_equal(ref_o, o)
    return outs

def export_conv_with_strides6():  # type: () -> None
        x = np.load("./Pooling66_Output_0.npy")
        W = np.load("./Parameter87.npy")
        # import ipdb; ipdb.set_trace()
        # Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
        node_with_asymmetric_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2],
            # auto_pad="SAME_UPPER",
            strides=[1, 1],  # Default values for other attributes: dilations=[1, 1], groups=1
        )
        y_with_asymmetric_padding = np.load("./Convolution110_Output_0.npy")
        expect(node_with_asymmetric_padding, inputs=[x, W], outputs=[y_with_asymmetric_padding],
               name='test_conv_with_strides_and_asymmetric_padding')



if __name__ == "__main__":
    export_conv_with_strides6()