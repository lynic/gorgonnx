package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/gorgonnx/onnx"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
func (d *Decoder) convOp(nx *onnx.NodeProto) error {
	var kernelShape tensor.Shape
	var pad, stride []int
	for _, attr := range nx.Attribute {
		switch *attr.Name {
		case "kernel_shape":
			shape := make([]int, len(attr.Ints))
			for i, v := range attr.Ints {
				shape[i] = int(v)
			}
			kernelShape = tensor.Shape(shape)
		case "strides":
			stride = make([]int, len(attr.Ints))
			for i, v := range attr.Ints {
				stride[i] = int(v)
			}
		case "auto_pad":
			if attr.S == nil {
				return fmt.Errorf("auto_pad specified without value")
			}
			switch string(attr.S) {
			case "NOTSET":
			case "SAME_UPPER":
			case "SAME_LOWER":
			case "VALID":
			default:
				return fmt.Errorf("Invalid auto_pad value: %v", string(attr.S))

			}
		case "pads":
		case "group":
		case "dilations":
		default:
			return fmt.Errorf("Unknown attribute: %v for convolution operator", attr.Name)
		}
	}
	n, err := gorgonia.Conv2d(d.db[nx.Input[0]], d.db[nx.Input[1]], kernelShape, pad, stride)
	if err != nil {
		return fmt.Errorf("Cannot apply Convolution operator: %v", err)
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
func (d *Decoder) reshapeOp(nx *onnx.NodeProto) error {
	if len(nx.Input) != 2 {
		return fmt.Errorf("Not enough input parameters for reshape")
	}
	n, err := gorgonia.Reshape(d.db[nx.Input[0]], d.db[nx.Input[1]].Shape())
	if err != nil {
		return fmt.Errorf("Cannot reshape: %v", err)
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n
	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
func (d *Decoder) addOp(nx *onnx.NodeProto) error {
	n, err := gorgonia.Add(d.db[nx.Input[0]], d.db[nx.Input[1]])
	if err != nil {
		return fmt.Errorf("Cannot Add: %v", err)
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n

	return nil
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
func (d *Decoder) reluOp(nx *onnx.NodeProto) error { return nil }

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
func (d *Decoder) maxPoolOp(nx *onnx.NodeProto) error { return nil }

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
func (d *Decoder) matMulOp(nx *onnx.NodeProto) error {
	n, err := gorgonia.Mul(d.db[nx.Input[0]], d.db[nx.Input[1]])
	if err != nil {
		return fmt.Errorf("Cannot Multiply: %v", err)
	}
	d.g.AddNode(n)
	d.db[nx.Output[0]] = n

	return nil

}