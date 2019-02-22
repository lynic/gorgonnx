package operators

import (
	"fmt"
	"os"
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestConv_with_npy2(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Conv{}

	attribute0Name := "kernel_shape"
	attribute0Type := onnx.AttributeProto_AttributeType(7)

	attribute0 := &onnx.AttributeProto{
		Name: &attribute0Name,
		Type: &attribute0Type,
		Ints: []int64{5, 5},
	}

	// attribute1Name := "auto_pad"
	// attribute1Type := onnx.AttributeProto_AttributeType(3)

	// attribute1 := &onnx.AttributeProto{
	// 	Name: &attribute1Name,
	// 	Type: &attribute1Type,
	// 	S:    []byte("SAME_UPPER"),
	// }

	attribute1Name := "pads"
	attribute1Type := onnx.AttributeProto_AttributeType(7)

	attribute1 := &onnx.AttributeProto{
		Name: &attribute1Name,
		Type: &attribute1Type,
		Ints: []int64{2, 2, 2, 2},
	}

	attribute2Name := "strides"
	attribute2Type := onnx.AttributeProto_AttributeType(7)

	attribute2 := &onnx.AttributeProto{
		Name: &attribute2Name,
		Type: &attribute2Type,
		Ints: []int64{1, 1},
	}

	attributes := []*onnx.AttributeProto{
		attribute0,
		attribute1,
		attribute2,
	}

	if len(attributes) != 0 {
		err := op.Init(attributes)
		t.Logf("Info: operator %#v", op)
		if err != nil {
			_, ok := err.(*onnx.ErrNotImplemented)
			if ok && skip {
				t.Skip(err)
			}

			t.Fatal(err)
		}
	}
	f, _ := os.Open("./Pooling66_Output_0.npy")
	xd := new(tensor.Dense)
	xd.ReadNpy(f)
	f.Close()
	fmt.Printf("x.shape = %v\n", xd.Shape())
	x := gorgonia.NodeFromAny(g,
		xd,
		gorgonia.WithName("x"))

	wf, _ := os.Open("./Parameter87.npy")
	wd := new(tensor.Dense)
	wd.ReadNpy(wf)
	wf.Close()
	fmt.Printf("w.shape = %v\n", wd.Shape())
	W := gorgonia.NodeFromAny(g,
		wd,
		gorgonia.WithName("W"))

	yT := new(tensor.Dense)
	yf, _ := os.Open("./Convolution110_Output_0.npy")
	yT.ReadNpy(yf)
	yf.Close()
	fmt.Printf("yT.shape = %v\n", yT.Shape())

	y := new(gorgonia.Node)

	o, err := op.Apply(
		x, W,
	)
	if err != nil {
		_, ok := err.(*onnx.ErrNotImplemented)
		if ok && skip {
			t.Skip(err)
		}
		_, ok = err.(*gorgonia.ErrNotImplemented)
		if ok && skip {
			t.Skip(err)
		}

		t.Fatal(err)
	}

	y = o[0]

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	fmt.Println(y.Value().Data())
	assert.Equal(yT.Shape(), y.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(yT.Data(), y.Value().Data(), 1e-5, "Tensors should be the same")
}
