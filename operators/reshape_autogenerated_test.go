
package operators

import (
	"os"
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)


// TestReshape_extended_dims is autogenerated from test_reshape_extended_dims
func TestReshape_extended_dims(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Reshape{}

	

	attributes := []*onnx.AttributeProto{
		
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
	
	data := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(2, 3, 4),
			tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292})),
			gorgonia.WithName("data"))
	
	shape := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(4),
			tensor.WithBacking([]int64{3, 2, 2, 2})),
			gorgonia.WithName("shape"))
	
	
	reshapedT := tensor.New(
		tensor.WithShape(3, 2, 2, 2),
		tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292}))
	reshaped := new(gorgonia.Node)
	 
	o, err := op.Apply(
		data,shape,
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
	
	reshaped = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(reshapedT.Shape(), reshaped.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(reshapedT.Data(), reshaped.Value().Data(), 1e-5,"Tensors should be the same")
	
}

// TestReshape_negative_dim is autogenerated from test_reshape_negative_dim
func TestReshape_negative_dim(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Reshape{}

	

	attributes := []*onnx.AttributeProto{
		
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
	
	data := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(2, 3, 4),
			tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292})),
			gorgonia.WithName("data"))
	
	shape := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(3),
			tensor.WithBacking([]int64{6, -1, 2})),
			gorgonia.WithName("shape"))
	
	
	reshapedT := tensor.New(
		tensor.WithShape(6, 2, 2),
		tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292}))
	reshaped := new(gorgonia.Node)
	 
	o, err := op.Apply(
		data,shape,
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
	
	reshaped = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(reshapedT.Shape(), reshaped.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(reshapedT.Data(), reshaped.Value().Data(), 1e-5,"Tensors should be the same")
	
}

// TestReshape_one_dim is autogenerated from test_reshape_one_dim
func TestReshape_one_dim(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Reshape{}

	

	attributes := []*onnx.AttributeProto{
		
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
	
	data := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(2, 3, 4),
			tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292})),
			gorgonia.WithName("data"))
	
	shape := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1),
			tensor.WithBacking([]int64{24})),
			gorgonia.WithName("shape"))
	
	
	reshapedT := tensor.New(
		tensor.WithShape(24),
		tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292}))
	reshaped := new(gorgonia.Node)
	 
	o, err := op.Apply(
		data,shape,
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
	
	reshaped = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(reshapedT.Shape(), reshaped.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(reshapedT.Data(), reshaped.Value().Data(), 1e-5,"Tensors should be the same")
	
}

// TestReshape_reduced_dims is autogenerated from test_reshape_reduced_dims
func TestReshape_reduced_dims(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Reshape{}

	

	attributes := []*onnx.AttributeProto{
		
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
	
	data := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(2, 3, 4),
			tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292})),
			gorgonia.WithName("data"))
	
	shape := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(2),
			tensor.WithBacking([]int64{3, 8})),
			gorgonia.WithName("shape"))
	
	
	reshapedT := tensor.New(
		tensor.WithShape(3, 8),
		tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292}))
	reshaped := new(gorgonia.Node)
	 
	o, err := op.Apply(
		data,shape,
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
	
	reshaped = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(reshapedT.Shape(), reshaped.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(reshapedT.Data(), reshaped.Value().Data(), 1e-5,"Tensors should be the same")
	
}

// TestReshape_reordered_dims is autogenerated from test_reshape_reordered_dims
func TestReshape_reordered_dims(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Reshape{}

	

	attributes := []*onnx.AttributeProto{
		
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
	
	data := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(2, 3, 4),
			tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292})),
			gorgonia.WithName("data"))
	
	shape := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(3),
			tensor.WithBacking([]int64{4, 2, 3})),
			gorgonia.WithName("shape"))
	
	
	reshapedT := tensor.New(
		tensor.WithShape(4, 2, 3),
		tensor.WithBacking([]float32{0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292}))
	reshaped := new(gorgonia.Node)
	 
	o, err := op.Apply(
		data,shape,
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
	
	reshaped = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(reshapedT.Shape(), reshaped.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(reshapedT.Data(), reshaped.Value().Data(), 1e-5,"Tensors should be the same")
	
}