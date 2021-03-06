
package operators

import (
	"os"
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)


// TestMatmul_2d is autogenerated from test_matmul_2d
func TestMatmul_2d(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Matmul{}

	

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
	
	a := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(3, 4),
			tensor.WithBacking([]float32{1.7640524, 0.4001572, 0.978738, 2.2408931, 1.867558, -0.9772779, 0.95008844, -0.1513572, -0.10321885, 0.41059852, 0.14404356, 1.4542735})),
			gorgonia.WithName("a"))
	
	b := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(4, 3),
			tensor.WithBacking([]float32{0.7610377, 0.121675014, 0.44386324, 0.33367434, 1.4940791, -0.20515826, 0.3130677, -0.85409576, -2.5529897, 0.6536186, 0.8644362, -0.742165})),
			gorgonia.WithName("b"))
	
	
	cT := tensor.New(
		tensor.WithShape(3, 3),
		tensor.WithBacking([]float32{3.247133, 1.9136808, -3.4609182, 1.2937015, -2.1752005, -1.2837971, 1.0540882, 1.7350072, -1.5771054}))
	c := new(gorgonia.Node)
	 
	o, err := op.Apply(
		a,b,
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
	
	c = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(cT.Shape(), c.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(cT.Data(), c.Value().Data(), 1e-5,"Tensors should be the same")
	
}

// TestMatmul_3d is autogenerated from test_matmul_3d
func TestMatmul_3d(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Matmul{}

	

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
	
	a := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(2, 3, 4),
			tensor.WithBacking([]float32{2.2697546, -1.4543657, 0.045758516, -0.18718386, 1.5327792, 1.4693588, 0.15494743, 0.37816253, -0.88778573, -1.9807965, -0.34791216, 0.15634897, 1.2302907, 1.2023798, -0.3873268, -0.30230275, -1.048553, -1.420018, -1.7062702, 1.9507754, -0.5096522, -0.4380743, -1.2527953, 0.7774904})),
			gorgonia.WithName("a"))
	
	b := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(2, 4, 3),
			tensor.WithBacking([]float32{-1.6138978, -0.21274029, -0.89546657, 0.3869025, -0.51080513, -1.1806322, -0.028182229, 0.42833188, 0.06651722, 0.3024719, -0.6343221, -0.36274117, -0.67246044, -0.35955316, -0.8131463, -1.7262826, 0.17742614, -0.40178093, -1.6301984, 0.46278226, -0.9072984, 0.051945396, 0.7290906, 0.12898292})),
			gorgonia.WithName("b"))
	
	
	cT := tensor.New(
		tensor.WithShape(2, 3, 3),
		tensor.WithBacking([]float32{-4.2837567, 0.3983639, -0.24447522, -1.7952335, -1.2501478, -3.2341933, 0.7235164, 0.9524713, 3.053718, -2.287253, -0.62867534, -1.1710705, 6.0393553, 0.7577226, 3.222876, 3.181653, 0.09261069, 1.8273739}))
	c := new(gorgonia.Node)
	 
	o, err := op.Apply(
		a,b,
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
	
	c = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(cT.Shape(), c.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(cT.Data(), c.Value().Data(), 1e-5,"Tensors should be the same")
	
}

// TestMatmul_4d is autogenerated from test_matmul_4d
func TestMatmul_4d(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Matmul{}

	

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
	
	a := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1, 2, 3, 4),
			tensor.WithBacking([]float32{1.1394007, -1.2348258, 0.40234163, -0.6848101, -0.87079716, -0.5788497, -0.31155252, 0.05616534, -1.1651498, 0.9008265, 0.46566245, -1.5362437, 1.4882522, 1.8958892, 1.1787796, -0.17992483, -1.0707526, 1.0544517, -0.40317693, 1.222445, 0.20827498, 0.97663903, 0.3563664, 0.7065732})),
			gorgonia.WithName("a"))
	
	b := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(1, 2, 4, 3),
			tensor.WithBacking([]float32{0.01050002, 1.7858706, 0.12691209, 0.40198937, 1.8831507, -1.347759, -1.270485, 0.9693967, -1.1731234, 1.9436212, -0.41361898, -0.7474548, 1.922942, 1.4805148, 1.867559, 0.90604466, -0.86122566, 1.9100649, -0.26800337, 0.8024564, 0.947252, -0.15501009, 0.61407936, 0.9222067})),
			gorgonia.WithName("b"))
	
	
	cT := tensor.New(
		tensor.WithShape(1, 2, 3, 3),
		tensor.WithBacking([]float32{-2.3266034, 0.38273817, 1.8487196, 0.26315218, -2.970441, 0.99314374, -3.2276042, 0.70241654, -0.75997543, 4.291556, 1.4060221, 7.3513436, -1.1850535, -2.0662394, 0.75981444, 1.080346, 0.1871081, 3.243585}))
	c := new(gorgonia.Node)
	 
	o, err := op.Apply(
		a,b,
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
	
	c = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(cT.Shape(), c.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(cT.Data(), c.Value().Data(), 1e-5,"Tensors should be the same")
	
}
