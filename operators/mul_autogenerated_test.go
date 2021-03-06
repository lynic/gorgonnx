
package operators

import (
	"os"
	"testing"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)


// TestMul_ is autogenerated from test_mul
func TestMul_(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Mul{}

	

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
	
	x := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(3, 4, 5),
			tensor.WithBacking([]float32{1.7640524, 0.4001572, 0.978738, 2.2408931, 1.867558, -0.9772779, 0.95008844, -0.1513572, -0.10321885, 0.41059852, 0.14404356, 1.4542735, 0.7610377, 0.121675014, 0.44386324, 0.33367434, 1.4940791, -0.20515826, 0.3130677, -0.85409576, -2.5529897, 0.6536186, 0.8644362, -0.742165, 2.2697546, -1.4543657, 0.045758516, -0.18718386, 1.5327792, 1.4693588, 0.15494743, 0.37816253, -0.88778573, -1.9807965, -0.34791216, 0.15634897, 1.2302907, 1.2023798, -0.3873268, -0.30230275, -1.048553, -1.420018, -1.7062702, 1.9507754, -0.5096522, -0.4380743, -1.2527953, 0.7774904, -1.6138978, -0.21274029, -0.89546657, 0.3869025, -0.51080513, -1.1806322, -0.028182229, 0.42833188, 0.06651722, 0.3024719, -0.6343221, -0.36274117})),
			gorgonia.WithName("x"))
	
	y := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(3, 4, 5),
			tensor.WithBacking([]float32{-0.67246044, -0.35955316, -0.8131463, -1.7262826, 0.17742614, -0.40178093, -1.6301984, 0.46278226, -0.9072984, 0.051945396, 0.7290906, 0.12898292, 1.1394007, -1.2348258, 0.40234163, -0.6848101, -0.87079716, -0.5788497, -0.31155252, 0.05616534, -1.1651498, 0.9008265, 0.46566245, -1.5362437, 1.4882522, 1.8958892, 1.1787796, -0.17992483, -1.0707526, 1.0544517, -0.40317693, 1.222445, 0.20827498, 0.97663903, 0.3563664, 0.7065732, 0.01050002, 1.7858706, 0.12691209, 0.40198937, 1.8831507, -1.347759, -1.270485, 0.9693967, -1.1731234, 1.9436212, -0.41361898, -0.7474548, 1.922942, 1.4805148, 1.867559, 0.90604466, -0.86122566, 1.9100649, -0.26800337, 0.8024564, 0.947252, -0.15501009, 0.61407936, 0.9222067})),
			gorgonia.WithName("y"))
	
	
	zT := tensor.New(
		tensor.WithShape(3, 4, 5),
		tensor.WithBacking([]float32{-1.1862555, -0.14387779, -0.7958572, -3.8684149, 0.3313536, 0.39265162, -1.5488327, -0.07004543, 0.0936503, 0.021328703, 0.105020806, 0.18757643, 0.86712694, -0.15024745, 0.17858467, -0.22850356, -1.3010398, 0.118755795, -0.09753703, -0.047970578, 2.9746156, 0.5887969, 0.40253547, 1.1401464, 3.3779674, -2.757316, 0.053939205, 0.033679023, -1.6412274, 1.5493679, -0.06247123, 0.4622829, -0.18490355, -1.9345231, -0.1239842, 0.11047199, 0.0129180765, 2.1472948, -0.049156453, -0.12152249, -1.9745833, 1.913842, 2.167791, 1.8910753, 0.5978849, -0.85145044, 0.51817995, -0.5811389, -3.103432, -0.31496513, -1.6723366, 0.35055095, 0.4399185, -2.2550843, 0.0075529325, 0.34371766, 0.06300857, -0.0468862, -0.3895241, -0.33452234}))
	z := new(gorgonia.Node)
	 
	o, err := op.Apply(
		x,y,
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
	
	z = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(zT.Shape(), z.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(zT.Data(), z.Value().Data(), 1e-5,"Tensors should be the same")
	
}

// TestMul_bcast is autogenerated from test_mul_bcast
func TestMul_bcast(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Mul{}

	

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
	
	x := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(3, 4, 5),
			tensor.WithBacking([]float32{1.7640524, 0.4001572, 0.978738, 2.2408931, 1.867558, -0.9772779, 0.95008844, -0.1513572, -0.10321885, 0.41059852, 0.14404356, 1.4542735, 0.7610377, 0.121675014, 0.44386324, 0.33367434, 1.4940791, -0.20515826, 0.3130677, -0.85409576, -2.5529897, 0.6536186, 0.8644362, -0.742165, 2.2697546, -1.4543657, 0.045758516, -0.18718386, 1.5327792, 1.4693588, 0.15494743, 0.37816253, -0.88778573, -1.9807965, -0.34791216, 0.15634897, 1.2302907, 1.2023798, -0.3873268, -0.30230275, -1.048553, -1.420018, -1.7062702, 1.9507754, -0.5096522, -0.4380743, -1.2527953, 0.7774904, -1.6138978, -0.21274029, -0.89546657, 0.3869025, -0.51080513, -1.1806322, -0.028182229, 0.42833188, 0.06651722, 0.3024719, -0.6343221, -0.36274117})),
			gorgonia.WithName("x"))
	
	y := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(5),
			tensor.WithBacking([]float32{-0.67246044, -0.35955316, -0.8131463, -1.7262826, 0.17742614})),
			gorgonia.WithName("y"))
	
	
	zT := tensor.New(
		tensor.WithShape(3, 4, 5),
		tensor.WithBacking([]float32{-1.1862555, -0.14387779, -0.7958572, -3.8684149, 0.3313536, 0.6571807, -0.3416073, 0.12307555, 0.17818491, 0.07285091, -0.0968636, -0.5228886, -0.618835, -0.21004546, 0.07875294, -0.22438279, -0.53720087, 0.16682369, -0.5404433, -0.15153892, 1.7167846, -0.23501062, -0.7029131, 1.2811866, 0.4027138, 0.97800344, -0.016452618, 0.15220787, -2.6460102, 0.26070267, -0.10419602, -0.13596953, 0.7218997, 3.4194145, -0.061728712, -0.105138496, -0.4423549, -0.9777107, 0.66863555, -0.05363641, 0.70511043, 0.51057196, 1.3874474, -3.3675897, -0.090425625, 0.29458764, 0.45044652, -0.6322134, 2.7860436, -0.037745688, 0.6021658, -0.13911203, 0.4153593, 2.0381048, -0.0050002644, -0.28803626, -0.023916475, -0.2459539, 1.0950192, -0.06435977}))
	z := new(gorgonia.Node)
	 
	o, err := op.Apply(
		x,y,
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
	
	z = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(zT.Shape(), z.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(zT.Data(), z.Value().Data(), 1e-5,"Tensors should be the same")
	
}

// TestMul_example is autogenerated from test_mul_example
func TestMul_example(t *testing.T) {
	debug := os.Getenv("SKIP_NOT_IMPLEMENTED")
	skip := true
	if debug == "false" {
		skip = false
	}

	assert := assert.New(t)

	g := gorgonia.NewGraph()
	op := &Mul{}

	

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
	
	x := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(3),
			tensor.WithBacking([]float32{1, 2, 3})),
			gorgonia.WithName("x"))
	
	y := gorgonia.NodeFromAny(g,
		tensor.New(
			tensor.WithShape(3),
			tensor.WithBacking([]float32{4, 5, 6})),
			gorgonia.WithName("y"))
	
	
	zT := tensor.New(
		tensor.WithShape(3),
		tensor.WithBacking([]float32{4, 10, 18}))
	z := new(gorgonia.Node)
	 
	o, err := op.Apply(
		x,y,
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
	
	z = o[0]
	

	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	
	assert.Equal(zT.Shape(), z.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(zT.Data(), z.Value().Data(), 1e-5,"Tensors should be the same")
	
}
